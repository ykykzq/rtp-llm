"""GPU-buffer acceptance through the exact KVCM SDK linked into RTP-LLM."""

import argparse
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import traceback
import uuid

from rtp_llm.utils.kvcm_service_preflight import (
    ServicePreflightError,
    service_preflight,
    service_settings,
)


def probe_config(address, instance_group, instance_id, size, sdk_config):
    if not address or not instance_group or size <= 0:
        raise ValueError(
            "KVCM probe requires an address, instance group and positive buffer size"
        )
    backends = json.loads(sdk_config)
    if not isinstance(backends, list) or not backends:
        raise ValueError("KVCM SDK config must be a nonempty JSON list")
    return json.dumps(
        {
            "instance_group": instance_group,
            "instance_id": instance_id,
            "block_size": 1,
            "location_spec_infos": {"probe": size},
            "location_spec_groups": {"probe": ["probe"]},
            "address": [address],
            "meta_channel_config": {
                "retry_time": 1,
                "connection_timeout": 5000,
                "call_timeout": 5000,
            },
            "sdk_config": {
                "sdk_backend_configs": backends,
                "timeout_config": {"put_timeout_ms": 10000, "get_timeout_ms": 10000},
            },
            "model_deployment": {
                "model_name": "rtp-native-kv-probe",
                "dtype": "uint8",
                "tp_size": 1,
            },
        }
    )


def run_probe(
    address,
    instance_group,
    sdk_config,
    *,
    size=65536,
    device=0,
    timeout_s=90,
    mooncake_master=None,
    master_metrics_port=None,
    storage_config=None,
    min_free_bytes=None,
    service_timeout_s=5,
):
    """Run preflight in a separate process; process exit owns all native buffers."""
    master, port = service_settings(
        mooncake_master, master_metrics_port, storage_config
    )
    instance_id = "rtp-probe-" + uuid.uuid4().hex
    context = {
        "address": address,
        "instance_group": instance_group,
        "instance_id": instance_id,
        "operation": "roundtrip",
        "MASTER_METRICS": port,
        "service_health": {
            "status": "not_checked",
            "reason": "Mooncake master not configured",
        },
    }
    if master:
        context["service_health"] = service_preflight(
            address,
            instance_group,
            mooncake_master=master,
            master_metrics_port=port,
            min_free_bytes=min_free_bytes if min_free_bytes is not None else size,
            timeout_s=service_timeout_s,
            instance_id=instance_id,
        )
    child_env = os.environ.copy()
    child_env["MASTER_METRICS"] = str(port)
    command = [
        sys.executable,
        "-m",
        "rtp_llm.utils.kvcm_probe",
        "--address",
        address,
        "--instance-group",
        instance_group,
        "--sdk-config",
        sdk_config,
        "--size",
        str(size),
        "--device",
        str(device),
        "--instance-id",
        instance_id,
    ]
    if master:
        command.extend(
            [
                "--mooncake-master",
                master,
                "--master-metrics-port",
                str(port),
                "--min-free-bytes",
                str(min_free_bytes if min_free_bytes is not None else size),
                "--service-timeout-s",
                str(service_timeout_s),
            ]
        )
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=child_env,
    )
    if result.returncode:
        raise RuntimeError(
            f"KVCM native storage preflight exited {result.returncode}: "
            f"{json.dumps(context, sort_keys=True)}; {result.stderr[-8192:]}"
        )
    for line in reversed(result.stdout.splitlines()):
        try:
            ticket = json.loads(line)
        except ValueError:
            continue
        if isinstance(ticket, dict) and ticket.get("verified_bytes") == size:
            ticket.setdefault("service_health", context["service_health"])
            return ticket
    raise RuntimeError(
        "KVCM native storage preflight returned no byte-equality evidence"
    )


def _run_probe(
    address,
    instance_group,
    sdk_config,
    *,
    phase="roundtrip",
    ticket=None,
    size=65536,
    device=0,
    hold_for_reader=False,
    instance_id=None,
    service_health=None,
):
    import torch

    from rtp_llm.ops import ensure_engine_ops_loaded

    ensure_engine_ops_loaded()
    import libth_transformer

    if not hasattr(libth_transformer, "NativeKVProbe"):
        raise RuntimeError("Rebuild RTP with the native KVCM GPU probe enabled")
    if phase not in ("roundtrip", "put", "get") or size <= 0:
        raise ValueError("Invalid probe phase or size")
    host = socket.gethostname()
    if phase == "get":
        if not ticket or ticket["source_host"] == host:
            raise ValueError(
                "Cross-node get requires a producer ticket from another host"
            )
        size = ticket["size"]
    else:
        ticket = {
            "source_host": host,
            "size": size,
            "instance_id": instance_id or "rtp-probe-" + uuid.uuid4().hex,
            "key": uuid.uuid4().int & ((1 << 63) - 1),
        }
    torch.cuda.set_device(device)
    expected = torch.randint(
        0,
        256,
        (size,),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(ticket["key"]),
    )
    buffer = (
        expected.to(device=f"cuda:{device}")
        if phase != "get"
        else torch.zeros(size, dtype=torch.uint8, device=f"cuda:{device}")
    )
    torch.cuda.synchronize(device)
    context = {
        "address": address,
        "instance_group": instance_group,
        "instance_id": ticket["instance_id"],
        "phase": phase,
        "MASTER_METRICS": os.environ.get("MASTER_METRICS", "9003"),
        "service_health": service_health or {"status": "not_checked"},
        "operation": "NativeKVProbe::Create",
    }
    # Flush the context before native code, which exits on uncertain DMA completion.
    print(
        "KVCM probe context: " + json.dumps(context, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )
    written = False
    try:
        probe = libth_transformer.NativeKVProbe(
            probe_config(
                address, instance_group, ticket["instance_id"], size, sdk_config
            ),
            buffer,
        )
        if phase != "get":
            context["operation"] = "put (StartWrite/SaveKvCaches/FinishWrite)"
            ticket["uri"] = probe.put(ticket["key"])
            written = True
        if phase != "put":
            context["operation"] = "get (LoadKvCaches)"
            buffer.zero_()
            torch.cuda.synchronize(device)
            probe.get(ticket["uri"])
            torch.cuda.synchronize(device)
            actual = buffer.cpu()
            if not torch.equal(expected, actual):
                raise RuntimeError("KVCM native GPU put/get byte mismatch")
            ticket["destination_host"] = host
            ticket["sha256"] = hashlib.sha256(actual.numpy().tobytes()).hexdigest()
            ticket["verified_bytes"] = size
        ticket["build_info"] = libth_transformer.get_kvcm_build_info()
        ticket["service_health"] = context["service_health"]
        if phase == "put" and hold_for_reader:
            print(json.dumps(ticket), flush=True)
            input("Press Enter only after the remote reader has completed: ")
        if written and (phase == "roundtrip" or hold_for_reader):
            context["operation"] = "remove (RemoveCache)"
            try:
                probe.remove(ticket["key"])
            except Exception:
                logging.exception("KVCM probe cleanup failed for key %s", ticket["key"])
        context["operation"] = "close"
        probe.close()
        return ticket
    except BaseException:  # noqa: BLE001 - process exit must retain uncertain DMA buffers
        # The SDK documents soft Mooncake timeouts: return does not guarantee
        # that DMA stopped. Only this isolated CLI process may own these buffers.
        traceback.print_exc()
        print(
            "KVCM probe failure: " + json.dumps(context, sort_keys=True),
            file=sys.stderr,
        )
        sys.stderr.flush()
        os._exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", required=True)
    parser.add_argument("--instance-group", required=True)
    parser.add_argument("--sdk-config", default='[{"type":"mooncake"}]')
    parser.add_argument(
        "--phase", choices=("put", "get", "roundtrip"), default="roundtrip"
    )
    parser.add_argument("--ticket", help="JSON ticket from the producer's stdout")
    parser.add_argument("--size", type=int, default=65536)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--instance-id")
    parser.add_argument("--service-only", action="store_true")
    parser.add_argument(
        "--mooncake-master",
        help="Configured Mooncake master RPC host:port (or KVCM_MOONCAKE_MASTER)",
    )
    parser.add_argument(
        "--storage-config",
        help="KVCM Mooncake storage JSON path (or KVCM_STORAGE_CONFIG)",
    )
    parser.add_argument(
        "--master-metrics-port",
        type=int,
        help="Must match MASTER_METRICS and the master metrics listener",
    )
    parser.add_argument("--min-free-bytes", type=int)
    parser.add_argument("--service-timeout-s", type=float, default=5)
    args = parser.parse_args()
    ticket = json.loads(args.ticket) if args.ticket else None
    master, port = service_settings(
        args.mooncake_master, args.master_metrics_port, args.storage_config
    )
    health = None
    if master or args.service_only:
        try:
            health = service_preflight(
                args.address,
                args.instance_group,
                mooncake_master=master,
                master_metrics_port=port,
                min_free_bytes=args.min_free_bytes
                if args.min_free_bytes is not None
                else args.size,
                timeout_s=args.service_timeout_s,
                instance_id=args.instance_id,
            )
        except ServicePreflightError as error:
            print(str(error), file=sys.stderr, flush=True)
            raise SystemExit(1) from error
    os.environ["MASTER_METRICS"] = str(port)
    if args.service_only:
        print(json.dumps(health), flush=True)
        return
    print(
        json.dumps(
            _run_probe(
                args.address,
                args.instance_group,
                args.sdk_config,
                phase=args.phase,
                ticket=ticket,
                size=args.size,
                device=args.device,
                hold_for_reader=args.phase == "put",
                instance_id=args.instance_id,
                service_health=health,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
