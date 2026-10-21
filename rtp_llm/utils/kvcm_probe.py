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
    address, instance_group, sdk_config, *, size=65536, device=0, timeout_s=90
):
    """Run preflight in a separate process; process exit owns all native buffers."""
    result = subprocess.run(
        [
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
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode:
        raise RuntimeError(
            f"KVCM native storage preflight exited {result.returncode}: {result.stderr[-8192:]}"
        )
    for line in reversed(result.stdout.splitlines()):
        try:
            ticket = json.loads(line)
        except ValueError:
            continue
        if isinstance(ticket, dict) and ticket.get("verified_bytes") == size:
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
            "instance_id": "rtp-probe-" + uuid.uuid4().hex,
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
    probe = libth_transformer.NativeKVProbe(
        probe_config(address, instance_group, ticket["instance_id"], size, sdk_config),
        buffer,
    )
    written = False
    try:
        if phase != "get":
            ticket["uri"] = probe.put(ticket["key"])
            written = True
        if phase != "put":
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
        if phase == "put" and hold_for_reader:
            print(json.dumps(ticket), flush=True)
            input("Press Enter only after the remote reader has completed: ")
        return ticket
    except BaseException:
        # The SDK documents soft Mooncake timeouts: return does not guarantee
        # that DMA stopped. Only this isolated CLI process may own these buffers.
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(1)
    finally:
        try:
            if written and (phase == "roundtrip" or hold_for_reader):
                try:
                    probe.remove(ticket["key"])
                except Exception:
                    logging.exception(
                        "KVCM probe cleanup failed for key %s", ticket["key"]
                    )
        finally:
            probe.close()


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
    args = parser.parse_args()
    ticket = json.loads(args.ticket) if args.ticket else None
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
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
