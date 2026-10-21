"""Opt-in cross-node GPU test using the KVCM SDK linked into libth_transformer."""

import hashlib
import json
import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RTP_RUN_NATIVE_KV_TEST") != "1",
    reason="requires two GPU Ray nodes, a KVCM service and its configured storage backend",
)


class GPUProbeWorker:
    def __init__(self, address, group, sdk):
        import torch

        from rtp_llm.ops import ensure_engine_ops_loaded

        ensure_engine_ops_loaded()
        import libth_transformer

        torch.cuda.set_device(0)
        self.native = libth_transformer
        self.address, self.group, self.sdk = address, group, sdk
        self.probe = None
        self.written_key = None

    def open(self, ticket, producer):
        import torch

        from rtp_llm.utils.kvcm_probe import probe_config

        assert self.probe is None
        self.expected = torch.randint(
            0,
            256,
            (ticket["size"],),
            dtype=torch.uint8,
            generator=torch.Generator().manual_seed(ticket["key"]),
        )
        self.buffer = (
            self.expected.cuda()
            if producer
            else torch.zeros_like(self.expected, device="cuda")
        )
        torch.cuda.synchronize()
        self.probe = self.native.NativeKVProbe(
            probe_config(
                self.address,
                self.group,
                ticket["instance_id"],
                ticket["size"],
                self.sdk,
            ),
            self.buffer,
        )

    def put(self, ticket):
        self.open(ticket, True)
        uri = self.probe.put(ticket["key"])
        self.written_key = ticket["key"]
        return {
            **ticket,
            "uri": uri,
            "producer_build": self.native.get_kvcm_build_info(),
        }

    def get(self, ticket):
        import torch

        self.open(ticket, False)
        self.probe.get(ticket["uri"])
        torch.cuda.synchronize()
        actual = self.buffer.cpu()
        assert torch.equal(actual, self.expected), "Native GPU buffer byte mismatch"
        return {
            "verified_bytes": actual.numel(),
            "sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
            "consumer_build": self.native.get_kvcm_build_info(),
        }

    def close(self):
        if self.probe is not None:
            try:
                if self.written_key is not None:
                    self.probe.remove(self.written_key)
            finally:
                self.probe.close()
                self.probe = None
                self.written_key = None
                self.buffer = None


def test_native_gpu_put_get_on_distinct_nodes():
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    connected_here = not ray.is_initialized()
    workers = []
    timeout = int(os.environ.get("RTP_NATIVE_KV_TIMEOUT_S", "180"))
    try:
        if connected_here:
            ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
        nodes = [
            node
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("GPU", 0) >= 1
        ]
        assert len(nodes) >= 2, "Native KV acceptance requires two GPU nodes"
        assert nodes[0]["NodeManagerAddress"] != nodes[1]["NodeManagerAddress"], (
            "Nodes must be on distinct hosts"
        )
        actor = ray.remote(num_gpus=1, num_cpus=1, max_restarts=0)(GPUProbeWorker)
        for node in nodes[:2]:
            workers.append(
                actor.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node["NodeID"], soft=False
                    )
                ).remote(
                    os.environ["KVCM_SERVER_ADDRESS"],
                    os.environ["KVCM_INSTANCE_GROUP"],
                    os.environ.get("KVCM_SDK_CONFIG", '[{"type":"mooncake"}]'),
                )
            )
        for size in (65536, 64 << 20):
            for _ in range(2):
                ticket = {
                    "instance_id": "rtp-probe-" + uuid.uuid4().hex,
                    "key": uuid.uuid4().int & ((1 << 63) - 1),
                    "size": size,
                }
                ticket = ray.get(workers[0].put.remote(ticket), timeout=timeout)
                result = ray.get(workers[1].get.remote(ticket), timeout=timeout)
                assert result["verified_bytes"] == size
                assert result["consumer_build"] == ticket["producer_build"], (
                    "Native builds differ between nodes"
                )
                print(
                    json.dumps(
                        {
                            **ticket,
                            **result,
                            "nodes": [n["NodeManagerAddress"] for n in nodes[:2]],
                        }
                    )
                )
                ray.get(workers[1].close.remote(), timeout=timeout)
                ray.get(workers[0].close.remote(), timeout=timeout)
    finally:
        # Process exit owns cleanup after uncertain native completion; do not
        # free buffers while a timed-out SDK call might still be accessing them.
        for worker in workers:
            ray.kill(worker, no_restart=True)
        if connected_here:
            ray.shutdown()
