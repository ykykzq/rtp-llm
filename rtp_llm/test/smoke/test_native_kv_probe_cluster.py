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


def _worker_runtime_env():
    return {
        "env_vars": {
            key: os.environ[key]
            for key in (
                "LD_PRELOAD",
                "JE_ARROW_MALLOC_CONF",
                "LD_LIBRARY_PATH",
                "MASTER_METRICS",
                "MC_GID_INDEX",
                "PYTHONPATH",
                "RTP_LLM_NATIVE_DIAGNOSTICS_DIR",
            )
            if key in os.environ
        }
    }


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
        self.open(ticket, False)
        return self.read(ticket)

    def read(self, ticket):
        import torch

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


class ConcurrentGPUProbeWorker:
    def __init__(self, address, group, sdk):
        self.arguments = address, group, sdk
        self.writer = GPUProbeWorker(address, group, sdk)
        self.reader = GPUProbeWorker(address, group, sdk)
        self.threaded_readers = []

    def put(self, ticket):
        return self.writer.put(ticket)

    def read_repeatedly(self, ticket, repeats, threads=1):
        if threads > 1:
            return self.read_threaded(ticket, repeats, threads)
        results = []
        for _ in range(repeats):
            # Recreate the SDK client to exercise both initialization and reads.
            results.append(self.reader.get(ticket))
            self.reader.close()
        return results

    def read_threaded(self, ticket, repeats, threads):
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier

        import torch

        assert not self.threaded_readers
        # Actor-owned references keep registered buffers alive on any failure.
        # The driver kills the actor if an SDK call's completion is uncertain.
        self.threaded_readers = [
            GPUProbeWorker(*self.arguments) for _ in range(threads)
        ]
        for reader in self.threaded_readers:
            reader.open(ticket, False)
        start = Barrier(threads)

        def read(reader):
            torch.cuda.set_device(0)
            start.wait()
            results = []
            for _ in range(repeats):
                reader.buffer.zero_()
                torch.cuda.synchronize()
                results.append(reader.read(ticket))
            return results

        with ThreadPoolExecutor(max_workers=threads) as pool:
            results = list(pool.map(read, self.threaded_readers))
        # Close only after every thread completed and verified all its reads.
        for reader in self.threaded_readers:
            reader.close()
        self.threaded_readers = []
        return [result for reads in results for result in reads]

    def close(self):
        self.writer.close()


@pytest.mark.skipif(
    os.environ.get("RTP_RUN_NATIVE_KV_STRESS") != "1",
    reason="opt-in concurrent native SDK stress test; requires idle GPUs",
)
def test_concurrent_native_gpu_reads_on_all_rollout_ranks():
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    per_node = int(os.environ.get("RTP_NATIVE_KV_WORKERS_PER_NODE", "6"))
    repeats = int(os.environ.get("RTP_NATIVE_KV_READ_REPEATS", "20"))
    threads = int(os.environ.get("RTP_NATIVE_KV_READER_THREADS", "1"))
    timeout = int(os.environ.get("RTP_NATIVE_KV_TIMEOUT_S", "600"))
    assert per_node > 0 and repeats > 0 and threads > 0 and timeout > 0
    connected_here = not ray.is_initialized()
    workers = []
    try:
        if connected_here:
            ray.init(address=os.environ.get("RAY_ADDRESS", "auto"))
        nodes = [
            node
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("GPU", 0) >= per_node
        ]
        assert len(nodes) >= 2, "Concurrent KV acceptance requires two GPU nodes"
        assert nodes[0]["NodeManagerAddress"] != nodes[1]["NodeManagerAddress"]
        runtime_env = _worker_runtime_env()
        actor = ray.remote(num_gpus=1, num_cpus=1, max_restarts=0)(
            ConcurrentGPUProbeWorker
        )
        for node in nodes[:2]:
            for _ in range(per_node):
                workers.append(
                    actor.options(
                        scheduling_strategy=NodeAffinitySchedulingStrategy(
                            node["NodeID"], soft=False
                        ),
                        runtime_env=runtime_env,
                    ).remote(
                        os.environ["KVCM_SERVER_ADDRESS"],
                        os.environ["KVCM_INSTANCE_GROUP"],
                        os.environ.get("KVCM_SDK_CONFIG", '[{"type":"mooncake"}]'),
                    )
                )
        for size in (65536, 64 << 20):
            tickets = ray.get(
                [
                    worker.put.remote(
                        {
                            "instance_id": "rtp-probe-" + uuid.uuid4().hex,
                            "key": uuid.uuid4().int & ((1 << 63) - 1),
                            "size": size,
                        }
                    )
                    for worker in workers
                ],
                timeout=timeout,
            )
            # Pair every reader with a writer on the other node. All writers
            # retain registered GPU buffers until every reader has completed.
            results = ray.get(
                [
                    worker.read_repeatedly.remote(
                        tickets[(index + per_node) % len(workers)], repeats, threads
                    )
                    for index, worker in enumerate(workers)
                ],
                timeout=timeout,
            )
            for index, reads in enumerate(results):
                producer = tickets[(index + per_node) % len(workers)]
                assert len(reads) == repeats * threads
                assert all(result["verified_bytes"] == size for result in reads)
                assert all(
                    result["consumer_build"] == producer["producer_build"]
                    for result in reads
                )
                assert len({result["sha256"] for result in reads}) == 1
            print(
                json.dumps(
                    {
                        "ranks": len(workers),
                        "reader_threads_per_rank": threads,
                        "reads_per_thread": repeats,
                        "reads_per_rank": repeats * threads,
                        "bytes_per_read": size,
                    }
                )
            )
            ray.get([worker.close.remote() for worker in workers], timeout=timeout)
    finally:
        # Do not close or release registered buffers after a native/Ray timeout.
        for worker in workers:
            ray.kill(worker, no_restart=True)
        if connected_here:
            ray.shutdown()


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
        runtime_env = _worker_runtime_env()
        for node in nodes[:2]:
            workers.append(
                actor.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(
                        node["NodeID"], soft=False
                    ),
                    runtime_env=runtime_env,
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
