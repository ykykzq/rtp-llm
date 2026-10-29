"""CPU checks for thread ownership and failure cleanup in the opt-in GPU probe."""

import importlib.util
import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Keep CPU collection outside the native package and load its helper by path.
_spec = importlib.util.spec_from_file_location(
    "native_kv_probe_cluster",
    Path(__file__).resolve().parents[1]
    / "rtp_llm/test/smoke/test_native_kv_probe_cluster.py",
)
assert _spec is not None and _spec.loader is not None
cluster = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cluster)


@pytest.fixture
def probe_workers(monkeypatch):
    state = SimpleNamespace(
        readers=[],
        cuda_threads=set(),
        read_barrier=None,
        fail_read=False,
        fail_open=False,
    )

    class Buffer:
        def __init__(self):
            self.clears = 0

        def zero_(self):
            self.clears += 1

    class Reader:
        def __init__(self, *args):
            self.index = len(state.readers)
            state.readers.append(self)
            self.buffer = Buffer()
            self.opens = self.closes = self.reads = 0
            self.thread_ids = set()

        def open(self, ticket, producer):
            self.opens += 1
            if state.fail_open and self.index == 3:
                raise RuntimeError("open failure")

        def read(self, ticket):
            self.reads += 1
            self.thread_ids.add(threading.get_ident())
            assert self.buffer.clears == self.reads
            if state.read_barrier:
                # Every reader must reach each round concurrently.
                state.read_barrier.wait(timeout=5)
            if state.fail_read and self.index == 2:
                raise RuntimeError("native read failure")
            return {
                "verified_bytes": ticket["size"],
                "sha256": "matching",
                "consumer_build": "same",
            }

        def get(self, ticket):
            self.open(ticket, False)
            self.buffer.zero_()
            return self.read(ticket)

        def close(self):
            self.closes += 1

    cuda = SimpleNamespace(
        set_device=lambda device: state.cuda_threads.add(threading.get_ident()),
        synchronize=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(cluster, "GPUProbeWorker", Reader)
    state.worker = cluster.ConcurrentGPUProbeWorker("endpoint", "group", "{}")
    state.ticket = {"size": 65536, "uri": "remote-payload"}
    return state


def test_threads_use_independent_clients_and_clear_every_read(probe_workers):
    state = probe_workers
    state.read_barrier = threading.Barrier(4)
    results = state.worker.read_repeatedly(state.ticket, repeats=3, threads=4)
    readers = state.readers[2:]

    assert len(results) == 12
    assert len(readers) == 4
    assert len(state.cuda_threads) == 4
    assert len(set.union(*(reader.thread_ids for reader in readers))) == 4
    assert all(
        (reader.opens, reader.closes, reader.reads, reader.buffer.clears)
        == (1, 1, 3, 3)
        for reader in readers
    )
    assert state.worker.threaded_readers == []
    assert state.worker.writer.closes == 0


def test_read_failure_retains_clients_and_registered_buffers(probe_workers):
    state = probe_workers
    state.read_barrier = threading.Barrier(4)
    state.fail_read = True

    with pytest.raises(RuntimeError, match="native read failure"):
        state.worker.read_repeatedly(state.ticket, repeats=1, threads=4)

    assert state.worker.threaded_readers == state.readers[2:]
    assert all(reader.closes == 0 for reader in state.readers)
    assert all(reader.reads == 1 for reader in state.worker.threaded_readers)


def test_open_failure_retains_already_registered_buffers(probe_workers):
    state = probe_workers
    state.fail_open = True

    with pytest.raises(RuntimeError, match="open failure"):
        state.worker.read_repeatedly(state.ticket, repeats=1, threads=4)

    assert len(state.worker.threaded_readers) == 4
    assert state.worker.threaded_readers[0].opens == 1
    assert all(reader.closes == 0 for reader in state.readers)


def test_default_preserves_serial_client_recreation(probe_workers):
    state = probe_workers
    results = state.worker.read_repeatedly(state.ticket, repeats=3)

    assert len(results) == 3
    assert (state.worker.reader.opens, state.worker.reader.closes) == (3, 3)
    assert len(state.readers) == 2


def test_worker_environment_preserves_explicit_values():
    configured = {
        "LD_PRELOAD": "/compatible/libcurl.so:/compatible/libssl.so",
        "JE_ARROW_MALLOC_CONF": "background_thread:false",
        "MASTER_METRICS": "9004",
    }
    with patch.dict(
        os.environ, {**configured, "UNRELATED_SECRET": "excluded"}, clear=True
    ):
        assert cluster._worker_runtime_env() == {"env_vars": configured}
    with patch.dict(os.environ, {}, clear=True):
        assert cluster._worker_runtime_env() == {"env_vars": {}}
