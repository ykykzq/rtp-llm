import json
import os
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from rtp_llm.utils import native_diagnostics as diagnostics


def test_disabled_diagnostics_do_not_read_proc_or_create_files(monkeypatch):
    monkeypatch.delenv("RTP_LLM_NATIVE_DIAGNOSTICS_DIR", raising=False)
    with patch.object(Path, "read_text", side_effect=AssertionError("unexpected I/O")):
        assert diagnostics.capture_native_diagnostics("engine_loaded") is None


def test_capture_preserves_pid_maps_and_exact_library_identity(tmp_path, monkeypatch):
    library = tmp_path / "kv_cache_manager_client.so"
    library.write_bytes(b"test ELF")
    stat = library.stat()
    proc = tmp_path / "proc"
    proc.mkdir()
    maps = (
        f"1000-2000 r-xp 00001000 {os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x} "
        f"{stat.st_ino} {library}\n"
    )
    (proc / "maps").write_text(maps)
    (proc / "stat").write_text("123 (ray worker) S 1 2 3\n")
    monkeypatch.setattr(diagnostics, "PROC_SELF", proc)
    monkeypatch.setenv("RTP_LLM_NATIVE_DIAGNOSTICS_DIR", str(tmp_path / "output"))
    with patch.object(
        diagnostics.subprocess,
        "run",
        return_value=CompletedProcess([], 0, "  Build ID: aabbcc1234\n", ""),
    ) as readelf:
        first = diagnostics.capture_native_diagnostics("engine_loaded")
        second = diagnostics.capture_native_diagnostics("engine_loaded")
    assert first != second
    record = json.loads(first.read_text())
    assert record["pid"] == os.getpid()
    assert record["stage"] == "engine_loaded"
    assert record["proc_stat"] == "123 (ray worker) S 1 2 3\n"
    assert (first.parent / "maps").read_text() == maps
    assert record["libraries"][0]["build_id"] == "aabbcc1234"
    assert record["libraries"][0]["mapped_file_matches"] is True
    assert readelf.call_args.args[0][-1] == str(library)


def test_replaced_binary_is_not_reported_as_loaded_build(tmp_path):
    library = tmp_path / "kv_cache_manager_client.so"
    library.write_bytes(b"new ELF")
    stat = library.stat()
    mapping = (
        f"1000-2000 r-xp 00001000 {os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x} "
        f"{stat.st_ino + 1} {library}\n"
    )
    with patch.object(diagnostics.subprocess, "run") as readelf:
        identity = diagnostics.library_identities(mapping)[0]
    assert identity["mapped_file_matches"] is False
    assert "build_id" not in identity
    readelf.assert_not_called()


def test_collection_failure_does_not_break_worker_startup(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("RTP_LLM_NATIVE_DIAGNOSTICS_DIR", str(tmp_path))
    monkeypatch.setattr(diagnostics, "PROC_SELF", tmp_path / "missing-proc")
    assert diagnostics.capture_native_diagnostics("engine_loaded") is None
    assert "native diagnostics" in caplog.text
