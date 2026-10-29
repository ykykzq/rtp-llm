"""Opt-in loader evidence captured outside native signal handlers."""

import json
import logging
import os
import re
import socket
import subprocess
import tempfile
import time
from pathlib import Path

PROC_SELF = Path("/proc/self")
logger = logging.getLogger(__name__)
_LIBRARIES = (
    "libth_transformer",
    "librtp_compute",
    "kv_cache_manager",
    "mooncake",
    "libcurl",
    "libc.so",
)


def library_identities(maps):
    records = {}
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        _, _, _, device, inode, path = fields
        if path in records or not any(name in Path(path).name for name in _LIBRARIES):
            continue
        record = records[path] = {
            "path": path,
            "mapped_device": device,
            "mapped_inode": int(inode),
        }
        try:
            stat = Path(path).stat()
            major, minor = (int(part, 16) for part in device.split(":"))
            record.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            record["mapped_file_matches"] = stat.st_ino == int(inode) and (
                os.major(stat.st_dev),
                os.minor(stat.st_dev),
            ) == (major, minor)
            if not record["mapped_file_matches"]:
                continue
            result = subprocess.run(
                ["readelf", "--notes", "--wide", "--", path],
                capture_output=True,
                text=True,
                check=False,
                timeout=5,
            )
            match = re.search(r"Build ID:\s*([0-9a-fA-F]+)", result.stdout)
            record["build_id"] = match.group(1) if match else None
            if result.returncode:
                record["build_id_error"] = result.stderr[-512:]
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            record["identity_error"] = str(error)
    return list(records.values())


def capture_native_diagnostics(stage):
    """Return an evidence JSON path, or None when disabled/unavailable.

    Use process-local maps and inode-matched ELF build IDs; maps from a different
    Ray worker or a previous process cannot resolve ASLR addresses reliably.
    """
    output = os.environ.get("RTP_LLM_NATIVE_DIAGNOSTICS_DIR")
    if not output:
        return None
    try:
        maps = (PROC_SELF / "maps").read_text()
        record = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "stage": stage,
            "captured_at_ns": time.time_ns(),
            "proc_stat": (PROC_SELF / "stat").read_text(),
            "libraries": library_identities(maps),
        }
        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        directory = Path(
            tempfile.mkdtemp(prefix=f"{record['host']}-{record['pid']}-", dir=root)
        )
        (directory / "maps").write_text(maps)
        destination = directory / "identity.json"
        destination.write_text(json.dumps(record, indent=2) + "\n")
        logger.info(
            "RTP native diagnostics pid=%s stage=%s path=%s",
            record["pid"],
            stage,
            destination,
        )
        return destination
    except (OSError, ValueError) as error:
        logger.warning(
            "Could not capture RTP native diagnostics at %s: %s", stage, error
        )
        return None
