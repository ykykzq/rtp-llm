"""Check the linked ELF, including curl's public API and private globals."""

import ctypes
import subprocess
import sys
from pathlib import Path


def main():
    path = Path(sys.argv[1]).resolve()
    conflicting = Path(sys.argv[2]).resolve()
    if len(sys.argv) == 4:
        order = (
            [conflicting, path]
            if sys.argv[3] == "conflict-first"
            else [path, conflicting]
        )
        libraries = {
            file: ctypes.CDLL(str(file), mode=ctypes.RTLD_GLOBAL) for file in order
        }
        probe = libraries[path].rtp_test_curl_features
        probe.restype = ctypes.c_int
        assert probe() & (1 << 7), "CURL_VERSION_ASYNCHDNS is missing"
        return
    symbols = subprocess.run(
        ["readelf", "--dyn-syms", "--wide", str(path)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    names = [
        line.split()[7].split("@")[0]
        for line in symbols.splitlines()
        if len(line.split()) >= 8 and line.split()[0].rstrip(":").isdigit()
    ]
    leaked = [name for name in names if name.startswith(("curl_", "Curl_"))]
    assert not leaked, f"private curl can bind to another DSO: {leaked}"
    for order in ("conflict-first", "rtp-first"):
        subprocess.run(
            [sys.executable, __file__, str(path), str(conflicting), order],
            check=True,
            timeout=30,
        )


if __name__ == "__main__":
    main()
