"""Build the Mooncake master/client pinned by the KVCM dependency reference."""

import argparse
import ast
import hashlib
import json
import subprocess
import urllib.request
from pathlib import Path


def load_pins():
    source = Path(__file__).resolve().parents[1] / "deps/kvcm_versions.bzl"
    # The lock contains literal assignments; parse them without executing Starlark.
    return {
        statement.targets[0].id: ast.literal_eval(statement.value)
        for statement in ast.parse(source.read_text()).body
        if isinstance(statement, ast.Assign)
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New build directory on a Linux CPU host",
    )
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.jobs <= 0:
        parser.error("--jobs must be positive")
    pins = load_pins()
    commit = pins["KVCM_MOONCAKE_MASTER_COMMIT"]
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    archive = root / "source.tar.gz"
    url = f"{pins['KVCM_MOONCAKE_MASTER_REPOSITORY']}/archive/{commit}.tar.gz"
    with (
        urllib.request.urlopen(url, timeout=60) as response,
        archive.open("wb") as output,
    ):
        while chunk := response.read(1 << 20):
            output.write(chunk)
    hasher = hashlib.sha256()
    with archive.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1 << 20), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest != pins["KVCM_MOONCAKE_MASTER_SHA256"]:
        raise RuntimeError(f"Mooncake archive hash mismatch: {digest}")
    source = root / "source"
    # Archives omit submodules, including pybind11. Fetch the same immutable
    # revision and its gitlinks after verifying the dependency archive pin.
    subprocess.run(["git", "init", str(source)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(source),
            "remote",
            "add",
            "origin",
            pins["KVCM_MOONCAKE_MASTER_REPOSITORY"],
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(source), "fetch", "--depth=1", "origin", commit], check=True
    )
    subprocess.run(
        ["git", "-C", str(source), "checkout", "--detach", commit], check=True
    )
    subprocess.run(
        ["git", "-C", str(source), "submodule", "update", "--init", "--recursive"],
        check=True,
    )
    build = root / "build"
    subprocess.run(
        [
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DUSE_CUDA=OFF",
            "-DWITH_EP=OFF",
            "-DWITH_STORE=ON",
            "-DUSE_HTTP=ON",
            "-DBUILD_UNIT_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
        ],
        check=True,
    )
    subprocess.run(
        [
            "cmake",
            "--build",
            str(build),
            "--target",
            "mooncake_master",
            "mooncake_client",
            "-j",
            str(args.jobs),
        ],
        check=True,
    )
    binaries = {
        name: str(build / "mooncake-store/src" / name)
        for name in ("mooncake_master", "mooncake_client")
    }
    evidence = {
        "master_commit": commit,
        "source_sha256": digest,
        "dependency_reference_commit": pins["KVCM_DEPENDENCY_REFERENCE_COMMIT"],
        "sdk_artifact_sha256": pins["KVCM_CUDA13_X86_SHA256"],
        "native_gpu_acceptance": "not_run",
        "binaries": binaries,
    }
    (root / "build-info.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
