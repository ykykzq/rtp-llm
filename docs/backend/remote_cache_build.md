# Building KVCM support for CUDA 13

`enable_remote_cache=True` requires a native engine built with the KVCM client.
It cannot enable a feature omitted from `libth_transformer.so`. The default
`cuda13` and `cuda13_arm` configurations set `use_remote_kv_cache=false`.
In that build, `KVCacheManager::init()` rejects remote cache requests.

The build capability API loads the native libraries in the supported order but
does not construct an engine, load a checkpoint, or contact a KVCM service:

```python
from rtp_llm.ops import get_build_capabilities

capabilities = get_build_capabilities()
assert capabilities["remote_kv_cache"] is True, capabilities
```

The value comes from the same C++ translation unit that gates the KVCM backend.
An older wheel without the API has unknown support and must be rebuilt before
using the verl `remote_prefix` preflight. Updating Python sources alone does not
update the compiled capability. `True` confirms the compiled feature and native
library load only; service reachability, configuration, storage, and transfer
operation still need runtime validation.

## Linux x86_64 recipe

Use the repository's CUDA 13.2 / Python 3.10 build environment with the matching
PyTorch ABI, Bazelisk, `rpm2cpio`, `cpio`, and `patchelf`. The builder must be able
to fetch the pinned repositories in `deps/http.bzl`. This recipe has not been
validated by a native CUDA build on the macOS development host.

```bash
bazelisk build //rtp_llm:rtp_llm --config=cuda13_remote_cache --verbose_failures
bazelisk test //rtp_llm/cpp/pybind:build_capabilities_test \
  --config=cuda13_remote_cache --test_output=errors
```

The named configuration expands `cuda13` first, then explicitly sets
`--define=use_remote_kv_cache=true`. For an existing build command, place that
define after the CUDA configuration. Avoid relying on a second
`--config=remote_kv_cache`: `cuda13` already expands that config through
`cuda12` and then overrides its value. Inspect Bazel's final configuration.

The dependency graph selects the CUDA 13.0 x86_64 KVCM client RPM rather than the
older default client. `rtp_llm/libs:whl_package_libs` includes the selected KVCM
client and the current x86_64 3FS libraries when remote cache is enabled.
Changing only `RTP_LLM_USE_REMOTE_KV_CACHE` through `--copt` is insufficient:
the Bazel define must also select the dependencies and wheel contents.

Obtain the exact output path with:

```bash
bazelisk cquery //rtp_llm:rtp_llm --config=cuda13_remote_cache --output=files
```

Install that output wheel in the inference environment. Run the capability
assertion above in a fresh process from outside the source checkout, on every
rollout node. Also record the loaded module path to catch a stale wheel or
source-tree shadowing:

```python
import libth_transformer

print(libth_transformer.__file__)
```

Run this after the capability query so the supported native load order is
preserved. Check the installed libraries with `file`, `readelf -d`, and `ldd`
on Linux: they must match the node architecture, have compatible CUDA/ABI
dependencies, and have no unresolved shared libraries. No locally rebuilt
wheel is supplied by this source change.

For the disabled-build regression, run the same Bazel test with
`--config=cuda13`; its expected capability is `False`.

## Linux ARM / GB200 prerequisite

The CUDA 13 ARM selection already points at an aarch64 KVCM client RPM.
However, the remote cache dependency graph also links
`//rtp_llm/cpp/cache:storage_3fs`, whose `@hf3fs_rpm` currently resolves to
`hf3fs-1.3.0-1.alios7.x86_64.rpm`. Both the KVCM client target and cache core
depend on it, and the wheel copies those libraries. Enabling KVCM as-is is
therefore not a complete ARM build recipe.

First provide an ABI-compatible aarch64 3FS package containing all headers and
libraries listed in `3rdparty/3fs/BUILD`. Its extracted layout must contain
`usr/include` and `usr/lib64`. A local Bazel repository override for
`hf3fs_rpm` must expose the package as `@hf3fs_rpm//file:file`, matching the
existing `http_file` interface. Do not substitute or rename an x86_64 binary.
Then, on the native ARM CUDA builder, use:

```bash
bazelisk build //rtp_llm:rtp_llm_aarch64 \
  --config=cuda13_arm --define=use_remote_kv_cache=true \
  --override_repository=hf3fs_rpm=/absolute/path/to/aarch64-hf3fs-repository \
  --verbose_failures
bazelisk test //rtp_llm/cpp/pybind:build_capabilities_test \
  --config=cuda13_arm --define=use_remote_kv_cache=true \
  --override_repository=hf3fs_rpm=/absolute/path/to/aarch64-hf3fs-repository \
  --test_output=errors
```

The ARM package, successful native build, installed-wheel loader checks, and
actual KVCM prefix-transfer test remain required. The opt-in does not certify
them or fall back to token-only migration.
