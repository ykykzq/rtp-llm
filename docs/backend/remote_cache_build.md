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

The native library also exposes `get_kvcm_build_info()`, including the pinned
SDK artifact version/SHA256, artifact source provenance, the recommended master
commit and the upstream reference used to select it. CUDA 13 pins live in
`deps/kvcm_versions.bzl`. Public KVCM at
[`b079c1ad`](https://github.com/alibaba/tair-kvcache/blob/b079c1adceb67f858d7d447360e685187e4b9771/open_source/deps/git.bzl#L173)
pins **openanolis/Mooncake `211b75742b6d1fee739ad9a486f2ae9ce2695847`**,
archive SHA256 `eb3f3f53d873d441cbd04cebd76506b56d7526c805da25b8525ed54abc2a06ba`.
The downloaded archive hash was checked locally.

The pinned source's
[`PutStart(client_id, key, slice_length, config)`](https://github.com/openanolis/Mooncake/blob/211b75742b6d1fee739ad9a486f2ae9ce2695847/mooncake-store/include/master_service.h#L197)
matches the demangled symbol in the exact CUDA13 x86 SDK RPM. The deployed
[`v0.3.13.post1` interface](https://github.com/kvcache-ai/Mooncake/blob/v0.3.13.post1/mooncake-store/include/master_service.h#L469)
adds `tenant_id` between `key` and `slice_length`; these are different RPC
schemas. This establishes a concrete incompatibility consistent with the
reported `invalid rpc arg`. A wheel version or the generic Store `2.0.0`
version string cannot identify the protocol. Changing the Python Mooncake
wheel does not replace the client embedded in the SDK.

Use the pinned master for the next acceptance run. This is source/schema
evidence, **not a passed native GPU test**. The SDK RPM's exact build commit and
embedded source commit remain `unknown`; the public dependency reference is
reported separately and must not be represented as binary build provenance.
Build the master and CPU storage client on a Linux host with the pinned
project's CMake/C++20/development dependencies installed:

```bash
python tools/build_kvcm_mooncake.py --output-dir /opt/kvcm-mooncake-211b757 --jobs 8
/opt/kvcm-mooncake-211b757/build/mooncake-store/src/mooncake_master \
  --enable_http_metadata_server=true --http_metadata_server_port=8080
```

The build tool verifies the archive SHA256 and writes `build-info.json`; it
does not install packages, replace a running master or alter the RDMA relay's
Python environment. Start a memory-serving `mooncake_client` built from the
same commit, using its `--host`, `--master_server_address`, `--metadata_server`,
`--global_segment_size`, `--protocol` and `--device_names` settings. Allocate
sufficient registered CPU memory and verify RDMA reachability from all GPU
nodes. The master alone supplies metadata, not KV storage capacity. The CMake
build and service startup still require Linux validation.

## Native Storage Acceptance

Keep model/checkpoint/log artifacts on OSS. Configure the KVCM instance group's
storage as Mooncake, pointing to the matching master and available memory
segments. The client backend list must also be `[{"type":"mooncake"}]`.
The address below is the **KVCM metadata RPC endpoint**, not the Mooncake master.

For the KVCM admin `addStorage` API, the Mooncake storage object uses the fields
below. Substitute reachable service addresses and a unique storage name, then
select that name in the dedicated instance group's `storage_candidates`.
Retain the group's existing quota, reclaim and metadata-index configuration.

```json
{
  "trace_id": "rtp-mooncake-storage",
  "storage": {
    "global_unique_name": "rtp_mooncake_kv",
    "mooncake": {
      "local_hostname": "kvcm-client",
      "metadata_connstring": "http://master-host:8080/metadata",
      "protocol": "rdma",
      "rdma_device": "",
      "master_service_entry": "master-host:50051"
    },
    "check_storage_available_when_open": true
  }
}
```

These are the public KVCM admin protobuf names; specifically the admin field is
`master_service_entry`, while the internal storage JSON uses
`master_server_entry`. Use the schema of the deployed KVCM server. Validate a
single compatible NIC first; an empty device selection requires working
multi-NIC connectivity. This configuration does not route KV through OSS.

```bash
python -m rtp_llm.utils.kvcm_probe \
  --address kvcm-host:rpc-port --instance-group mooncake-probes
RTP_RUN_NATIVE_KV_TEST=1 RAY_ADDRESS=auto \
KVCM_SERVER_ADDRESS=kvcm-host:rpc-port KVCM_INSTANCE_GROUP=mooncake-probes \
python -m pytest rtp_llm/test/smoke/test_native_kv_probe_cluster.py -q -s
```

Both commands use `NativeKVProbe` compiled into `libth_transformer`, the actual
linked SDK and registered CUDA buffers. The first exercises SDK allocation,
write, commit and read; the second keeps the producer alive while another GPU
node reads, repeats 64 KiB/64 MiB transfers and checks every byte. There is no
Python Mooncake storage client substitute. SDK codes, including write error
105 from incompatible FUSE allocation, propagate rather than being ignored.
Startup preflight runs in a child process. During actual engine KV transfers,
SDK `ER_SDK_TIMEOUT=100` exits the affected rank with code 70 before releasing
block pins; the replica must be restarted. The SDK's documented Mooncake
soft timeout does not prove DMA completion, so falling back while reusing those
buffers is unsafe. A cache-flush wait deadline (writes still in flight) remains
a normal structured `timeout` and does not itself terminate a rank.
Use a test instance group configured with the same storage policy as rollout.
Direct two-terminal testing is also available with `--phase put` and
`--phase get --ticket '<producer JSON>'`; keep the producer at its prompt until
the reader completes. Record build info, master build/image/source revision,
both host identities and byte-comparison results for a compatibility record.

The cluster test requires Ray, pytest and two GPU nodes with this rebuilt
wheel. It is opt-in and was not run on the macOS development host. Native tests
for structured write errors and TP propagation are under
`cpp/cache/block_tree_cache/storage_backend/test` and `storage_backend/kvcm/test`.
The capability test now also runs a fresh child interpreter with and without
an explicit second `import libth_transformer`, checking its process exit status
for the reported shutdown failure.

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
