# RTP private curl resolver

The 2026-10-29 TP4 restore crash resolved to `hostip.c:alarmfunc` in
`librtp_compute_ops.so`. Curl 7.60's synchronous DNS timeout uses a process-wide
`SIGALRM` and a global `curl_jmpenv`. It is unsafe in a multi-threaded Ray worker.
The executing handler was identified; the initiating easy handle and its
runtime symbol bindings were not captured.

`curl.BUILD` uses the POSIX threaded resolver, including `asyn-thread.c`, pthread
headers and compile/link flags. It omits the alarm/setjmp configuration. DNS
timeouts can return while the resolver thread finishes independently. The
existing publisher's `CURLOPT_NOSIGNAL=1` remains useful but was already present
in the failing build.

The static curl target compiles with hidden visibility. `CURL_STATICLIB` makes
curl 7.60's public declarations honor that visibility as well. `alwayslink = True`
also forces the private archive into the final link: an SDK DSO earlier on the
link command can otherwise satisfy the curl references before the linker reaches
the archive. Hidden visibility alone did not prevent this; the affected wheel's
`libth_transformer.so` retained 18 undefined `curl_*` symbols. Together these
settings keep RTP's curl functions and data from resolving against the SDK/system
curl, or being used to resolve their references. They do not hide unrelated RTP
exports.

## Native regression and wheel acceptance

Run in the matching Linux build environment, preserving the existing CUDA
toolchain overrides:

```bash
bazel test --config=cuda13_remote_cache \
  //3rdparty/curl:resolver_test \
  //3rdparty/curl:symbol_isolation_test \
  //3rdparty/curl:sdk_linked_symbol_isolation_test \
  //rtp_llm/cpp/pybind:build_capabilities_test
```

`resolver_test` links the real `@curl` target and injects DNS failures and blocked
lookups at `getaddrinfo`. Sixteen concurrent callers recreate easy handles for
four rounds, exercising `NOSIGNAL=0` and `1`. Requests must time out before DNS
is released, and the process's `SIGALRM` handler must remain unchanged. The test
keeps fortify enabled and has an independent deadline.

`symbol_isolation_test` checks the linked ELF's public and private curl symbols
and exercises both load orders with a conflicting global curl implementation.
`sdk_linked_symbol_isolation_test` applies the same checks to a probe linked with
the conflicting SDK DSO before the private curl archive. A separate SDK marker
keeps the DSO in `DT_NEEDED`, covering archive extraction as well as runtime
interposition. Both tests require no public or undefined `curl_*` / `Curl_*`
symbols and require the private implementation to report asynchronous DNS.
`build_capabilities_test` checks the actual RTP modules, queries the compute
library's runtime `CURL_VERSION_ASYNCHDNS`, and retains the fresh-process import
and exit checks. After building and installing the wheel on **both** nodes:

```bash
EXPECT_REMOTE_KV_CACHE=true python rtp_llm/cpp/pybind/build_capabilities_test.py
```

The 2026-10-29 Linux validation handoff recorded the SDK-linked regression failing
before `alwayslink`, then all three curl tests passing after it. The rebuilt
wheel's three capability/isolation tests passed on both validation nodes. Those
are handed-off results, not a new Linux build performed during local integration;
macOS native checks cannot substitute for the Bazel/ELF and installed-wheel checks.

The native resolver regression is a focused transport test, not proof of TP4
KV migration or resumed RL. Preserve the existing checkpoint 5, use a new output
directory, and rerun the exact phase-2 configuration from the validation manifest.
Require restored optimizer/RNG/scheduler state, real updates 6-10 with nonzero
parameter changes, remote KV reuse, checkpoint 10, final evaluation, and step-7
profiles. A standalone SDK URI put/get probe does not cover the TP4
`GetReplicaList` path. Keep `RTP_LLM_NATIVE_DIAGNOSTICS_DIR` enabled.

## SDK boundary

`kv_cache_manager_client.so` is a separately supplied binary. This change does
not rebuild its curl, establish its DNS behavior, or isolate it from system
curl. Inspect the exact installed SDK's `DT_NEEDED`, defined/undefined curl
symbols and build ID. If `SIGALRM` is still registered during resumed KV reads,
capture the caller/TID, handler owner, and easy-handle options; an SDK-owned
signal resolver needs a corresponding SDK rebuild or provider fix. Do not
disable fortify or mask the resulting abort.

TCP KV descriptors and a successful separate RDMA probe also do not establish
that the training run used RDMA.
