"""Immutable SDK artifacts; a release date is not a Mooncake protocol version.

The vendor RPMs do not publish their source commits. The master pin below is
the public KVCM dependency, also matching the RPM's four-argument PutStart ABI.
It is source/schema evidence; GPU cross-node acceptance is still required.
"""

KVCM_CUDA13_X86_VERSION = "2026_09_15_17_40-cuda130.x86_64"
KVCM_CUDA13_X86_SHA256 = "392ac1aadf4ea9e3bad177a69ae8f35c5c1c59888c8505e8bee01cb8fe88d9e0"
KVCM_CUDA13_ARM_VERSION = "2026_09_15_17_39-cuda130.aarch64"
KVCM_CUDA13_ARM_SHA256 = "84e482dfd219df4a7a240061f43627eafb0e79d582bd67228e2d5cac7ba8ffe4"
KVCM_SOURCE_COMMIT = "unknown"
KVCM_MOONCAKE_SOURCE_COMMIT = "unknown"
KVCM_DEPENDENCY_REFERENCE_COMMIT = "b079c1adceb67f858d7d447360e685187e4b9771"
KVCM_MOONCAKE_MASTER_REPOSITORY = "https://github.com/openanolis/Mooncake"
KVCM_MOONCAKE_MASTER_COMMIT = "211b75742b6d1fee739ad9a486f2ae9ce2695847"
KVCM_MOONCAKE_MASTER_SHA256 = "eb3f3f53d873d441cbd04cebd76506b56d7526c805da25b8525ed54abc2a06ba"
