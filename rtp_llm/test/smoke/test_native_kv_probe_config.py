import json
import subprocess
import sys
import unittest
from subprocess import CompletedProcess
from unittest.mock import patch

from rtp_llm.utils.kvcm_probe import probe_config, run_probe


class NativeKVProbeConfigTest(unittest.TestCase):
    def test_native_failures_include_probe_context_and_actual_service_health(self):
        with (
            patch(
                "rtp_llm.utils.kvcm_probe.subprocess.run",
                return_value=CompletedProcess(
                    [], 1, "", "StartWrite sdk_error_code=51"
                ),
            ),
            self.assertRaises(RuntimeError) as error,
        ):
            run_probe("kvcm:6381", "rollout-group", '[{"type":"mooncake"}]')
        for expected in (
            "kvcm:6381",
            "rollout-group",
            "instance_id",
            "service_health",
            "not_checked",
            "sdk_error_code=51",
        ):
            self.assertIn(expected, str(error.exception))

    def test_constructor_and_cleanup_failures_terminate_without_unwinding_buffers(self):
        # Only CUDA/native imports are replaced: the real probe control flow
        # runs in a child, so normal Python unwinding is observable as exit 8.
        source = """
import sys, types
from rtp_llm.utils.kvcm_probe import _run_probe
class Buffer:
    def to(self, **kwargs): return self
class Generator:
    def manual_seed(self, seed): return self
class Probe:
    def __init__(self, *args):
        if sys.argv[1] == "construct": raise RuntimeError("sdk_error_code=51")
    def put(self, key):
        if sys.argv[1] == "put": raise RuntimeError("sdk_error_code=51")
        return "mooncake://probe"
    def close(self): raise RuntimeError("close failure")
sys.modules["torch"] = types.SimpleNamespace(
    cuda=types.SimpleNamespace(set_device=lambda d: None, synchronize=lambda d: None),
    randint=lambda *args, **kwargs: Buffer(), uint8="uint8", Generator=Generator)
sys.modules["rtp_llm.ops"] = types.SimpleNamespace(ensure_engine_ops_loaded=lambda: None)
sys.modules["libth_transformer"] = types.SimpleNamespace(NativeKVProbe=Probe, get_kvcm_build_info=lambda: {})
try:
    _run_probe("kvcm:6381", "rollout-group", '[{"type":"mooncake"}]', phase="put")
except BaseException:
    print("unsafe normal unwind", flush=True)
    sys.exit(8)
"""
        for operation in ("construct", "put", "close"):
            with self.subTest(operation=operation):
                result = subprocess.run(
                    [sys.executable, "-c", source, operation],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertNotIn("unsafe normal unwind", result.stdout)
                for expected in (
                    "kvcm:6381",
                    "rollout-group",
                    "instance_id",
                    "operation",
                ):
                    self.assertIn(expected, result.stderr)

    def test_preflight_preserves_native_failure_and_exit_code(self):
        with (
            patch(
                "rtp_llm.utils.kvcm_probe.subprocess.run",
                return_value=CompletedProcess([], 1, "", "sdk_error_code=105"),
            ),
            self.assertRaisesRegex(RuntimeError, "exited 1.*sdk_error_code=105"),
        ):
            run_probe("kvcm:6380", "gpu-probes", '[{"type":"mooncake"}]')

    def test_preflight_requires_byte_equality_evidence_from_child(self):
        for stdout in (
            '{"verified_bytes": 65536}\n',
            'native logging\n{"verified_bytes": 0}\n',
        ):
            with (
                self.subTest(stdout=stdout),
                patch(
                    "rtp_llm.utils.kvcm_probe.subprocess.run",
                    return_value=CompletedProcess([], 0, stdout, ""),
                ) as child,
            ):
                if "65536" in stdout:
                    self.assertEqual(
                        run_probe("kvcm:6380", "gpu-probes", "[]")["verified_bytes"],
                        65536,
                    )
                else:
                    with self.assertRaisesRegex(
                        RuntimeError, "no byte-equality evidence"
                    ):
                        run_probe("kvcm:6380", "gpu-probes", "[]")
                self.assertEqual(child.call_args.kwargs["timeout"], 90)

    def test_probe_uses_explicit_mooncake_backend_and_gpu_block_spec(self):
        config = json.loads(
            probe_config(
                "kvcm:6380", "gpu-probes", "unique", 65536, '[{"type":"mooncake"}]'
            )
        )
        self.assertEqual(config["address"], ["kvcm:6380"])
        self.assertEqual(config["instance_group"], "gpu-probes")
        self.assertEqual(config["location_spec_infos"], {"probe": 65536})
        self.assertEqual(config["location_spec_groups"], {"probe": ["probe"]})
        self.assertEqual(
            config["sdk_config"]["sdk_backend_configs"], [{"type": "mooncake"}]
        )

    def test_probe_rejects_missing_storage_backend(self):
        for sdk in ("[]", "{}", "null"):
            with (
                self.subTest(sdk=sdk),
                self.assertRaisesRegex(ValueError, "nonempty JSON list"),
            ):
                probe_config("kvcm:6380", "gpu-probes", "unique", 65536, sdk)


if __name__ == "__main__":
    unittest.main()
