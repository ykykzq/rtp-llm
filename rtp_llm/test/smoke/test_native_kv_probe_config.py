import json
import unittest
from subprocess import CompletedProcess
from unittest.mock import patch

from rtp_llm.utils.kvcm_probe import probe_config, run_probe


class NativeKVProbeConfigTest(unittest.TestCase):
    def test_preflight_preserves_native_failure_and_exit_code(self):
        with patch(
            "rtp_llm.utils.kvcm_probe.subprocess.run",
            return_value=CompletedProcess([], 1, "", "sdk_error_code=105"),
        ):
            with self.assertRaisesRegex(RuntimeError, "exited 1.*sdk_error_code=105"):
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
