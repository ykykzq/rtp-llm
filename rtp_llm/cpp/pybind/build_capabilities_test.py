import importlib
import os
import subprocess
import sys
import unittest

from rtp_llm.ops import get_build_capabilities


class BuildCapabilitiesTest(unittest.TestCase):
    def test_private_curl_uses_async_dns(self):
        get_build_capabilities()
        compute = importlib.import_module("librtp_compute_ops")
        info = compute.get_curl_build_info()
        self.assertEqual(info["version"], "7.60.0")
        self.assertIs(info["async_dns"], True)
        # Inspect the actual linked modules, including references that could
        # bind to the SDK's or system curl despite a safe RTP resolver build.
        for name in (
            "librtp_compute_ops",
            "libth_transformer",
            "libth_transformer_config",
        ):
            with self.subTest(module=name):
                module = importlib.import_module(name)
                result = subprocess.run(
                    ["readelf", "--dyn-syms", "--wide", module.__file__],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                symbols = [
                    line.split()[7].split("@")[0]
                    for line in result.stdout.splitlines()
                    if len(line.split()) >= 8 and line.split()[0].rstrip(":").isdigit()
                ]
                self.assertFalse(
                    [
                        symbol
                        for symbol in symbols
                        if symbol.startswith(("curl_", "Curl_"))
                    ]
                )

    def test_native_import_and_process_exit(self):
        for extra_import in (False, True):
            with self.subTest(extra_import=extra_import):
                code = "from rtp_llm.ops import get_build_capabilities; print(get_build_capabilities(), flush=True)"
                if extra_import:
                    code += "; import libth_transformer; print(libth_transformer.__file__, flush=True)"
                env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=env,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_remote_cache_matches_the_native_build_selection(self):
        capabilities = get_build_capabilities()
        self.assertIsInstance(capabilities, dict)
        self.assertEqual(capabilities["remote_prefix_write_status"], 1)
        self.assertIs(
            capabilities["remote_kv_cache"],
            os.environ["EXPECT_REMOTE_KV_CACHE"] == "true",
        )


if __name__ == "__main__":
    unittest.main()
