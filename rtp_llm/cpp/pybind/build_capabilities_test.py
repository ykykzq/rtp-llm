import os
import subprocess
import sys
import unittest

from rtp_llm.ops import get_build_capabilities


class BuildCapabilitiesTest(unittest.TestCase):
    def test_native_import_and_process_exit(self):
        for extra_import in (False, True):
            with self.subTest(extra_import=extra_import):
                code = "from rtp_llm.ops import get_build_capabilities; print(get_build_capabilities(), flush=True)"
                if extra_import:
                    code += "; import libth_transformer; print(libth_transformer.__file__, flush=True)"
                env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
                result = subprocess.run(
                    [sys.executable, "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env=env,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_remote_cache_matches_the_native_build_selection(self):
        capabilities = get_build_capabilities()
        self.assertIsInstance(capabilities, dict)
        self.assertIs(
            capabilities["remote_kv_cache"],
            os.environ["EXPECT_REMOTE_KV_CACHE"] == "true",
        )


if __name__ == "__main__":
    unittest.main()
