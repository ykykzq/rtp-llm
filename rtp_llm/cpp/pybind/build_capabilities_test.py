import os
import unittest

from rtp_llm.ops import get_build_capabilities


class BuildCapabilitiesTest(unittest.TestCase):
    def test_remote_cache_matches_the_native_build_selection(self):
        capabilities = get_build_capabilities()
        self.assertIsInstance(capabilities, dict)
        self.assertIs(
            capabilities["remote_kv_cache"],
            os.environ["EXPECT_REMOTE_KV_CACHE"] == "true",
        )


if __name__ == "__main__":
    unittest.main()
