#include <curl/curl.h>

extern "C" int rtp_test_sdk_marker();

extern "C" int rtp_test_curl_features() {
    // Keep the SDK dependency in DT_NEEDED before the private curl archive.
    if (rtp_test_sdk_marker() != 1) {
        return 0;
    }
    return curl_version_info(CURLVERSION_NOW)->features;
}
