#include <curl/curl.h>

extern "C" int rtp_test_curl_features() {
    if (curl_global_init(CURL_GLOBAL_DEFAULT) != CURLE_OK) {
        return 0;
    }
    // Pull the easy/resolver objects into the DSO, including their private data.
    CURL* handle = curl_easy_init();
    const int features = handle ? curl_version_info(CURLVERSION_NOW)->features : 0;
    curl_easy_cleanup(handle);
    curl_global_cleanup();
    return features;
}
