#include <cstdlib>

extern "C" void* curl_version_info(int) {
    // A private RTP curl must never bind to a different library's ABI.
    std::abort();
}

extern "C" int rtp_test_sdk_marker() {
    return 1;
}
