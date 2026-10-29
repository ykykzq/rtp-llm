#include <cstdlib>

extern "C" void* curl_version_info(int) {
    // A private RTP curl must never bind to a different library's ABI.
    std::abort();
}
