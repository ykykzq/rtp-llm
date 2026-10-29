#include <curl/curl.h>

#include <netdb.h>
#include <signal.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

namespace {
std::atomic<int> entered{0};
std::atomic<int> completed{0};
std::atomic<int> signal_changes{0};
std::atomic<bool> release_slow{false};

void alarm_handler(int) {}

void require(bool condition, const char* message) {
    if (!condition) {
        std::fprintf(stderr, "%s\n", message);
        std::abort();
    }
}

void check_alarm_handler() {
    struct sigaction current {};
    require(sigaction(SIGALRM, nullptr, &current) == 0, "sigaction query failed");
    if (current.sa_handler != alarm_handler) {
        ++signal_changes;
    }
}

void request(bool slow, long no_signal) {
    CURL* handle = curl_easy_init();
    require(handle != nullptr, "curl_easy_init failed");
    require(curl_easy_setopt(handle, CURLOPT_URL, slow ? "http://slow.invalid/" : "http://fail.invalid/") == CURLE_OK,
            "URL option failed");
    require(curl_easy_setopt(handle, CURLOPT_PROXY, "") == CURLE_OK, "proxy option failed");
    require(curl_easy_setopt(handle, CURLOPT_NOSIGNAL, no_signal) == CURLE_OK, "NOSIGNAL option failed");
    require(curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, slow ? 100L : 5000L) == CURLE_OK, "timeout option failed");
    const CURLcode result = curl_easy_perform(handle);
    curl_easy_cleanup(handle);
    require(result == (slow ? CURLE_OPERATION_TIMEDOUT : CURLE_COULDNT_RESOLVE_HOST),
            "unexpected DNS failure/timeout result");
    check_alarm_handler();
}
}  // namespace

// Linked only into this test. Block the real curl resolver at the libc boundary
// so timeout coverage does not depend on an external DNS server or network.
extern "C" int getaddrinfo(const char* name, const char*, const struct addrinfo*, struct addrinfo** result) {
    ++entered;
    *result = nullptr;
    check_alarm_handler();
    if (std::strcmp(name, "slow.invalid") == 0) {
        while (!release_slow.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    ++completed;
    return EAI_NONAME;
}

int main() {
    require(curl_global_init(CURL_GLOBAL_DEFAULT) == CURLE_OK, "curl_global_init failed");
    const auto* info = curl_version_info(CURLVERSION_NOW);
    require((info->features & CURL_VERSION_ASYNCHDNS) != 0, "curl was built with the unsafe synchronous resolver");
    require(info->ares == nullptr, "this regression must exercise the POSIX threaded resolver");
    struct sigaction handler {}, previous {};
    handler.sa_handler = alarm_handler;
    sigemptyset(&handler.sa_mask);
    require(sigaction(SIGALRM, &handler, &previous) == 0, "sigaction installation failed");

    // With a synchronous resolver these requests cannot finish while DNS is
    // blocked. Keep an independent deadline even if curl ignores its timeout.
    std::atomic<bool> finished{false};
    std::thread watchdog([&] {
        for (int i = 0; i < 1000 && !finished.load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        require(finished.load(), "curl did not return while DNS was blocked");
    });
    constexpr int workers = 16;
    constexpr int rounds = 4;
    for (int round = 0; round < rounds; ++round) {
        std::vector<std::thread> threads;
        for (int worker = 0; worker < workers; ++worker) {
            threads.emplace_back([worker] {
                request(false, worker % 2);
                request(true, worker % 2);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }
    release_slow = true;
    while (completed.load() != entered.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    require(entered == workers * rounds * 2, "DNS injection did not cover every request");
    require(signal_changes == 0, "curl changed the process SIGALRM handler");
    check_alarm_handler();
    require(sigaction(SIGALRM, &previous, nullptr) == 0, "sigaction restore failed");
    finished = true;
    watchdog.join();
    // Detached resolver threads can still finish their internal cleanup just
    // after getaddrinfo returns. Do not tear down curl's globals underneath them.
    std::printf("passed: %d DNS failures/timeouts; SIGALRM unchanged\n", entered.load());
    return 0;
}
