#include "rtp_llm/cpp/cache/block_tree_cache/storage_backend/RemoteWriteLedger.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <future>
#include <iostream>

using namespace rtp_llm;

int main() {
    using Key = RemoteWriteLedger::Key;
    const Key a{1, "full"}, b{2, "full"}, linear{2, "linear"};
    RemoteWriteLedger ledger(8, 4);
    auto failed = ledger.begin({a});
    auto old = ledger.capture({a});
    ledger.complete(failed, {"write_failed", 55, "quota"});
    auto good = ledger.begin({b, linear});
    ledger.complete(good, {});
    assert(ledger.wait(ledger.capture({b, linear}), 0).status == "success");
    assert(ledger.wait(old, 0).error_code == 55);
    assert(ledger.wait(old, 0).error_code == 55);
    assert(ledger.wait(ledger.capture({a, b, linear}), 0).status == "write_failed");

    auto unrelated = ledger.begin({Key{3, "full"}});
    assert(ledger.wait(ledger.capture({a, Key{3, "full"}}), 0).error_code == 55);
    ledger.complete(unrelated, {});

    auto retry = ledger.begin({a});
    auto current = ledger.capture({a, b, linear});
    assert(ledger.wait(current, 0).status == "timeout");
    auto first = std::async(std::launch::async, [&] { return ledger.wait(current, 5000); });
    auto second = std::async(std::launch::async, [&] { return ledger.wait(current, 5000); });
    ledger.complete(retry, {});
    assert(first.get().status == "success" && second.get().status == "success");
    assert(ledger.wait(old, 0).error_code == 55);

    auto older = ledger.begin({a});
    auto newer = ledger.begin({a});
    ledger.complete(newer, {});
    ledger.complete(older, {"write_failed", 105, "older failure"});
    assert(ledger.wait(ledger.capture({a}), 0).status == "success");
    assert(ledger.wait(ledger.capture({Key{99, "full"}}), 0).status == "unavailable");
    assert(ledger.wait(ledger.capture({}), 0).status == "unavailable");

    RemoteWriteLedger bounded(1, 1);
    auto pinned = bounded.begin({a});
    auto snapshot = bounded.capture({a});
    auto rejected = bounded.begin({b});
    assert(!rejected.admitted);
    assert(bounded.wait(bounded.capture({b}), 0).status != "success");
    bounded.complete(pinned, {});
    auto next = bounded.begin({b});
    bounded.complete(next, {"write_failed", -1, "shutdown rejected"});
    assert(bounded.wait(snapshot, 0).status == "success");
    assert(bounded.wait(bounded.capture({a}), 0).status == "unavailable");
    assert(bounded.wait(bounded.capture({b}), 0).status == "write_failed");
    std::cout << "RemoteWriteLedger tests passed\n";
}
