#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm {

struct RemoteCacheStatus {
    std::string status{"success"};
    int error_code{0};
    std::string message;
    std::vector<uint64_t> write_ids;
    size_t prefix_tokens{0};
    RemoteCacheStatus() = default;
    RemoteCacheStatus(std::string state, int code, std::string detail):
        status(std::move(state)), error_code(code), message(std::move(detail)) {}
};

// CPU-only completion metadata. This class never owns or releases GPU buffers.
// Captures hold immutable write identities even when retries replace the index.
class RemoteWriteLedger {
public:
    using Key = std::pair<int64_t, std::string>;
    struct Record {
        uint64_t id;
        bool done{false};
        RemoteCacheStatus result;
    };
    struct Ticket {
        std::shared_ptr<Record> record;
        bool admitted{false};
    };
    struct Snapshot {
        std::vector<std::shared_ptr<Record>> records;
        bool complete{false};
    };

    explicit RemoteWriteLedger(size_t max_entries = 65536, size_t max_pending = 64):
        max_entries_(max_entries), max_pending_(max_pending) {}

    Ticket begin(const std::vector<Key>& keys) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto record = std::make_shared<Record>();
        record->id = ++sequence_;
        const bool admitted = pending_ < max_pending_ && !keys.empty() && keys.size() <= max_entries_;
        if (admitted) {
            ++pending_;
        } else {
            record->done = true;
            record->result = {"write_failed", -2, "remote write admission limit reached"};
        }
        for (const auto& key : keys) {
            auto existing = latest_.find(key);
            if (existing != latest_.end()) {
                order_.erase(existing->second.order);
                latest_.erase(existing);
            }
            // Bound metadata without expiring active snapshots or their results.
            for (auto it = order_.begin(); latest_.size() >= max_entries_ && it != order_.end();) {
                if (latest_.at(*it).record->done) {
                    latest_.erase(*it);
                    it = order_.erase(it);
                } else {
                    ++it;
                }
            }
            if (latest_.size() < max_entries_) {
                order_.push_back(key);
                latest_.emplace(key, Entry{record, std::prev(order_.end())});
            }
        }
        return {record, admitted};
    }

    void complete(const Ticket& ticket, RemoteCacheStatus result) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!ticket.record || ticket.record->done) {
            return;
        }
        ticket.record->result = std::move(result);
        ticket.record->done = true;
        --pending_;
        cv_.notify_all();
    }

    Snapshot capture(const std::vector<Key>& keys) {
        std::lock_guard<std::mutex> lock(mutex_);
        Snapshot snapshot;
        snapshot.complete = !keys.empty();
        std::map<uint64_t, std::shared_ptr<Record>> unique;
        for (const auto& key : keys) {
            auto it = latest_.find(key);
            if (it == latest_.end()) {
                snapshot.complete = false;
            } else {
                unique.emplace(it->second.record->id, it->second.record);
            }
        }
        for (const auto& [id, record] : unique) {
            snapshot.records.push_back(record);
        }
        return snapshot;
    }

    RemoteCacheStatus wait(const Snapshot& snapshot, int64_t timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!snapshot.complete) {
            return {"unavailable", -1, "required prefix writes are missing or their status expired"};
        }
        auto settled = [&] {
            bool all_done = true;
            for (const auto& record : snapshot.records) {
                // A known failed dependency already rules out this prefix;
                // pending siblings continue owning their I/O and source pins.
                if (record->done && record->result.status != "success") {
                    return true;
                }
                all_done = all_done && record->done;
            }
            return all_done;
        };
        if (timeout_ms < 0) {
            cv_.wait(lock, settled);
        } else if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), settled)) {
            return {"timeout", 0, "required prefix writes are still pending"};
        }
        RemoteCacheStatus result;
        for (const auto& record : snapshot.records) {
            if (result.status == "success" && record->result.status != "success") {
                result = record->result;
            }
        }
        for (const auto& record : snapshot.records) {
            result.write_ids.push_back(record->id);
        }
        return result;
    }

private:
    struct Entry {
        std::shared_ptr<Record> record;
        std::list<Key>::iterator order;
    };
    const size_t max_entries_, max_pending_;
    uint64_t sequence_{0};
    size_t pending_{0};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::map<Key, Entry> latest_;
    std::list<Key> order_;
};

}  // namespace rtp_llm
