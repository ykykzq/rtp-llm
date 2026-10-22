#include <torch/extension.h>
#include <cstdio>
#include <cstdlib>
#include "kvcm_client/meta_client.h"
#include "kvcm_client/transfer_client.h"

namespace rtp_llm {
namespace {

// Shares the SDK and allocator API linked into the engine. The tensor outlives
// both clients and their registered memory; no Python Mooncake client is used.
class NativeKVProbe {
public:
    NativeKVProbe(const std::string& config, at::Tensor buffer): buffer_(std::move(buffer)) {
        // Construction holds the GIL; keep only a native string for calls that
        // release it. Never include SDK credentials or raw storage config.
        namespace py = pybind11;
        const auto json = py::module_::import("json");
        const auto fields = json.attr("loads")(config);
        py::dict context;
        for (const char* key : {"address", "instance_group", "instance_id"}) {
            context[py::str(key)] = fields.attr("get")(key, py::none());
        }
        const char* metrics_port = std::getenv("MASTER_METRICS");
        context["MASTER_METRICS"] = metrics_port && *metrics_port ? metrics_port : "9003";
        context_ = py::str(json.attr("dumps")(context));
        TORCH_CHECK(buffer_.is_cuda() && buffer_.scalar_type() == at::kByte
                    && buffer_.is_contiguous() && buffer_.numel() > 0,
                    "KVCM probe requires a nonempty contiguous CUDA uint8 buffer");
        span_ = {buffer_.data_ptr(), static_cast<size_t>(buffer_.numel())};
        kv_cache_manager::InitParams params{kv_cache_manager::RoleType::HYBRID, &span_, "probe"};
        meta_ = kv_cache_manager::MetaClient::Create(config, params);
        TORCH_CHECK(meta_, "KVCM native probe MetaClient::Create failed; ", context_);
        params.storage_configs = meta_->GetStorageConfig();
        transfer_ = kv_cache_manager::TransferClient::Create(config, params);
        TORCH_CHECK(transfer_, "KVCM native probe TransferClient::Create failed; ", context_);
    }

    std::string put(int64_t key) {
        requireOpen();
        auto [ec, location] = meta_->StartWrite("rtp_probe_allocate", {key}, {}, {"probe"}, 60);
        check(ec, "StartWrite");
        TORCH_CHECK(location.locations.size() == 1 && location.locations[0].size() == 1,
                    "KVCM probe requires one newly allocated block; use a fresh probe key");
        bool transfer_returned = false;
        try {
            auto [write_ec, uris] = transfer_->SaveKvCaches({location.locations[0][0].uri}, buffers());
            transfer_returned = true;
            check(write_ec, "SaveKvCaches");
            TORCH_CHECK(uris.size() <= 1, "KVCM probe returned an invalid URI count");
            if (!uris.empty() && !uris[0].empty()) {
                location.locations[0][0].uri = uris[0];
            }
            check(meta_->FinishWrite("rtp_probe_commit", location.write_session_id,
                                     kv_cache_manager::BlockMaskOffset{1}, location.locations), "FinishWrite");
            return location.locations[0][0].uri;
        } catch (...) {
            if (!transfer_returned) {
                uncertainTransfer("SaveKvCaches", "SDK threw before confirming transfer completion");
            }
            try {
                (void)meta_->FinishWrite("rtp_probe_abort", location.write_session_id,
                                         kv_cache_manager::BlockMaskOffset{0}, {});
            } catch (...) {
                // Preserve the original allocation/transfer error.
            }
            throw;
        }
    }

    void get(const std::string& uri) {
        requireOpen();
        kv_cache_manager::ClientErrorCode ec;
        try {
            ec = transfer_->LoadKvCaches({uri}, buffers());
        } catch (...) {
            uncertainTransfer("LoadKvCaches", "SDK threw before confirming transfer completion");
        }
        check(ec, "LoadKvCaches");
    }

    void remove(int64_t key) {
        requireOpen();
        check(meta_->RemoveCache("rtp_probe_remove", {key}, {}, kv_cache_manager::BlockMaskOffset{0}), "RemoveCache");
    }

    void close() {
        transfer_.reset();
        meta_.reset();
    }

private:
    void requireOpen() const { TORCH_CHECK(meta_ && transfer_, "KVCM probe is closed"); }
    [[noreturn]] void uncertainTransfer(const char* operation, const char* reason) const {
        std::fprintf(stderr, "KVCM native probe %s failed: %s; %s; exiting without releasing DMA buffers\n",
                     operation, reason, context_.c_str());
        std::fflush(stderr);
        std::_Exit(70);
    }
    void check(kv_cache_manager::ClientErrorCode ec, const char* operation) const {
        if (ec == kv_cache_manager::ER_SDK_TIMEOUT) {
            uncertainTransfer(operation, "sdk_error_code=100 (soft timeout)");
        }
        TORCH_CHECK(ec == kv_cache_manager::ER_OK, "KVCM native probe ", operation,
                    " failed: sdk_error_code=", static_cast<int>(ec), "; ", context_,
                    static_cast<int>(ec) == 51
                        ? "; inspect configured Mooncake /health and /metrics capacity; MASTER_METRICS must match "
                          "the master's metrics port (service health is in the Python preflight report)"
                        : "");
    }
    kv_cache_manager::BlockBuffers buffers() {
        return {{{{kv_cache_manager::MemoryType::GPU, buffer_.data_ptr(),
                    static_cast<size_t>(buffer_.numel()), false}}}};
    }
    at::Tensor buffer_;
    std::string context_;
    kv_cache_manager::RegistSpan span_;
    std::unique_ptr<kv_cache_manager::MetaClient> meta_;
    std::unique_ptr<kv_cache_manager::TransferClient> transfer_;
};
}  // namespace

void registerNativeKVProbe(pybind11::module_& m) {
    namespace py = pybind11;
    py::class_<NativeKVProbe>(m, "NativeKVProbe")
        .def(py::init<const std::string&, at::Tensor>())
        .def("put", &NativeKVProbe::put, py::call_guard<py::gil_scoped_release>())
        .def("get", &NativeKVProbe::get, py::call_guard<py::gil_scoped_release>())
        .def("remove", &NativeKVProbe::remove, py::call_guard<py::gil_scoped_release>())
        .def("close", &NativeKVProbe::close, py::call_guard<py::gil_scoped_release>());
}
}  // namespace rtp_llm
