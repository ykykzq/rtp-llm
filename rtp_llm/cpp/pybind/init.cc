#include "rtp_llm/cpp/multimodal_processor/FeatureHashOp.h"
#include "rtp_llm/cpp/multimodal_processor/MultimodalTypes.h"
#include "rtp_llm/cpp/cache/Types.h"
#include "rtp_llm/cpp/cache/KVCacheManager.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpLLMOp.h"
#include "rtp_llm/cpp/pybind/multi_gpu_gpt/RtpEmbeddingOp.h"
#include "rtp_llm/models_py/bindings/OpDefs.h"
#include "rtp_llm/cpp/embedding_engine/EmbeddingQuery.h"
#include "pybind11/pybind11.h"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

namespace rtp_llm {
void registerNativeKVProbe(pybind11::module_& m);

void registerEmbeddingOutput(const py::module& m) {
    py::class_<TypedOutput>(m, "TypedOutput")
        .def(py::init<>())
        .def_readwrite("isTensor", &TypedOutput::isTensor)
        .def_property(
            "t",
            [](const TypedOutput& self) -> py::object {
                return self.t.has_value() ? py::cast(self.t.value()) : py::none();
            },
            [](TypedOutput& self, const at::Tensor& tensor) { self.setTensorOuput(tensor); })
        .def_property(
            "map",
            [](const TypedOutput& self) -> py::object {
                return self.map.has_value() ? py::cast(self.map.value()) : py::none();
            },
            // FIX: Take by value instead of const reference
            [](TypedOutput& self, std::vector<std::map<std::string, at::Tensor>> map_val) {
                self.setMapOutput(map_val);
            });

    py::class_<EmbeddingOutput>(m, "EmbeddingCppOutput")
        .def(py::init<>())
        .def_readwrite("output", &EmbeddingOutput::output)
        .def("setTensorOutput", &EmbeddingOutput::setTensorOutput)
        .def("setMapOutput", &EmbeddingOutput::setMapOutput);
}

PYBIND11_MODULE(libth_transformer, m) {
    registerNativeKVProbe(m);
    m.def("get_kvcm_build_info", []() {
        py::dict info;
        info["remote_kv_cache"] = KVCacheManager::remoteCacheCompiled();
        info["sdk_version"] = RTP_KVCM_SDK_VERSION;
        info["sdk_artifact_sha256"] = RTP_KVCM_SDK_SHA256;
        info["sdk_source_commit"] = RTP_KVCM_SOURCE_COMMIT;
        info["mooncake_source_commit"] = RTP_KVCM_MOONCAKE_SOURCE_COMMIT;
        info["dependency_reference_commit"] = RTP_KVCM_DEPENDENCY_REFERENCE_COMMIT;
        info["mooncake_master_commit"] = RTP_KVCM_MOONCAKE_MASTER_COMMIT;
        info["compatibility_evidence"] = "upstream_pin_and_rpc_schema; GPU acceptance required";
        return info;
    });
    m.def("get_build_capabilities", []() {
        py::dict capabilities;
        capabilities["remote_kv_cache"] = KVCacheManager::remoteCacheCompiled();
        return capabilities;
    });
    registerRtpLLMOp(m);
    registerRtpEmbeddingOp(m);
    registerEmbeddingOutput(m);
    m.def("get_multimodal_feature_hash", &getMultimodalFeatureHash, py::call_guard<py::gil_scoped_release>());
}

}  // namespace rtp_llm
