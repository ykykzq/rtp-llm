"""CPU routing regressions; CUDA execution is covered by trtllm_gen_test."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


def _module(name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    return module


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _AttentionBase:
    accepts_fmha_config = False

    @classmethod
    def support_parallelism_config(cls, config):
        return config is None


class _LegacyNativeXQA:
    """Old extensions accepted SM100; the Python guard must also protect them."""

    def __init__(self, config):
        pass

    def support(self, inputs):
        return True


class FlashInferTRTLLMDecodeImpl(_AttentionBase):
    @classmethod
    def support(cls, config, inputs):
        return not inputs.is_prefill

    def __init__(self, config, inputs, parallelism):
        pass


class FlashInferTRTLLMPrefillImpl(FlashInferTRTLLMDecodeImpl):
    @classmethod
    def support(cls, config, inputs):
        return inputs.is_prefill


class SM100AttentionSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only external/native imports are replaced. Execute the entire factory
        # and XQA modules so the real routing and architecture checks are tested.
        root = Path(__file__).resolve().parents[1]
        package = "rtp_llm.models_py.modules.factory.attention"
        cls.arch = _module(
            "rtp_llm.models_py.utils.arch",
            get_sm=lambda: (10, 0),
            is_sm10x=lambda: True,
            is_sm12x=lambda: False,
            get_num_device_sms=lambda: 1,
        )
        cls.device_type = types.SimpleNamespace(Cuda="cuda", ROCm="rocm")
        modules = {
            package: _module(package, common=types.SimpleNamespace()),
            package + ".fmha_impl_base": _module(
                package + ".fmha_impl_base",
                FMHAImplBase=_AttentionBase,
                MlaImplBase=_AttentionBase,
            ),
            cls.arch.__name__: cls.arch,
            "rtp_llm.device.device_type": _module(
                "rtp_llm.device.device_type",
                DeviceType=cls.device_type,
                get_device_type=lambda: "cuda",
            ),
            "rtp_llm.model_loader.model_weight_info": _module(
                "rtp_llm.model_loader.model_weight_info", ModelWeights=object
            ),
            "rtp_llm.ops": _module(
                "rtp_llm.ops",
                AttentionConfigs=object,
                FMHAConfig=object,
                KvCacheDataType=object,
                ParallelismConfig=object,
                RopeStyle=types.SimpleNamespace(Mrope="mrope"),
            ),
            "rtp_llm.ops.compute_ops": _module(
                "rtp_llm.ops.compute_ops",
                PyAttentionInputs=object,
                FusedRopeKVCacheDecodeOp=object,
                LayerKVCache=object,
                XQAAttnOp=_LegacyNativeXQA,
            ),
            "rtp_llm.ops.fused_rope_kvcache_op": _module(
                "rtp_llm.ops.fused_rope_kvcache_op",
                DecodeRopeContractError=type(
                    "DecodeRopeContractError", (RuntimeError,), {}
                ),
            ),
            "rtp_llm.utils.model_weight": _module(
                "rtp_llm.utils.model_weight", W=object
            ),
        }
        cls.modules = patch.dict(sys.modules, modules)
        cls.modules.start()
        cls.addClassCleanup(cls.modules.stop)
        cls.factory = _load("_sm100_factory_test", root / "attn_factory.py")
        cls.xqa = _load("_sm100_xqa_test", root / "cuda_impl/xqa.py")

    def setUp(self):
        self.config = types.SimpleNamespace(
            head_num=10,
            kv_head_num=2,
            size_per_head=128,
            kernel_tokens_per_block=16,
            dtype=torch.bfloat16,
            rope_config=types.SimpleNamespace(style=None),
        )
        self.inputs = types.SimpleNamespace(is_prefill=False)

    def test_bundled_xqa_accepts_only_sm90(self):
        for sm, expected in [
            ((8, 9), False),
            ((9, 0), True),
            ((9, 1), False),
            ((10, 0), False),
            ((10, 3), False),
            ((12, 0), False),
            ((12, 1), False),
        ]:
            with (
                self.subTest(sm=sm),
                patch.object(self.xqa, "get_sm", return_value=sm),
                patch.object(self.xqa, "is_sm12x", return_value=sm[0] == 12),
            ):
                self.assertEqual(
                    self.xqa.XQAImpl.support(self.config, self.inputs), expected
                )

    def test_flashinfer_jit_xqa_keeps_its_separate_sm100_support(self):
        self.assertTrue(self.xqa.XQADecodeImpl.support(self.config, self.inputs))

    def _select(self, is_prefill, enable_xqa, enable_trtllm):
        self.inputs.is_prefill = is_prefill
        fallback = (
            FlashInferTRTLLMPrefillImpl if is_prefill else FlashInferTRTLLMDecodeImpl
        )
        implementations = [fallback] if is_prefill else [self.xqa.XQAImpl, fallback]
        registry = "PREFILL_MHA_IMPS" if is_prefill else "DECODE_MHA_IMPS"
        fmha_config = types.SimpleNamespace(
            enable_xqa=enable_xqa,
            enable_flashinfer_trtllm_gen=enable_trtllm,
        )
        with (
            patch.object(self.factory, registry, implementations),
            patch.object(self.xqa.XQAImpl, "__init__", return_value=None),
        ):
            try:
                selected = self.factory.get_fmha_impl(
                    self.config, None, self.inputs, fmha_config=fmha_config
                )
            finally:
                self.assertEqual(
                    fmha_config.enable_flashinfer_trtllm_gen, enable_trtllm
                )
        return selected

    def test_sm100_selects_enabled_trtllm_for_qwen25_tp4(self):
        for is_prefill in (False, True):
            for enable_xqa in (False, True):
                with self.subTest(is_prefill=is_prefill, enable_xqa=enable_xqa):
                    selected = self._select(is_prefill, enable_xqa, True)
                    expected = (
                        FlashInferTRTLLMPrefillImpl
                        if is_prefill
                        else FlashInferTRTLLMDecodeImpl
                    )
                    self.assertIsInstance(selected, expected)

    def test_sm100_disabled_trtllm_reports_actionable_configuration_error(self):
        for is_prefill in (False, True):
            for enable_xqa in (False, True):
                with (
                    self.subTest(is_prefill=is_prefill, enable_xqa=enable_xqa),
                    self.assertRaisesRegex(
                        RuntimeError, "ENABLE_FLASHINFER_TRTLLM_GEN=True"
                    ),
                ):
                    self._select(is_prefill, enable_xqa, False)


if __name__ == "__main__":
    unittest.main()
