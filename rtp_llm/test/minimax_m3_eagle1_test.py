import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rtp_llm.models.minimax_m3_eagle1 import (
    MiniMaxM3Eagle1,
    _external_lm_head_path,
    _load_external_lm_head,
)
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeDecoderLayer
from rtp_llm.models_py.model_desc.minimax_m3 import (
    MiniMaxM3DecoderLayer,
    _MiniMaxM3ModelMixin,
)
from rtp_llm.models_py.model_desc.minimax_m3_eagle1 import MiniMaxM3Eagle1Model
from rtp_llm.models_py.modules.hybrid.msa_attention import (
    _repeat_request_block_table_for_verify_tokens,
)


class EagleConfigTest(unittest.TestCase):
    def test_rejects_multi_layer_hass_checkpoint(self):
        with TemporaryDirectory() as tmpdir:
            config = {
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "hidden_size": 8,
                "num_hidden_layers": 2,
                "vocab_size": 32,
            }
            with open(Path(tmpdir) / "config.json", "w") as writer:
                json.dump(config, writer)

            with self.assertRaisesRegex(ValueError, "exactly one draft layer"):
                MiniMaxM3Eagle1._create_config(tmpdir)


class EagleExternalLmHeadTest(unittest.TestCase):
    def test_loads_lm_head_from_bundle_assets_sibling(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ckpt = root / "draft_model"
            assets = root / "assets"
            ckpt.mkdir()
            assets.mkdir()
            expected = torch.randn(3, 4, dtype=torch.bfloat16)
            torch.save(expected, assets / "lm_head.pt")

            self.assertEqual(
                _external_lm_head_path(str(ckpt)), str(assets / "lm_head.pt")
            )
            actual = _load_external_lm_head([], ckpt_path=str(ckpt))
            torch.testing.assert_close(actual, expected)

    def test_rejects_missing_lm_head(self):
        with TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "draft_model"
            ckpt.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "external lm_head"):
                _load_external_lm_head([], ckpt_path=str(ckpt))


class EagleFcInputTest(unittest.TestCase):
    def test_hass_input_normalizes_embedding_and_hidden_before_projection(self):
        draft = SimpleNamespace(
            hidden_size=4,
            embedding_norm=lambda value: value + 1,
            hidden_norm=lambda value: value * 2,
        )
        embedding = torch.randn(2, 4)
        hidden = torch.randn(2, 4)

        actual = MiniMaxM3Eagle1Model._build_fc_input(draft, embedding, hidden)

        torch.testing.assert_close(
            actual, torch.cat([embedding + 1, hidden * 2], dim=-1)
        )

    def test_hass_input_rejects_wrong_target_hidden_width(self):
        draft = SimpleNamespace(
            hidden_size=4,
            embedding_norm=lambda value: value,
            hidden_norm=lambda value: value,
        )
        with self.assertRaisesRegex(RuntimeError, "HASS draft expected target hidden"):
            MiniMaxM3Eagle1Model._build_fc_input(
                draft, torch.randn(2, 4), torch.randn(2, 5)
            )


class DecoderAttentionHookTest(unittest.TestCase):
    def test_generic_decoder_keeps_causal_attention_call_contract(self):
        class FakeCausalAttention:
            def __init__(self):
                self.qkv_proj = object()
                self.kwargs = None

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return kwargs["hidden_states"] + 1

        layer = object.__new__(GenericMoeDecoderLayer)
        torch.nn.Module.__init__(layer)
        attention = FakeCausalAttention()
        layer.self_attn = attention
        hidden_states = torch.randn(2, 4)
        fp8_states = torch.randn(2, 4)
        fp8_scale = torch.randn(2, 1)

        with patch(
            "rtp_llm.models_py.model_desc.generic_moe.CausalAttention",
            FakeCausalAttention,
        ):
            self.assertIs(layer._input_quant_projection(), attention.qkv_proj)
            actual, topk = layer._forward_attention(
                hidden_states,
                fmha_impl="fmha",
                kv_cache="kv_cache",
                prev_topk_indices=None,
                force_reuse_topk_indices=False,
                attn_inputs="unused",
                x_fp8=fp8_states,
                x_scale=fp8_scale,
            )

        torch.testing.assert_close(actual, hidden_states + 1)
        self.assertIsNone(topk)
        self.assertEqual(attention.kwargs["fmha_impl"], "fmha")
        self.assertEqual(attention.kwargs["kv_cache"], "kv_cache")
        self.assertIs(attention.kwargs["x_fp8"], fp8_states)
        self.assertIs(attention.kwargs["x_scale"], fp8_scale)

    def test_minimax_decoder_owns_msa_attention_call_contract(self):
        class FakeMSAAttention:
            def __init__(self):
                self.qkv_proj = object()
                self.kwargs = None

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return kwargs["hidden_states"] + 2

        layer = object.__new__(MiniMaxM3DecoderLayer)
        torch.nn.Module.__init__(layer)
        attention = FakeMSAAttention()
        layer.self_attn = attention
        hidden_states = torch.randn(2, 4)
        fp8_states = torch.randn(2, 4)
        fp8_scale = torch.randn(2, 1)
        attention_inputs = object()

        with patch(
            "rtp_llm.models_py.model_desc.minimax_m3.MSAAttention",
            FakeMSAAttention,
        ):
            self.assertIs(layer._input_quant_projection(), attention.qkv_proj)
            actual, topk = layer._forward_attention(
                hidden_states,
                fmha_impl="unused",
                kv_cache="kv_cache",
                prev_topk_indices=None,
                force_reuse_topk_indices=False,
                attn_inputs=attention_inputs,
                x_fp8=fp8_states,
                x_scale=fp8_scale,
            )

        torch.testing.assert_close(actual, hidden_states + 2)
        self.assertIsNone(topk)
        self.assertIs(attention.kwargs["attn_inputs"], attention_inputs)
        self.assertEqual(attention.kwargs["kv_cache"], "kv_cache")
        self.assertIs(attention.kwargs["x_fp8"], fp8_states)
        self.assertIs(attention.kwargs["x_scale"], fp8_scale)
        self.assertNotIn("fmha_impl", attention.kwargs)


class TargetVerifyAttentionInputsTest(unittest.TestCase):
    def test_expands_request_metadata_for_multi_token_decode(self):
        inputs = SimpleNamespace(
            is_target_verify=True,
            prefix_lengths=torch.tensor([10, 20], dtype=torch.int32),
            total_tokens=6,
            is_prefill=True,
            sequence_lengths_plus_1_d=None,
            decode_cu_seqlens_d=None,
            cu_seqlens=None,
        )

        actual = _MiniMaxM3ModelMixin.prepare_target_verify_attention_inputs(
            None, inputs
        )

        torch.testing.assert_close(
            actual.sequence_lengths_plus_1_d,
            torch.tensor([11, 12, 13, 21, 22, 23], dtype=torch.int32),
        )
        torch.testing.assert_close(
            actual.decode_cu_seqlens_d, torch.tensor([0, 3, 6], dtype=torch.int32)
        )
        self.assertTrue(actual.is_prefill)

    def test_target_verify_does_not_fall_back_to_prefill_attention(self):
        model = SimpleNamespace(
            config=SimpleNamespace(getAttentionConfigs=lambda _: object()),
            parallelism_config=object(),
        )
        inputs = SimpleNamespace(
            attention_inputs=SimpleNamespace(is_target_verify=True)
        )

        with patch(
            "rtp_llm.models_py.modules.factory.attention.cuda_impl."
            "py_flashinfer_mha.PyFlashinferSpecDecodeImpl.support",
            return_value=False,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "requires the FlashInfer speculative decode"
            ):
                _MiniMaxM3ModelMixin.prepare_fmha_impl(model, inputs)


class TargetVerifyBlockTableTest(unittest.TestCase):
    def test_expands_request_rows_to_verify_token_rows(self):
        table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        actual = _repeat_request_block_table_for_verify_tokens(
            table, batch_size=2, total_tokens=6
        )
        expected = torch.tensor(
            [[1, 2], [1, 2], [1, 2], [3, 4], [3, 4], [3, 4]],
            dtype=torch.int32,
        )
        torch.testing.assert_close(actual, expected)

    def test_rejects_non_divisible_token_rows(self):
        with self.assertRaisesRegex(RuntimeError, "batch \* verify_tokens"):
            _repeat_request_block_table_for_verify_tokens(
                torch.zeros((2, 3), dtype=torch.int32), batch_size=2, total_tokens=5
            )

    def test_rejects_wrong_block_rows_for_single_verify_token(self):
        with self.assertRaisesRegex(RuntimeError, "block table batch mismatch"):
            _repeat_request_block_table_for_verify_tokens(
                torch.zeros((1, 3), dtype=torch.int32), batch_size=2, total_tokens=2
            )


if __name__ == "__main__":
    unittest.main()
