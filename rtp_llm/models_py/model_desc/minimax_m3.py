from typing import Any, Dict, Optional

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_desc.generic_moe import (
    GenericMoeDecoderLayer,
    GenericMoeModel,
)
from rtp_llm.models_py.model_desc.multimodal_generic import MultimodalGenericModel
from rtp_llm.models_py.modules import CausalAttention
from rtp_llm.models_py.modules.factory.attention.fmha_impl_base import FMHAImplBase
from rtp_llm.models_py.modules.hybrid.msa_attention import MSAAttention
from rtp_llm.ops import HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache, PyAttentionInputs, PyModelInputs
from rtp_llm.utils.model_weight import W


class MiniMaxM3DecoderLayer(GenericMoeDecoderLayer):
    def _create_attention(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        quant_config: Any,
        hw_kernel_config: Optional[HWKernelConfig],
    ) -> nn.Module:
        if config.attn_config.use_mla:
            return super()._create_attention(
                config,
                parallelism_config,
                weights,
                global_weights,
                layer_idx,
                quant_config,
                hw_kernel_config,
            )

        # MiniMax-M3 attention weights are not tensor-parallel sharded.
        attn_configs = config.getAttentionConfigs(1)
        msa_config = config.msa_sparse_config
        is_sparse_layer = (
            msa_config is not None
            and layer_idx in set(msa_config.get("sparse_layer_ids", []))
            and W.msa_idx_q_w in weights
        )
        if is_sparse_layer:
            return MSAAttention(
                attn_configs,
                parallelism_config,
                weights,
                config.layernorm_eps,
                msa_config,
                layer_idx,
                quant_config,
                hw_kernel_config,
            )
        return CausalAttention(
            attn_configs,
            parallelism_config,
            weights,
            config.layernorm_eps,
            quant_config,
            hw_kernel_config,
            layer_idx,
        )

    def _input_quant_projection(self) -> Optional[nn.Module]:
        if isinstance(self.self_attn, MSAAttention):
            return getattr(self.self_attn, "qkv_proj", None)
        return super()._input_quant_projection()

    def _forward_attention(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        prev_topk_indices: Optional[torch.Tensor],
        force_reuse_topk_indices: bool,
        attn_inputs: Optional[Any],
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if not isinstance(self.self_attn, MSAAttention):
            return super()._forward_attention(
                hidden_states,
                fmha_impl,
                kv_cache,
                prev_topk_indices,
                force_reuse_topk_indices,
                attn_inputs,
                x_fp8,
                x_scale,
            )

        quantized_inputs = {}
        if x_fp8 is not None:
            quantized_inputs = {"x_fp8": x_fp8, "x_scale": x_scale}
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attn_inputs=attn_inputs,
            kv_cache=kv_cache,
            **quantized_inputs,
        )
        return hidden_states, None


class _MiniMaxM3ModelMixin:
    decoder_layer_cls = MiniMaxM3DecoderLayer
    requires_grouped_physical_kv_tables = True

    def prepare_target_verify_attention_inputs(
        self, attn_inputs: PyAttentionInputs
    ) -> PyAttentionInputs:
        if not bool(getattr(attn_inputs, "is_target_verify", False)):
            return attn_inputs

        batch_size = int(attn_inputs.prefix_lengths.numel())
        total_tokens = int(attn_inputs.total_tokens)
        if batch_size <= 0 or total_tokens % batch_size != 0:
            raise RuntimeError(
                "MiniMax-M3 target verify expects a positive batch size and "
                f"a flat token window divisible by the batch, got batch={batch_size}, "
                f"tokens={total_tokens}"
            )

        verify_tokens = total_tokens // batch_size
        device = attn_inputs.prefix_lengths.device
        relative_positions = torch.arange(
            verify_tokens, dtype=torch.int32, device=device
        )
        attn_inputs.sequence_lengths_plus_1_d = (
            attn_inputs.prefix_lengths.to(dtype=torch.int32).unsqueeze(1)
            + relative_positions.unsqueeze(0)
            + 1
        ).reshape(-1)
        attn_inputs.decode_cu_seqlens_d = torch.arange(
            0,
            total_tokens + 1,
            verify_tokens,
            dtype=torch.int32,
            device=device,
        )
        attn_inputs.cu_seqlens = attn_inputs.decode_cu_seqlens_d
        return attn_inputs

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        attn_inputs = inputs.attention_inputs
        if attn_inputs is not None and bool(
            getattr(attn_inputs, "is_target_verify", False)
        ):
            from rtp_llm.models_py.modules.factory.attention.cuda_impl.py_flashinfer_mha import (
                PyFlashinferSpecDecodeImpl,
            )

            attn_inputs.is_cuda_graph = is_cuda_graph
            attn_configs = self.config.getAttentionConfigs(1)
            if not PyFlashinferSpecDecodeImpl.support(attn_configs, attn_inputs):
                raise RuntimeError(
                    "MiniMax-M3 target verify requires the FlashInfer "
                    "speculative decode attention implementation"
                )
            return PyFlashinferSpecDecodeImpl(
                attn_configs, attn_inputs, self.parallelism_config
            )
        return super().prepare_fmha_impl(inputs, is_cuda_graph)


class MiniMaxM3Model(_MiniMaxM3ModelMixin, GenericMoeModel):
    pass


class MiniMaxM3MultimodalModel(_MiniMaxM3ModelMixin, MultimodalGenericModel):
    pass
