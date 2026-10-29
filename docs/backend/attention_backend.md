# Attention Backend

## Supporting matrix for different attention backends

| **Backend**           | **Page Size > 1** | **Spec Decoding** | **MLA** | **Sliding Window** |         **Device Support**         |         **Server Args**         |         **Stage**         |
|-----------------------|-------------------|-------------------|---------|--------------------|------------------------------------|---------------------------------|---------------------------|
| **FLASHINFER_TRTLLM_GEN**        | ✅                | ✅                 | ❌      | ❌                 | NV SM100 ✅<br> AMD ❌ | --enable_flashinfer_trtllm_gen        | PREFILL ✅ <br>  DECODE✅  |
| **FLASHINFER_TRT_FMHA_V2**       | ❌                | ❌                 | ❌      | ❌                 | NV SM90/SM12x ✅<br> AMD ❌ | --enable_flashinfer_trt_fmha_v2       | PREFILL ✅ <br>  DECODE❌  |
| **PAGED_FLASHINFER_TRT_FMHA_V2** | ✅                | ❌                 | ❌      | ❌                 | NV SM90/SM12x ✅<br> AMD ❌ | --enable_paged_flashinfer_trt_fmha_v2 | PREFILL ✅ <br>  DECODE❌  |
| **OPEN_SOURCE**       | ❌                | ❌                 | ❌      | ❌                 | NV ✅<br> AMD ❌        | --enable_open_source_fmha       | PREFILL ✅ <br>  DECODE❌  |
| **PAGED_OPEN_SOURCE** | ✅                | ❌                 | ❌      | ❌                 | NV ✅<br> AMD ❌        | --enable_paged_open_source_fmha | PREFILL ✅ <br>  DECODE❌  |
| **CKFMHA**            | ❌                | ❌                 | ✅      | ✅                 | NV ❌<br> AMD ✅        | None                            | PREFILL ✅ <br>  DECODE❌  |
| **FLASHINFER_NATIVE** | ✅                | ✅                 | ✅      | ✅                 | NV ✅<br> AMD ✅        | --disable_flashinfer_native<br>--disable_flashinfer_hybrid_prefill | PREFILL ✅ <br>  DECODE✅  |
| **XQA**               | ✅                | ❌                 | ❌      | ❌                 | NV Hopper ✅<br> AMD ❌ | --enable_xqa                    | PREFILL ❌ <br>  DECODE✅  |
| **FlashMLA**          | ✅                | ✅                 | ✅      | ❌                 | NV Hopper ✅<br> AMD ❌ | None                            | PREFILL ❌ <br>  DECODE✅  |
| **MMHA**              | ✅                | ❌                 | ❌      | ❌                 | NV ✅<br> AMD ✅        | None                            | PREFILL ❌ <br>  DECODE✅  |
| **AiterPA**           | ✅                | ❌                 | ❌      | ❌                 | NV ❌<br> AMD ✅        | None                            | PREFILL ❌ <br>  DECODE✅  |

## SM100 native XQA and TRT-LLM Gen selection

The bundled native `XQAImpl` dispatches kernels compiled for `sm_90a` and is
eligible only on SM90. A newer compute capability is not compatible with those
architecture-specific kernels. Python selection, native `support()`, and the
native launch path all enforce this constraint. The separate FlashInfer JIT
`XQADecodeImpl` retains its own SM90/SM10x eligibility checks.

TRT-LLM Gen is enabled by default and handles SM100 prefill and decode. For
Qwen2.5-32B with TP4, the per-rank layout is Q10/KV2, head dimension 128, BF16,
and page size 16. When inheriting a configuration that disables TRT-LLM Gen,
set `ENABLE_FLASHINFER_TRTLLM_GEN=True` (or
`--enable_flashinfer_trtllm_gen True`). In Ray deployments, propagate the setting
to both the driver and worker runtime environments. `ENABLE_XQA=False` can also
be used to disable both XQA implementations.

The factory preserves explicit backend disables. If no enabled implementation
supports the SM10x configuration, it reports the shape and TRT-LLM Gen setting
with the configuration remedy. It does not silently re-enable a disabled backend.
The CPU `sm100_attention_selection_test` covers the routing/disable matrix;
`trtllm_gen_test.FlashInferPythonMHATest.test_qwen25_tp4_sm100_page16_bf16`
covers native XQA rejection and numerical TRT-LLM Gen prefill/decode on SM100.

## ROCm KV-cache V layout and PA flag combinations

The matrix uses `[prefill writer:layout] (decode reader:layout)`, where `V` is vectorized and `L` is
linear. It assumes the default `page=16`, BF16/FP16 for BASE cache, and FP8 E4M3; FP8 has
`width=16`, so its linear and vectorized addressing coincide when `page=width`. The rows apply to
full-attention MHA with an interleaved RoPE KV cache, not MLA.

| aiter/asm/triton | 128 BASE | 128 FP8 | 256 BASE | 256 FP8 | fix for ❌ |
|---|---|---|---|---|---|
| `1/1/1` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | — |
| `1/1/0` **(default)** | ✅ `[Asm:V] (Asm:V)` | ✅ `[Asm:V] (Asm:V)` | ❌ `[Asm:V] (NonAsm:L)` | ✅ `[Asm:V] (NonAsm:L)` `page=width` | `--use_triton_pa 1`, or `--use_asm_pa 0` on BASE |
| `1/0/1` | ✅ `[NonAsm:L] (TritonLin:L)` | ✅ `[NonAsm:V] (Triton:V)` | ✅ `[NonAsm:L] (TritonLin:L)` | ✅ `[NonAsm:V] (Triton:V)` | — |
| `1/0/0` | ✅ `[NonAsm:L] (NonAsm:L)` | ✅ `[NonAsm:V] (NonAsm:L)` `page=width` | ✅ `[NonAsm:L] (NonAsm:L)` | ✅ `[NonAsm:V] (NonAsm:L)` `page=width` | `--use_triton_pa 1` for non-default FP8 pages |
| `0/1/1` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | ✅ `[Asm:V] (Triton:V)` | — |
| `0/1/0` | ✅ `[Asm:V] (Asm:V)` | ✅ `[Asm:V] (Asm:V)` | ❌ `[Asm:V] (none)` | ❌ `[Asm:V] (none)` | `--use_triton_pa 1` |
| `0/0/1` | ❌ `[none] (TritonLin:L)` | ❌ `[none] (Triton:V)` | ❌ `[none] (TritonLin:L)` | ❌ `[none] (Triton:V)` | `--use_asm_pa 1`, or `--use_aiter_pa 1` |
| `0/0/0` | ❌ `[none] (none)` | ❌ `[none] (none)` | ❌ `[none] (none)` | ❌ `[none] (none)` | `--use_aiter_pa 1` |

The vector width is `16 // itemsize`; both `head_dim` and `page` must be divisible by it. Non-ASM
decode also requires `page` to divide its selected 512- or 256-token partition. The factory rejects
invalid geometry, missing implementations, and non-equivalent writer/reader layouts; disaggregated
prefill and decode roles must therefore use the same PA flags because no layout negotiation exists.
