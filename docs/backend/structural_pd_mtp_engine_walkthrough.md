# Structural + Think + PD 分离 + MTP 约束采样：Engine 全链路导读

本文用两条本地分支互相补全：

- 第 1～16、19～24 节沿当前 `feature/grammar_logits_processor`，解释 structural + PD + MTP 的主语义链；
- 第 17～18 节以 `feat/dsv4_on_dev@dd20403c3466bb510e7ae60b9d6935dc527c0247` 的真实代码为准，逐行解释已经存在的 MTP/XGrammar 异步链，而不是假想设计。

跟踪的请求同时满足以下条件：

- 使用 `structural_tag` 约束最终输出；
- 开启 think/reasoning，并限制 thinking token 数；
- 走 Prefill/Decode（PD）分离；
- Engine 开启 MTP speculative decoding；
- 最终在 target logits 上执行 XGrammar 硬约束采样。

重点是 C++ engine。Python 只解释一个必要前提：think 与最终结构约束如何进入 engine。文中的代码均为精简摘录，省略了日志、metrics 和与主线无关的 tensor 字段。

两条分支在 think + grammar 的封装上有一个必须先知道的差异：

```text
feature/grammar_logits_processor             feat/dsv4_on_dev
────────────────────────────────             ────────────────────────────────
Python 合成 structural sequence              C++ 看到 grammar_key + in_think_mode
                 │                                            │
                 ▼                                            ▼
GrammarLogitsProcessor                        ReasoningGrammarLogitsProcessor
一个 matcher 表达 think+final                 Think 状态机 + final grammar matcher
```

二者在 MTP correctness 上遵守同一个核心协议：

```text
对 draft 做临时预演并回滚
          │
          ▼
生成 P+1 行约束 mask + grammar cap
          │
          ▼
target 采样 + probability rejection
          │
          ▼
只把实际 accepted tokens 永久 commit
```

## 1. 先给结论

这条链路不是“ThinkModeLogitsProcessor 再叠加 GrammarLogitsProcessor”，而是：

```text
think 参数 + 最终 structural/json 约束
                  │
                  ▼
      Python 归一化为一个 structural_tag sequence
                  │
                  ▼
       C++ 只安装 GrammarLogitsProcessor
                  │
          ┌───────┴────────┐
          ▼                ▼
   normal/prefill      MTP decode
  process(logits)   prepareSpeculative(drafts)
          │                │
          └───────┬────────┘
                  ▼
         updateStatus(最终接受 tokens)
```

PD 两端也不会共享同一个 matcher：

```text
Prefill process                          Decode process
┌─────────────────────┐                 ┌─────────────────────┐
│ GrammarMatcher(P)   │                 │ GrammarMatcher(D)   │
│ committed = [t0]    │  -- 发送 t0 --> │ 初始 matcher        │
└─────────────────────┘                 │ updateStatus([t0])  │
                                        │ committed = [t0]    │
                                        └─────────────────────┘
```

这里 `t0` 是 Prefill 端 target model 采出的第一个输出 token。Decode 端用它重放自己新建的 matcher 状态。

## 2. 示例请求在 engine 里长什么样

假设用户需要：

1. 先自由输出 reasoning；
2. 最多思考 64 token；
3. 遇到 `</think>\n\n` 后进入 JSON schema；
4. JSON 完成后结束 grammar。

进入 C++ 前的核心配置大致会被归一化为：

```json
{
  "in_think_mode": true,
  "max_thinking_tokens": 64,
  "structural_tag": {
    "type": "structural_tag",
    "format": {
      "type": "sequence",
      "elements": [
        {
          "type": "tag",
          "begin": "",
          "content": {"type": "any_text", "max_tokens": 64},
          "end": "</think>\n\n"
        },
        {
          "type": "json_schema",
          "json_schema": {"type": "object"},
          "style": "json"
        }
      ]
    }
  },
  "grammar_terminate_without_stop_token": true
}
```

对应的 grammar 状态机可以先粗略看成：

```text
┌──────────────────────────┐
│ THINK_ANY_TEXT           │
│ 已生成 reasoning < 64    │
└─────────────┬────────────┘
              │ 生成 </think> 或达到预算
              ▼
┌──────────────────────────┐
│ THINK_END                │
│ 完成 end tag/token       │
└─────────────┬────────────┘
              │
              ▼
┌──────────────────────────┐
│ FINAL_JSON_SCHEMA        │
│ 只允许合法 JSON token    │
└─────────────┬────────────┘
              │ schema 完整
              ▼
┌──────────────────────────┐
│ TERMINATED               │
│ 下一 token 只允许 EOS    │
└──────────────────────────┘
```

### 2.1 Python 的必要前置步骤

源码：[response_format_builder.py](../../rtp_llm/config/response_format_builder.py)

```python
def apply(self) -> None:
    constraint = self._resolve_grammar_constraint()
    if not self.config.in_think_mode:
        return
    if constraint is not None:
        self._wrap_grammar_with_reasoning_envelope(constraint)

def _wrap_final_format_with_reasoning_envelope(self, final_format):
    reasoning_prefix = self.reasoning_format.prefix_format(
        self.config.max_thinking_tokens
    )
    envelope = {
        "type": "structural_tag",
        "format": {
            "type": "sequence",
            "elements": [reasoning_prefix, final_format],
        },
    }
    self.config.structural_tag = dump_compact_json(envelope)
    self.config.json_schema = None
```

讲解：

- think 不再是 C++ 中另一个独立 processor；
- thinking budget、think end tag/token、最终 JSON/regex/EBNF 都被编译进一个 grammar；
- `GenerateConfig::in_think_mode` 仍会跨 RPC 传输，但在这条 engine 路径里主要用于配置与日志；真正限制 token 的是 `structural_tag`；
- 当前 `LogitsProcessorFactory` 没有创建 `ThinkModeLogitsProcessor`。看到该类仍在仓库里，不代表这个请求会经过它。

## 3. 全链路总图

```text
Client / Frontend
      │ GenerateInputPB
      ▼
┌───────────────────────────────────────────────────────────────┐
│ PrefillRpcServer::GenerateStreamCall                          │
│  判断可走 PD，设置 pd_separation=true                         │
└──────────────┬────────────────────────────────────────────────┘
               │
       ┌───────┴──────────────────────────┐
       │                                  │
       ▼                                  ▼
DecodeRpcServer::RemoteGenerate       Prefill engine enqueue
  allocateResource                      │
  makeStream(D)                         ▼
  compile grammar(D)                MtpExecutor::prefillStep
  load KV cache                         │
       │                                ├─ target prefill
       │                                ├─ grammar mask first logits
       │                                ├─ sample first target token t0
       │                                ├─ draft model proposes d1
       │                                └─ specUpdate commits t0
       │                                  │
       └────────────── wait KV ───────────┤
                                          ▼
                            PrefillRpcServer::remoteGenerate
                              send {t0, d1, probs, hidden}
                                          │
                                          ▼
                              DecodeRpcServer::localGenerate
                              update([t0]) 重放 matcher(D)
                              restore SP buffer [t0,d1]
                              enqueue decode stream
                                          │
                                          ▼
                              MtpExecutor::decodeStep
                                          │
             ┌────────────────────────────┼────────────────────┐
             ▼                            ▼                    ▼
       draftModelDecode             target verify       XGrammar verify
       产生 d1..dP                  并行算 P+1 行 logits   产生 P+1 行 mask
             │                            │                    │
             └────────────────────────────┴──────────┬─────────┘
                                                     ▼
                                      mask target logits + sample
                                                     │
                                                     ▼
                                      speculative rejection sampling
                                                     │
                                                     ▼
                                      grammar cap 修正 accept_len
                                                     │
                                                     ▼
                                      specUpdate(accepted tokens)
                                                     │
                                  ┌──────────────────┴──────────────┐
                                  │                                 │
                              未结束                           grammar/请求结束
                                  │                                 │
                                  └── next MTP round                ▼
                                                            output / EOS
```

## 4. Engine 初始化：为什么会进入 MTP executor

### 4.1 `NormalEngine::initExecutor`

源码：[NormalEngine.cc](../../rtp_llm/cpp/normal_engine/NormalEngine.cc)

```cpp
void NormalEngine::initExecutor(...,
                                std::unique_ptr<ProposeModelEngineInitParams>& propose_params) {
    if (propose_params_) {
        executor_.reset(new MtpExecutor(params,
                                        propose_params,
                                        resource_context_.cache_manager,
                                        ...));
    } else {
        executor_.reset(new NormalExecutor(...));
    }
}
```

讲解：

- 服务启动时只要带了 propose model 参数，`NormalEngine` 就选择 `MtpExecutor`；
- 不是请求逐条选择 executor；请求只能通过 `force_disable_sp_run` 等字段影响是否走 speculative 细节；
- PD 的 Prefill、Decode 进程各有一个自己的 engine/executor。

```text
Prefill node                              Decode node
┌──────────────────────┐                 ┌──────────────────────┐
│ NormalEngine(P)      │                 │ NormalEngine(D)      │
│   MtpExecutor(P)     │                 │   MtpExecutor(D)     │
│   XGrammarBackend(P) │                 │   XGrammarBackend(D) │
└──────────────────────┘                 └──────────────────────┘
```

### 4.2 `MtpExecutor` 构造：初始化 grammar backend

源码：[MtpExecutor.cc](../../rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc)

```cpp
MtpExecutor::MtpExecutor(...) {
    propose_step_ = propose_params->gen_num_per_circle;

    batch_stream_processor_.reset(new MtpBatchStreamProcessor(...));

    LogitsProcessorFactory::init(params.model_config_,
                                 params.grammar_config,
                                 params.sp_config.tree_decode_config);

    model_.reset(new PyWrappedModel(...));       // target model
    draft_model_.reset(new PyWrappedModel(...)); // MTP draft model
}
```

讲解：

- `propose_step_ = P`：每轮最多验证 P 个 draft token，并额外保留一个 target bonus/fallback 位置；
- target model 和 draft model 各有自己的模型实例和 KV cache layout；
- `LogitsProcessorFactory::init()` 是进程级初始化，不是每请求初始化。

### 4.3 `LogitsProcessorFactory::init`

源码：[LogitsProcessorFactory.cc](../../rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.cc)

```cpp
void LogitsProcessorFactory::init(const ModelConfig& model_config,
                                  const GrammarConfig& grammar_config,
                                  const std::string& tree_decode_config) {
    grammarBackend() =
        XGrammarBackend::create(model_config.tokenizer_info_json, grammar_config);
    PrefixToCandidateTokens::instance()->reloadPrefixDictWithPrefix(
        model_config.ckpt_path, tree_decode_config);
}
```

讲解：

- backend 保存 tokenizer 信息和 `xgrammar::GrammarCompiler`；
- 编译结果由 XGrammar compiler cache 复用；
- 每个 stream 仍有独立 matcher，不能在请求之间共享 matcher 状态。

## 5. PD 请求进入 engine

### 5.1 `PrefillRpcServer::GenerateStreamCall`

源码：[PrefillRpcServer.cc](../../rtp_llm/cpp/model_rpc/PrefillRpcServer.cc)

```cpp
auto pd_separation =
    request->generate_config().max_new_tokens() > 1
    && request->generate_config().num_beams() <= 1
    && request->generate_config().variable_num_beams().size() == 0
    && request->generate_config().num_return_sequences() <= 1
    && request->generate_config().can_use_pd_separation();

if (!pd_separation) {
    return LocalRpcServer::GenerateStreamCall(...);
}

prepareAllocateResource(prefill_context);
enqueueRequest(prefill_context);
remoteLoadCacheStart(prefill_context);
pollLocalOutput(prefill_context);
remoteLoadCacheEnd(prefill_context);
remoteGenerate(prefill_context);
pollRemoteOutput(prefill_context);
```

讲解：

- structural grammar 本身也拒绝 beam/multi-return；PD gate 与它方向一致；
- 先让 Decode 端建 stream/分配资源，再让 Prefill 端实际跑首 token；
- Prefill 首 token 和 KV transfer 完成后，Decode 才开始后续生成。

PD 阶段图：

```text
Prefill RPC thread               Prefill engine            Decode RPC thread
       │                               │                           │
       │ ALLOCATE ------------------------------------------------>│
       │                               │                    makeStream(D)
       │                               │                    allocate KV
       │<--------------------------------------------------- ACK    │
       │ enqueue(P)                    │                           │
       │------------------------------>│                           │
       │ LOAD request -------------------------------------------->│
       │                               │ prefill + first token     │ load KV
       │<------------------------------│                           │
       │<------------------------------------------------ load ACK │
       │ GENERATE {t0,d1,...} ------------------------------------>│
       │                               │                    enqueue(D)
       │<================================================ outputs ==│
```

### 5.2 `PrefillRpcServer::getRpcConnection`

```cpp
auto input = QueryConverter::transQuery(request);
input->generate_config->pd_separation = true;
if (engine_->isMTPEagle()) {
    input->generate_config->force_disable_sp_run = false;
} else {
    input->generate_config->force_disable_sp_run = true;
}
prefill_context.generate_input = input;
```

讲解：

- `pd_separation` 是 Prefill 端本地运行时字段；
- MTP/Eagle 时允许 Prefill 端同时产出首个 draft token；
- Decode 收到的是原始 PB，重新构造自己的 `GenerateInput`，不会共享 C++ 对象。

### 5.3 `QueryConverter::transGenerateConfig`

源码：[QueryConverter.cc](../../rtp_llm/cpp/model_rpc/QueryConverter.cc)

```cpp
TRANS_OPTIONAL(json_schema);
TRANS_OPTIONAL(regex);
TRANS_OPTIONAL(ebnf);
TRANS_OPTIONAL(structural_tag);

generate_config->grammar_terminate_without_stop_token =
    config_proto->grammar_terminate_without_stop_token();
generate_config->in_think_mode       = config_proto->in_think_mode();
generate_config->max_thinking_tokens = config_proto->max_thinking_tokens();
for (auto token_id : config_proto->end_think_token_ids()) {
    generate_config->end_think_token_ids.push_back(token_id);
}
```

讲解：

- typed grammar 字段通过 proto 原样跨 PD；
- `structural_tag` 已经包含实际 think grammar；
- `in_think_mode/max_thinking_tokens/end_think_token_ids` 仍被保留，但这条路径不会据此创建旧 think processor。

## 6. 每个 PD 节点如何创建自己的 Grammar processor

### 6.1 `NormalEngine::makeStream/enqueue`

```cpp
std::shared_ptr<GenerateStream>
NormalEngine::makeStream(const std::shared_ptr<GenerateInput>& input) {
    return std::make_shared<NormalGenerateStream>(
        input, model_config_, runtime_config, resource_context_, metrics_reporter_);
}
```

`NormalGenerateStream` 的基类构造最终进入 `GenerateStream::GenerateStream`。

### 6.2 `GenerateStream` 构造

源码：[GenerateStream.cc](../../rtp_llm/cpp/engine_base/stream/GenerateStream.cc)

```cpp
auto processors_result = LogitsProcessorFactory::createLogitsProcessors(
    generate_input_, init_batch_size, maxBatchSize(), special_tokens_.eos_token_id);
if (processors_result.ok()) {
    logits_processors_ = std::move(processors_result.value());
} else {
    reportEventWithoutLock(StreamEvents::Error,
                           err.code(), err.ToString());
}
```

讲解：

- grammar 编译/processor 创建失败不会留下一个“无约束继续跑”的 stream；
- stream 会带 Error event，后续 admission/schedule/output 路径会报告失败；
- Prefill 和 Decode 各执行一次，因此有两个独立 processor/matcher。

### 6.3 `LogitsProcessorFactory::createLogitsProcessors`

```cpp
auto grammar_key_result = keyFromGenerateConfig(config);
GrammarKeyCpp grammar_key = grammar_key_result.value();

if (!grammar_key.empty()) {
    if (config.hasNumBeams() || config.num_return_sequences > 1) {
        return ErrorInfo(INVALID_PARAMS,
                         "grammar-constrained decoding does not support ...");
    }

    auto matcher_or = backend->createMatcherFromKey(
        grammar_key, terminate_without_stop_token);

    result.push_back(std::make_shared<GrammarLogitsProcessor>(
        std::move(matcher_or.value()), eos_token_id));
}
```

`keyFromGenerateConfig()` 对当前请求返回：

```cpp
return GrammarKeyCpp{"structural_tag", config.structural_tag.value()};
```

讲解：

- 四种 grammar 字段严格 one-of；
- 当前请求最终只得到一个 `GrammarLogitsProcessor`；
- grammar processor 声明 `MtpProcessorMode::SPEC_VERIFY`，所以能进入 MTP decode。

### 6.4 `XGrammarBackend::compile/createMatcherFromKey`

源码：[XGrammarBackend.cc](../../rtp_llm/cpp/engine_base/grammar/XGrammarBackend.cc)

```cpp
if (key.key_type == "structural_tag") {
    result = compileWithErrorClassification(
        [&] { return compiler_.CompileStructuralTag(key.key_string); });
}

auto compiled_or = compile(key);
return createMatcher(std::move(compiled_or.value()),
                     terminate_without_stop_token);
```

```cpp
return std::make_shared<RtpGrammarMatcher>(
    std::move(compiled),
    options_.override_stop_tokens,
    terminate_without_stop_token);
```

讲解：

- compiled grammar 可由 compiler cache 复用；
- matcher 是 per-stream 状态机；
- `terminate_without_stop_token` 决定 grammar 完成是否可直接进入 terminated，而不要求结构本身带 stop token。

对象关系：

```text
Process
└── XGrammarBackend
    ├── TokenizerInfo
    ├── GrammarCompiler
    └── compiled grammar cache
          │ shared compiled grammar
          ├───────────────────────────────┐
          ▼                               ▼
Stream A                              Stream B
GrammarLogitsProcessor                GrammarLogitsProcessor
└── RtpGrammarMatcher A               └── RtpGrammarMatcher B
    state = prefix A                       state = prefix B
```

## 7. Prefill 端：MTP 首轮如何做约束采样

### 7.1 `NormalEngine::step`

```cpp
streams = scheduler_->schedule();
status = executor_->process(streams);
```

这里的 `executor_` 是 `MtpExecutor`。

### 7.2 `MtpExecutor::process`

```cpp
prepareStreams(streams, prefill_streams, decode_streams);

if (role_type_ == RoleType::PREFILL || role_type_ == RoleType::PDFUSION) {
    prefillStep(prefill_streams, metrics_collector);
}
if (role_type_ == RoleType::DECODE || role_type_ == RoleType::PDFUSION) {
    decodeStep(decode_streams, metrics_collector);
}
```

讲解：

- Prefill 节点只执行 `prefillStep`；
- Decode 节点只执行 `decodeStep`；
- PDFUSION 才可能在同一进程两边都执行。

### 7.3 `MtpExecutor::prepareStreams`

```cpp
for (auto& stream : streams) {
    if (auto error = validateMtpCompatibility(
            stream->getAllLogitsProcessorPtr()); error.has_value()) {
        stream->reportError(error->code(), error->ToString());
        continue;
    }

    if (stream->isContextStream()) {
        prefill_streams.push_back(stream);
    } else {
        stream->setScoreLen(propose_step_ + 1);
        decode_streams.push_back(stream);
    }

    stream->setReturnAllProbs(ReturnAllProbsMode::DEFAULT);
    stream->getSPOutputBuffer()->propose_step = propose_step_;
}
```

讲解：

- processor 必须显式声明支持 MTP；默认 capability 是 `UNSUPPORTED`；
- Grammar processor 返回 `SPEC_VERIFY`；
- MTP rejection sampling 需要 draft/target 概率，所以强制保留 all probs。

### 7.4 `MtpExecutor::prefillStep`

核心代码：

```cpp
model_input  = batch_stream_processor_->gatherModelInput(stream_groups);
model_output = model_->forward(model_input); // target prefill

auto sampler_input = batch_stream_processor_->gatherSamplerInput(
    stream_groups, model_input, model_output);
sampler_output = sampler_->forward(sampler_input); // target 首 token

batch_stream_processor_->updatePrefillPostDraftModelInput(
    stream_groups, model_input, model_output, sampler_output);

draft_model_output = draft_model_->forward(model_input);
draft_sampler_output = fast_topk_sampler_->forward(draft_model_output.logits);

batch_stream_processor_->dispatchPrefill(
    stream_groups,
    {model_output, sampler_output},
    {draft_model_output, draft_sampler_output});
```

Prefill 数据流：

```text
prompt tokens
    │
    ▼
target model prefill
    │ logits(t0)
    ▼
GrammarLogitsProcessor::process
    │ masked logits(t0)
    ▼
Sampler::forward
    │ target first token = t0
    ▼
draft model prefill(t0 + hidden)
    │
    ▼
FastTopKSampler
    │ next draft = d1
    ▼
SP buffer = [t0, d1, draft_probs, draft_hidden]
```

注意：Prefill 的第一个 target token 使用的是 normal sampler processor 链；从 Decode MTP round 开始，才使用专门的 speculative grammar verify 链。

## 8. 把整条链穿起来：一次 normal constrained sampling 如何得到 `t0`

前面各节已经分别介绍过函数。这一节换一个读法：只跟踪 **Prefill 首 token `t0` 的一次真实调用**，像单步调试一样看上一个函数留下什么、下一个函数拿走什么。

先固定现场。设请求已经完成 prompt prefill：

```text
GenerateStream
├── 已提交输出长度：N = 0
├── GrammarLogitsProcessor
│   ├── committed_output_len_ = 0
│   └── matcher_ = S0             # grammar 位于输出开头
└── generate config               # top-k/top-p/temperature 等

target model
└── model_output.logits = L0      # shape=[batch, vocab]，尚未约束
```

本节跟踪的三个东西分别是：

```text
控制流：谁调用谁
    MtpExecutor → gather → Sampler → processor → matcher → CUDA → Sampler

数据流：logits 怎样变化
    raw L0 → sampler_inputs.logits → 原地 hard mask → sampling → t0

状态流：grammar 状态怎样变化
    S0 ──只读，生成 mask──> S0 ──第 9 节 commit(t0)──> S1
```

完整调用栈先展开一次，后面逐层进入再逐层返回：

```text
MtpExecutor::prefillStep
│
├─ target model::forward
│    └─ 产出 raw logits L0
│
├─ MtpBatchStreamProcessor::gatherSamplerInput
│    └─ NormalSamplerInputGatherer::gather
│         ├─ 把 L0 放入 SamplerInputs::logits
│         └─ setLogitsProcessorInputs
│              └─ 建立 processor → logits row 的调用计划
│
└─ Sampler::forward
     ├─ preprocessLogits
     │    └─ LogitsProcessorStates::batchProcess
     │         └─ GrammarLogitsProcessor::process(row 0)
     │              └─ DecodeMaskBuilder::apply
     │                   ├─ buildState
     │                   │    └─ RtpGrammarMatcher::fillBitmask
     │                   │         └─ XGrammar::FillNextTokenBitmask
     │                   └─ runtimeApplyPackedMaskLogits
     │                        └─ L0 中非法 token 原地写成 -inf
     │
     └─ execSampleGreedy(L0 已被 mask)
          └─ 产出 t0
```

注意这不是两条独立的 sampler 链。`preprocessLogits()` 和 `execSampleGreedy()` 在同一次 `Sampler::forward()` 中串行执行，并且后者读取的就是前者已经改过的那块 logits。

### 8.1 起点：`MtpExecutor::prefillStep` 拿到 raw logits

源码：[MtpExecutor.cc](../../rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc)

```cpp
// target model prefill
model_output = std::move(model_->forward(model_input));

// target model sample
if (isTpRank0()) {
    CHECK_AND_RETURN_REF(
        sampler_input,
        batch_stream_processor_->gatherSamplerInput(
            stream_groups, model_input, model_output));

    sampler_output = std::move(sampler_->forward(sampler_input));

    batch_stream_processor_->updatePrefillPostDraftModelInput(
        stream_groups, model_input, model_output, sampler_output);
}
```

进入这一段时，target model 已经根据 prompt 算出“输出第一个 token”的分数 `L0`。这些分数只表达模型偏好，还不知道 JSON Schema 允许什么。

```text
模型视角                          grammar 视角

"a"  logit=12.1  ← 模型最想选      当前必须先输出 "{"
"{"  logit= 8.3
"["  logit= 7.9
EOS  logit= 1.2
```

因此 `prefillStep()` 不直接采样，而是把 `model_output` 和对应的 `stream_groups` 一起交给 `gatherSamplerInput()`。两类信息会在 gather 中合并：

```text
model_output                         GenerateStream
└── raw logits L0                    ├── sampling config
                                     └── GrammarLogitsProcessor(S0)
            │                                      │
            └──────────────┬───────────────────────┘
                           ▼
                     SamplerInputs
```

**本步输出给下一步：** `model_output.logits=L0` 与持有 grammar processor 的 streams。

### 8.2 中转：`gatherSamplerInput` 把模型结果和请求状态装进同一个输入

`MtpBatchStreamProcessor` 没有为 Prefill 首 token 另造 sampler，它沿用 normal sampler gather：

源码：[NormalBatchStreamProcessor.cc](../../rtp_llm/cpp/normal_engine/NormalBatchStreamProcessor.cc)

```cpp
absl::StatusOr<SamplerInputs>
NormalBatchStreamProcessor::gatherSamplerInput(
    const StreamGroups& stream_groups,
    const GptModelInputs& model_inputs,
    const GptModelOutputs& model_output) const {
    return sampler_input_gatherer_->gather(
        stream_groups, model_inputs, model_output);
}
```

随后 `NormalSamplerInputGatherer::gather()` 做两件会在后面汇合的事：

源码：[NormalSamplerInputGatherer.cc](../../rtp_llm/cpp/normal_engine/NormalSamplerInputGatherer.cc)

```cpp
SamplerInputs sampler_inputs = allocateSamplerInputs(...);
fillSamplerCommonInputs(sampler_inputs, all_streams);

setLogitsProcessorInputs(sampler_inputs, all_streams);

...
sampler_inputs.logits = logits_tensor;
return std::move(sampler_inputs);
```

```text
支路 A：模型数据                         支路 B：请求控制数据

model_output.logits                     stream->generateConfig()
       │                                stream->getAllLogitsProcessorPtr()
       ▼                                         │
sampler_inputs.logits                            ▼
                                      top-k/top-p/... + 调用计划
       │                                         │
       └───────────────────┬─────────────────────┘
                           ▼
                  sampler_->forward(inputs)
```

`logits_tensor` 是否 clone 取决于 tiling、是否返回 logits 等条件；但进入 `Sampler::forward()` 后，grammar processor 和 sampler 看到的始终是同一个 `sampler_inputs.logits` tensor。后面说“原地修改”都是指它。

**本步输出给下一步：** 一个同时包含 `L0`、sampling 参数和 processor 调用计划的 `SamplerInputs`。

### 8.3 `setLogitsProcessorInputs` 建的不是 mask，而是一张“稍后怎么调用”的表

```cpp
void NormalSamplerInputGatherer::setLogitsProcessorInputs(
    SamplerInputs& sampler_inputs,
    std::list<GenerateStreamPtr>& all_streams) const {
    auto state_ptr = std::make_shared<LogitsProcessorStates>();
    size_t idx = 0;
    for (auto& stream : all_streams) {
        const size_t batch_size = stream->currentBatchSize();
        for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
            if (processor) {
                state_ptr->insert(processor, idx, idx + batch_size);
            }
        }
        idx += batch_size;
    }
    sampler_inputs.logits_processor_states_ptr = state_ptr;
}
```

这个函数当前完全不调用 XGrammar，也完全不碰 logits。它只解决一个 batch bookkeeping 问题：某个 stream 的 processor 应该修改 batched logits 的哪几行。

```text
all_streams                     SamplerInputs::logits

stream A, batch=1  ───────────> row 0
  processor A                    interval [0, 1)

stream B, batch=1  ───────────> row 1
  processor B                    interval [1, 2)

最终 invocations_：
┌─────────────────┬───────────────┐
│ processor A     │ [0, 1)        │
│ processor B     │ [1, 2)        │
└─────────────────┴───────────────┘
```

对本节这个 structural 请求而言，grammar processor 自己只允许单序列，所以跟踪项可以简化成：

```text
invocation = { GrammarLogitsProcessor(S0), interval=[0,1) }
```

这里容易产生一个误解：`logits_processor_states_ptr` 的名字像“processor 状态”，其实它在这条路径中更像一个 **延迟执行计划**。真正的 parser 状态仍在 `GrammarLogitsProcessor::matcher_` 里。

**本步输出给下一步：** `SamplerInputs.logits_processor_states_ptr`，里面记录“让 grammar processor 处理 row 0”。

### 8.4 `Sampler::forward` 是两半：先约束，再采样

源码：[Sampler.cc](../../rtp_llm/cpp/models/Sampler.cc)

```cpp
SamplerOutput Sampler::forward(const SamplerInputs& inputs) {
    ...
    auto processor_errors = preprocessLogits(inputs);

    ...
    auto logits = inputs.logits.narrow(
        0, from_batch_idx_in, batch_size_in);

    auto greedy_output = execSampleGreedy({
        logits,
        ...
        top_k,
        top_p,
        temperature,
        repetition_penalty,
        ...
    });

    return SamplerOutput({
        ...,
        std::move(processor_errors)});
}

std::vector<std::optional<ErrorInfo>>
Sampler::preprocessLogits(const SamplerInputs& inputs) {
    if (inputs.logits_processor_states_ptr != nullptr) {
        return inputs.logits_processor_states_ptr->batchProcess(inputs);
    }
    return {};
}
```

调用顺序不能颠倒：

```text
时间 ───────────────────────────────────────────────────────────>

inputs.logits=L0
      │
      ├─ preprocessLogits(inputs)
      │      └─ grammar 把非法项原地改成 -inf
      │
      └─ inputs.logits.narrow(...)      # 只是同一 tensor 的 view
             └─ execSampleGreedy(masked L0)
                    └─ repetition/presence/frequency penalty
                    └─ temperature
                    └─ top-k / top-p
                    └─ sample
```

所以 grammar 是 sampler 前面的硬门禁。后续 temperature、top-k、top-p 都只能重新排序或筛选合法 token，无法把 `-inf` 的非法 token“救回来”。

此时程序先进入 `preprocessLogits()`，还没有产生 `t0`。

**本步向下调用：** 把整个 `SamplerInputs` 交给 `LogitsProcessorStates::batchProcess()`。

### 8.5 `batchProcess` 按调用计划把 row 0 派给 grammar processor

源码：[LogitsProcessorStates.cc](../../rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.cc)

```cpp
std::vector<std::optional<ErrorInfo>>
LogitsProcessorStates::batchProcess(const SamplerInputs& inputs) {
    std::vector<std::optional<ErrorInfo>> processor_errors(
        inputs.batch_size);

    for (const auto& invocation : invocations_) {
        const auto& interval = invocation.interval;
        auto error = invocation.processor->process(
            inputs, interval.first, interval.second);
        if (error.has_value()) {
            setIntervalError(processor_errors, interval, error.value());
        }
    }
    return processor_errors;
}
```

上一函数放进 `invocations_` 的 `{processor, [0,1)}`，现在第一次真正被消费：

```text
SamplerInputs
├── logits: [ row0=L0, row1=..., ... ]
└── invocation: GrammarProcessor → [0,1)
                               │
                               ▼
             processor->process(inputs, 0, 1)
```

`batchProcess()` 自己仍不懂 grammar。它只做动态分发，并把 processor 错误写回对应 row 的 `processor_errors[row]`。这保证某个请求的 grammar 失败不会被误记到同 batch 的另一个请求。

**本步向下调用：** `GrammarLogitsProcessor::process(inputs, 0, 1)`。

### 8.6 `GrammarLogitsProcessor::process` 在这里把“row 0”和“matcher S0”接上

源码：[GrammarLogitsProcessor.cc](../../rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.cc)

```cpp
std::optional<ErrorInfo>
GrammarLogitsProcessor::process(
    const SamplerInputs& inputs,
    size_t start_idx,
    size_t finish_idx) {
    if (!matcher_) {
        return std::nullopt;
    }
    const size_t batch_size = finish_idx - start_idx;
    if (batch_size != 1) {
        return ErrorInfo(
            ErrorCode::INVALID_PARAMS,
            "grammar logits processor only supports single sequence decoding");
    }
    if (inputs.finished_mask.defined()
        && inputs.finished_mask.data_ptr<bool>()[start_idx]) {
        return std::nullopt;
    }

    std::lock_guard<std::mutex> lock(state_mutex_);
    auto error = decode_mask_builder_->apply(
        inputs.logits[start_idx],
        *matcher_,
        committed_output_len_,
        eos_token_id_);
    if (error.hasError()) {
        return error;
    }
    return std::nullopt;
}
```

在这一行，前面一直平行传播的两条数据第一次真正汇合：

```text
batched GPU logits                         CPU grammar state

inputs.logits[start_idx=0]                 matcher_ = S0
          │                                committed_output_len_ = 0
          │                                         │
          └─────────────────┬───────────────────────┘
                            ▼
             DecodeMaskBuilder::apply(L0, S0, 0, EOS)
```

四个实参各有明确来源：

```text
inputs.logits[0]       ← target model，经 gather 传来
matcher_               ← 创建请求时编译 grammar 并挂在 stream 上
committed_output_len_  ← 已经提交给 grammar 的输出 token 数
eos_token_id_          ← 模型配置
```

`state_mutex_` 把 `process()`、speculative prepare 和 `updateStatus()` 串行化，避免一个线程基于 `S0` 造 mask 时，另一个线程正把 matcher 推进到 `S1`。

更关键的是：`process()` **只查询 S0 并生成下一 token 的 allow-mask，不接受 token，也不推进 matcher**。

```text
进入 process：matcher=S0, committed_len=0
离开 process：matcher=S0, committed_len=0       # 不变
                         logits=L0'             # 变了
```

**本步向下调用：** `DecodeMaskBuilder::apply(row0 logits, S0, version=0, EOS)`。

### 8.7 `DecodeMaskBuilder::apply` 先决定复用哪一版 mask，还是向 XGrammar 重算

```cpp
ErrorInfo DecodeMaskBuilder::apply(
    const torch::Tensor& logits,
    RtpGrammarMatcher& matcher,
    int64_t accepted_token_len,
    int64_t eos_token_id) {
    if (device_mask_state_.mode != DeviceMaskMode::UNSET
        && device_mask_state_.token_len == accepted_token_len) {
        return applyDeviceMaskState(
            logits, device_mask_state_, eos_token_id);
    }

    auto state_or = buildState(matcher, accepted_token_len);
    if (!state_or.ok()) {
        device_mask_state_ = finishedState(accepted_token_len);
        applyDeviceMaskState(logits, device_mask_state_, eos_token_id);
        return state_or.status();
    }

    device_mask_state_ = std::move(state_or.value());
    return applyDeviceMaskState(
        logits, device_mask_state_, eos_token_id);
}
```

`accepted_token_len` 在这里是 cache version，而不是“让 matcher 前进多少步”的命令：

```text
cached mask version == committed_output_len_ ?

             yes                           no
              │                             │
              ▼                             ▼
     复用 device_mask_state_       buildState(matcher 当前状态)
              │                             │
              └────────────┬────────────────┘
                           ▼
              applyDeviceMaskState(logits)
```

为什么用 committed length 做版本？因为 matcher 每成功提交一批 token，`committed_output_len_` 都递增；旧状态 `S0` 的 mask 就不能误用于新状态 `S1`。

首次生成 `t0` 时还没有 cache，因此进入 `buildState()`：

```cpp
ErrorResult<DeviceMaskState>
DecodeMaskBuilder::buildState(
    RtpGrammarMatcher& matcher,
    int64_t accepted_token_len) {
    DeviceMaskState state;
    state.token_len = accepted_token_len;

    if (matcher.finished()) {
        state.mode = DeviceMaskMode::FINISHED;
        return ErrorResult<DeviceMaskState>(std::move(state));
    }
    auto terminated = matcher.isTerminated();
    if (terminated.value()) {
        state.mode = DeviceMaskMode::TERMINATED;
        return ErrorResult<DeviceMaskState>(std::move(state));
    }

    const int32_t grammar_vocab_size = matcher.vocabSize().value();
    if (grammar_vocab_size <= 0) {
        state.mode = DeviceMaskMode::NOOP;
        return ErrorResult<DeviceMaskState>(std::move(state));
    }

    auto bitmask = prepareBitmask(grammar_vocab_size);
    auto filled = fillMatcherBitmask(matcher, bitmask);

    state.mode = DeviceMaskMode::MASK;
    state.mask_required = filled.value();
    state.packed_allow_mask_cpu =
        filled.value() ? std::move(bitmask) : torch::Tensor{};
    state.grammar_vocab_size = grammar_vocab_size;
    return ErrorResult<DeviceMaskState>(std::move(state));
}
```

这里有几条语义分支：

```text
matcher 状态          DeviceMaskMode       下一步对 logits 做什么
────────────────────────────────────────────────────────────────
正常且有约束          MASK                 应用 packed allow-mask
本步允许整个词表      MASK/mask_required=0 不需要启动 mask kernel
grammar 已完整结束    TERMINATED           强制只允许 EOS
matcher 被标为完成    FINISHED             不再构造或应用 mask
无 grammar vocab      NOOP                 不处理
```

本例处于正常约束状态，于是 `prepareBitmask()` 先在 pinned CPU memory 中准备 `[1, ceil(vocab/32)]` 的 int32 buffer，再进入 matcher。

**本步向下调用：** `RtpGrammarMatcher::fillBitmask()`；返回后再把得到的 mask 应用到 `L0`。

### 8.8 `RtpGrammarMatcher::fillBitmask` 让 XGrammar 从 S0 计算“下一步允许谁”

源码：[RtpGrammarMatcher.cc](../../rtp_llm/cpp/engine_base/grammar/RtpGrammarMatcher.cc)

```cpp
ErrorResult<bool>
RtpGrammarMatcher::fillBitmask(DLTensor* bitmask, int32_t idx) {
    return matcherCall("fillBitmask", [&] {
        return matcher_->FillNextTokenBitmask(bitmask, idx);
    });
}
```

传下来的 `DLTensor` 指向 `DecodeMaskBuilder` 刚准备的 pinned CPU buffer。XGrammar 根据 matcher 当前自动机状态 `S0` 写 packed bits：

```text
                    XGrammar matcher S0
                            │
                            │ FillNextTokenBitmask
                            ▼
token id:       0  1  2  3  4  5 ... 31 │ 32 33 ...
allowed:        0  1  0  0  1  0 ...  0 │  1  0 ...
                └──────── int32 word0 ───┘ └ word1 ...

bit=1：这个 token 可作为 S0 的下一 token
bit=0：这个 token 会违反 structural grammar
```

这个调用是一次“只读查询”：

```text
FillNextTokenBitmask(S0)
        │
        ├─ 输出 allow-mask M0
        └─ matcher 仍然是 S0
```

只有第 9 节的 `acceptToken(t0)` 才会执行 `S0 → S1`。如果 fill mask 就提前推进，sampling 最后没选中预想 token、或请求取消时，parser 状态都会失真。

函数的 bool 返回值不是“token 合法/非法”，而是该行是否真的需要 mask；all-true 行可以跳过 GPU mask。

**本步返回给上层：** pinned CPU 上的 packed allow-mask `M0`，matcher 保持 `S0`。

### 8.9 回到 `DecodeMaskBuilder`：把 CPU 的 M0 上传并原地改写 GPU logits

XGrammar 返回后，控制流沿原调用栈回到 `applyDeviceMaskState()`：

```cpp
if (state.mask_required && mask_vocab_size > 0) {
    const int64_t words = state.packed_allow_mask_cpu.size(1);
    ...
    auto packed_allow_mask_gpu =
        reusable_bitmask_gpu_.narrow(1, 0, words);

    packed_allow_mask_gpu.copy_(
        state.packed_allow_mask_cpu,
        /*non_blocking=*/true);

    runtimeApplyPackedMaskLogits(
        logits,
        packed_allow_mask_gpu,
        mask_vocab_size);
}
```

至此 CPU matcher 路径和 GPU logits 路径真正汇合：

```text
CPU                                           GPU

matcher S0
   │ FillNextTokenBitmask
   ▼
pinned packed mask M0
   │ non_blocking copy
   └────────────────────────────────────────> M0_gpu
                                                │
raw logits L0 ──────────────────────────────────┤
                                                ▼
                                  runtimeApplyPackedMaskLogits
                                                │
                                                ▼
                                      masked logits L0'
```

源码：[mask_logits.cu](../../rtp_llm/models_py/bindings/common/kernels/mask_logits.cu)

```cpp
const int word_idx = vocab_idx / 32;
bool allowed = false;
if (word_idx < bitmask_words) {
    const uint32_t word = static_cast<uint32_t>(
        packed_allow_mask[
            compact_row * bitmask_row_stride + word_idx]);
    allowed = (word & (1u << (vocab_idx % 32))) != 0u;
}
if (!allowed) {
    logits_batch[
        logits_row * logits_row_stride + vocab_idx]
        = NegativeInfinity<T>();
}
```

每个 CUDA thread 负责检查一个 vocab token。合法项保留原值，非法项写成负无穷：

```text
toy vocab          raw L0       M0 allow bit       masked L0'
─────────────────────────────────────────────────────────────
token "a"           12.1            0                -inf
token "{"            8.3            1                 8.3
token "["            7.9            0                -inf
token EOS            1.2            0                -inf
```

它没有另造一份“masked logits 返回值”。`const torch::Tensor& logits` 的 const 只限制 Tensor handle，不阻止底层 storage 被 kernel 改写；因此修改直接留在 `SamplerInputs::logits` 中。

```text
调用前：inputs.logits[0] ──> storage L0
                                 │
                           CUDA 原地写入
                                 │
调用后：inputs.logits[0] ──> storage L0'
```

**本步返回给上层：** 没有新 tensor；返回的是成功/失败状态，真正的结果是已原地修改的 `L0'`。

### 8.10 控制流逐层返回，`execSampleGreedy` 最终看到的就是 L0'

现在把刚才“向下钻”的栈反向收回来：

```text
runtimeApplyPackedMaskLogits
    │ 完成：L0 → L0'
    ▼
DecodeMaskBuilder::apply
    │ return Ok
    ▼
GrammarLogitsProcessor::process
    │ return nullopt
    ▼
LogitsProcessorStates::batchProcess
    │ return processor_errors[row0]=nullopt
    ▼
Sampler::preprocessLogits
    │ return processor_errors
    ▼
Sampler::forward
    │ 继续执行下一行代码
    ▼
execSampleGreedy(logits view of L0')
    │
    └─ sample 得到 t0 = "{"
```

因此从 C++ 对象角度看，函数之间传的是三种不同形态的结果：

```text
setLogitsProcessorInputs   返回：调用计划（存在 SamplerInputs 中）
fillBitmask                返回：packed allow-mask（写进外部 buffer）
runtimeApply...            返回：void；副作用是原地修改 logits
process/batchProcess       返回：错误状态；数据结果仍在 logits 中
execSampleGreedy           返回：采样 token t0（写进 SamplerOutput）
```

这正是只逐个看函数容易觉得断裂的原因：核心数据并不总通过 C++ return value 传递，而是在共享的 Tensor/buffer 上通过副作用向后流动。

### 8.11 第 8 节只做“读状态并选 token”，第 9 节才做“提交 token 并推进状态”

本次 normal constrained sampling 结束时：

```text
SamplerOutput.token_ids 包含 t0
Grammar matcher 仍是 S0
committed_output_len_ 仍是 0
cached mask 是 version 0 的 M0
```

紧接着第 9 节的 dispatch/update 才会提交 `t0`：

```text
第 8 节：mask/read phase                 第 9 节：commit/write phase

matcher S0                              sampled t0
   │ FillNextTokenBitmask                   │
   ├──────────────> M0                       ▼
   │                                  GenerateStream::specUpdate
   │                                         │ updateLogitProcessorStatus
   │                                         ▼
   └────────────────────────────────> matcher.acceptToken(t0)
                                             │
                                             ▼
                                         matcher S1
                                  committed_output_len_: 0 → 1
```

两阶段分开的必要性与数据库的“先校验、后提交”类似：

```text
1. S0 只负责回答：哪些候选合法？
2. sampler 在合法候选中真正选出 t0。
3. 只有确定 t0 会成为 stream 输出后，才能把它提交给 matcher。
```

如果在 `fillBitmask()` 时就推进 matcher，系统还不知道 sampler 最终选哪个 token；如果 sampled 后从不 `updateStatus()`，下一轮仍会错误复用 S0 的语法位置。这也解释了为什么第 9 节讨论的 `updateLogitProcessorStatus()` 不是可有可无的 bookkeeping，而是本节约束采样的状态闭环。

最后用一张图把第 8 节所有函数放回同一条流水线：

```text
                        ┌──────────── CPU control/state ─────────────┐
                        │                                            │
GenerateStream          │  LogitsProcessorStates                     │
  matcher=S0 ───────────┼─> invocation [processor,row0]              │
                        │          │                                 │
                        │          ▼                                 │
                        │  GrammarLogitsProcessor                     │
                        │          │ (L0,S0,version0)                 │
                        │          ▼                                 │
                        │  DecodeMaskBuilder                         │
                        │          │                                 │
                        │          ▼                                 │
                        │  XGrammar FillNextTokenBitmask(S0)         │
                        │          │                                 │
                        │       M0_cpu                               │
                        └──────────┼──────────────────────────────────┘
                                   │ H2D
                                   ▼
                        ┌────────────── GPU data ─────────────────────┐
target model ──> L0 ───>│ runtimeApplyPackedMaskLogits(L0, M0_gpu)   │
                        │                   │                         │
                        │                   ▼                         │
                        │                  L0'                        │
                        │                   │                         │
                        │             execSampleGreedy               │
                        │                   │                         │
                        │                   ▼                         │
                        │                  t0                         │
                        └───────────────────┼─────────────────────────┘
                                            │
                                            ▼
                              第 9 节：commit t0，S0 → S1
```

## 9. Prefill 首 token 如何提交并触发 PD handoff

### 9.1 `MtpBatchStreamProcessor::dispatchPrefill`

源码：[MtpBatchStreamProcessor.cc](../../rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.cc)

```cpp
preparePrefillSpecUpdateInfo(..., spec_update_infos);
updateProposeTokens(..., spec_update_infos);
stream_groups.updateStreams(spec_update_infos);
```

这里构造：

```text
StreamSpecUpdateInfo
├── new_tokens       = [t0]
├── num_new_tokens   = 1
├── draft_token      = d1
├── draft_probs      = P_draft(d1)
└── draft_hidden     = H_draft
```

### 9.2 `StreamGroups::updateStreams`

```cpp
for (auto& stream : context_streams_) {
    stream->specUpdate(spec_update_infos[stream_idx++]);
}
```

### 9.3 `GenerateStream::specUpdate`

```cpp
complete_token_ids_->update(new_tokens, ..., num_new_tokens, ...);

int accept_token_num = (seqLength() - 1) - cur_cached_len;
if (auto error = updateLogitProcessorStatus(
        new_tokens, accept_token_num); error.has_value()) {
    reportEventWithoutLock(StreamEvents::Error, ...);
    return;
}

sp_output_buffer_->tokens[0] = target_last_token;
sp_output_buffer_->tokens[1] = update_info.draft_token;

updateOutput({new_tokens, num_new_tokens, ...});
```

这里建立核心不变量：

```text
CompleteTokenIds 写入成功
          │
          ▼
processor updateStatus 恰好一次
          │
          ▼
SP buffer / output 才对外可见
```

### 9.4 `NormalGenerateStream::updateOutput`

源码：[NormalGenerateStream.cc](../../rtp_llm/cpp/normal_engine/NormalGenerateStream.cc)

```cpp
finished_ = needFinish();
if (finished_) {
    reportEventWithoutLock(StreamEvents::GenerateDone);
}

if (!finished_ && queryPdSep() && update_info.update_remote_generate) {
    holdKVCacheForPDSep();
    reportEventWithoutLock(StreamEvents::NeedRemoteGenerate);
    reportEventWithoutLock(StreamEvents::GenerateDone);
}

enqueueGenerateOutput(prepareGenerateOutput(update_info));
```

讲解：

- Prefill 的 stream 在本地被标记 `GenerateDone`，但只是本地阶段结束；
- `NeedRemoteGenerate` 告诉 RPC 层把执行权交给 Decode；
- 这正是“不能因为 `is_done` 就跳过最后一次 processor update”的一个原因：这里的 done 不是整个请求结束。

## 10. PD handoff：把首 token 和 MTP 状态送到 Decode

### 10.1 先把 PD 看成“三阶段所有权交接”

它很像 TCP 三次握手，但交接的不是连接序号，而是一个推理请求的三类状态：

```text
阶段 1：ALLOCATE
确认 Decode 有资源接这个请求。

阶段 2：LOAD
确认 Decode 已经收到 Prefill 产生的 KV cache。

阶段 3：GENERATE
把首 token、MTP continuation state 交给 Decode，正式转移生成权。
```

三阶段分别建立三个不变量：

```text
ALLOCATE ACK
    => Decode stream/matcher 已创建，目标 KV blocks 已分配

LOAD ACK
    => Decode 的目标 blocks 已装入 Prefill KV，Prefill 源 blocks 可以释放

GENERATE
    => Decode 已拿到 t0 + draft state，可以重放 matcher 并进入 decode scheduler
```

把它类比成搬家：

```text
ALLOCATE：先确认新房存在，而且放得下家具
LOAD：    把大件家具搬进新房，收货方签收
GENERATE：交钥匙和当前生活状态，正式在新房继续生活
```

如果跳过任何一步：

```text
没 ALLOCATE 就 Prefill：算完才发现 Decode OOM，前面的 GPU 计算全部浪费
没 LOAD ACK 就释放源 KV：Decode 可能读取已回收/复用的 KV，形成 use-after-free
没等 LOAD 完就 GENERATE：Decode kernel 会读取未完成的 KV，结果错误或崩溃
没发送 t0 就启动 Decode：Decode matcher/MTP 状态比 Prefill 少推进一个 token
```

### 10.2 一条 gRPC 双向流上的完整报文时序

Prefill 与 Decode 之间只建立一条 `RemoteGenerate` 双向流，然后按固定顺序交换三组消息：

```text
Client        Prefill RPC        Prefill Engine       Cache Store        Decode RPC        Decode Engine
  │                │                   │                   │                  │                   │
  │ GenerateInput  │                   │                   │                  │                   │
  ├───────────────>│                   │                   │                  │                   │
  │                │ open RemoteGenerate bidi stream      │                  │                   │
  │                ├─────────────────────────────────────────────────────────>│                   │
  │                │                   │                   │                  │                   │
  │                │ ① ALLOCATE{input, peer_addrs}         │                  │                   │
  │                ├─────────────────────────────────────────────────────────>│ makeStream         │
  │                │                   │                   │                  ├──────────────────>│
  │                │                   │                   │                  │ create matcher      │
  │                │                   │                   │                  │ allocate KV blocks  │
  │                │                   │                   │                  │<───────────────────┤
  │                │ ① ALLOCATE ACK                         │                  │                   │
  │                │<─────────────────────────────────────────────────────────┤                   │
  │                │                   │                   │                  │                   │
  │                │ enqueue Prefill   │                   │                  │                   │
  │                ├──────────────────>│                   │                  │                   │
  │                │ wait stream leaves WAITING            │                  │                   │
  │                │                   │                   │                  │                   │
  │                │ ② LOAD trigger                         │                  │                   │
  │                ├─────────────────────────────────────────────────────────>│                   │
  │                │                   │ target prefill     │<──pull/wait KV───┤ load into blocks  │
  │                │                   │ grammar sample t0 │──publish prompt KV>│                   │
  │                │                   │ draft sample d1   │                  │                   │
  │                │<──local t0 output─┤                   │                  │                   │
  │ first token t0 │                   │                   │                  │                   │
  │<───────────────┤                   │                   │                  │                   │
  │                │                   │                   │                  │                   │
  │                │ ② LOAD ACK                            │                  │                   │
  │                │<─────────────────────────────────────────────────────────┤ KV fully loaded    │
  │                │ release Prefill KV reference          │                  │                   │
  │                │                   │                   │                  │                   │
  │                │ ③ GENERATE{t0,d1,probs,hidden}        │                  │                   │
  │                ├─────────────────────────────────────────────────────────>│ replay t0          │
  │                │                   │                   │                  ├──────────────────>│ enqueue decode     │
  │                │                   │                   │                  │                   │ MTP decode rounds │
  │                │                   │                   │                  │<───────────────────┤ output t1..tn      │
  │                │<─────────────────────────────────────────────────────────┤                   │
  │ t1..tn         │ relay output      │                   │                  │                   │
  │<───────────────┤                   │                   │                  │                   │
```

最重要的两个并发点：

```text
并发点 A
Prefill engine 在算 t0
Decode 端已经分配好目标 blocks，并开始等待/搬运 KV

并发点 B
客户端可以先收到 t0
Prefill RPC 随后才等待 LOAD ACK、发送 GENERATE handoff
```

因此 TTFT 不必等待整个 Decode 接管流程结束。

### 10.3 第一阶段：ALLOCATE，先确认“接收方有地方放”

Prefill 打开双向 RPC 流并发送：

```cpp
prefill_context.client_stream =
    grpc_connection.stub->RemoteGenerate(client_context);

GenerateRequestPB request;
request.set_stage(RemoteStage::ALLOCATE);
request.set_client_id(process_id_);
request.set_request_id(request_id);
request.set_allocated_input(new GenerateInputPB(original_request));
request.add_peer_addrs(prefill_cache_store_addr);

client_stream->Write(request);
client_stream->Read(&allocate_response);
```

ALLOCATE 带两类信息：

```text
GenerateInputPB
├── prompt tokens
├── generation config
├── structural/json/regex constraint
├── think config
└── request identity

peer_addrs
└── 后续 Decode 从哪些 Prefill cache-store endpoint 拉 KV
```

Decode 收到后不是立刻 enqueue：

```cpp
auto input           = QueryConverter::transQuery(&allocate_request.input());
auto generate_stream = engine_->makeStream(input);

generate_stream->reportEvent(StreamEvents::CanRun);
while (!generate_stream->hasError()
       && generate_stream->moveToNext() == StreamState::LOADING_CACHE) {
    sleep(1ms);
}

grpc_stream->Write(GenerateOutputsPB()); // ALLOCATE ACK
```

状态图：

```text
收到 ALLOCATE
      │
      ▼
engine_->makeStream
      │
      ├─ create CompleteTokenIds
      ├─ create Decode-side Grammar/Reasoning matcher
      └─ create StreamCacheResource
      │
      ▼
CanRun + 手工驱动 state machine
      │
      ▼
分配 Decode KV blocks
      │
      ├─ 失败 -> RESOURCE_EXHAUSTED，Prefill 不开始昂贵计算
      └─ 成功 -> ALLOCATE ACK
```

为什么必须在 Prefill enqueue 之前完成：

```text
正确顺序：
Decode reserve success -> Prefill compute

反向顺序：
Prefill compute 100ms -> Decode reserve OOM -> 100ms GPU 结果作废
```

它也是 grammar 的 fail-fast 点：Decode matcher 在这里创建；schema 编译失败，不应该等 KV 已经传完才发现。

### 10.4 第二阶段：LOAD，先启动接收，再等待生产完成

ALLOCATE ACK 后，Prefill 才 enqueue 本地请求：

```cpp
auto stream = engine_->enqueue(prefill_context.generate_input);
prefill_context.setStream(stream);
```

接着 `remoteLoadCacheStart()` 等 Prefill stream 不再是 `WAITING`：

```cpp
while (!stream->hasError()
       && stream->getStatus() == StreamState::WAITING) {
    usleep(100);
}

GenerateRequestPB load_request;
load_request.set_request_id(request_id);
load_request.set_start_time(currentTimeUs());
client_stream->Write(load_request);
```

为什么不是等 t0 完全算好以后才发 LOAD：

```text
较晚启动：
[Prefill 完整计算] -> [建立 load 上下文/连接] -> [传 KV]

当前顺序：
[Prefill 进入 RUNNING------------------------]
        [Decode 建立 load 上下文/等待 KV------]
                         [KV ready/transfer----]
```

接收端先准备好，可以隐藏连接、buffer 注册、等待等开销；具体能否按层边产边传取决于 cache-store 实现，但协议不要求 Decode 等到 GENERATE 才开始准备 LOAD。

Decode 的第二次 `Read()` 固定被解释为 load trigger：

```cpp
GenerateRequestPB load_request;
grpc_stream->Read(&load_request);

auto error_info = loadCacheForAllRank(decode_context);

GenerateOutputsPB load_response;
load_response.mutable_error_info()->set_error_code(...);
grpc_stream->Write(load_response); // LOAD ACK
```

KV 数据不经过这条 gRPC 控制流：

```text
gRPC control plane
PrefillRpcServer ──LOAD trigger/ACK── DecodeRpcServer

cache data plane
Prefill CacheStore ===== huge KV tensors =====> Decode CacheStore/KV blocks
```

这样设计是因为 KV 远大于 token/MTP metadata：

```text
控制消息：request id、地址、token，通常很小
KV 数据：每层、每 block 的 K/V，可能是 MB/GB 级
```

把 KV 塞进普通 gRPC protobuf 会增加序列化、复制和内存峰值，也无法充分使用专门的 RDMA/cache-store 通道。

### 10.5 为什么首 token 可以先返回，但 Prefill KV 不能先释放

Prefill 发出 LOAD trigger 后调用：

```cpp
pollLocalOutput(..., prefill_stream);
```

Prefill engine 产生 t0 时，`NormalGenerateStream::updateOutput()` 会：

```cpp
if (!finished_ && queryPdSep() && update_remote_generate) {
    holdKVCacheForPDSep();
    reportEventWithoutLock(StreamEvents::NeedRemoteGenerate);
    reportEventWithoutLock(StreamEvents::GenerateDone);
}
```

这里的 `GenerateDone` 只表示：

```text
Prefill 本地生成职责完成，可以停止 Prefill stream
```

不表示：

```text
整个用户请求结束
```

状态分叉：

```text
                         t0 产生
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
输出路径                               KV 生命周期路径
enqueue local output t0                holdKVCacheForPDSep
             │                             │ 增加 connector ref
             ▼                             ▼
Client 立即看到 TTFT                 Decode 仍可安全读取源 KV
```

Prefill 的调用顺序是：

```cpp
pollLocalOutput(prefill_context);     // t0 可先写给 Client
remoteLoadCacheEnd(prefill_context);  // 再等待 Decode LOAD ACK
```

收到 LOAD ACK 后才执行：

```cpp
prefill_stream->releaseKVCacheForPDSep();
```

引用计数屏障：

```text
Prefill stream 逻辑结束
      │
      ├─ 普通 stream resource 可以进入释放流程
      │
      └─ pd_kvcache_ref_ 仍持有 connector blocks
                         │
                         │ Decode load in progress
                         ▼
                    收到 LOAD ACK
                         │
                         ▼
                  reset pd_kvcache_ref_
                         │
                         ▼
                    源 KV 才可回收
```

这和网络协议里的 ACK 很像：发送方不能因为“数据已经发起发送”就回收 send buffer，必须确认接收方已经消费完成。

### 10.6 第三阶段：GENERATE，正式提交生成权

只有 LOAD ACK 成功后，Prefill 才发送 `RemoteStage::GENERATE`：

```cpp
GenerateRequestPB generate_request;
generate_request.set_stage(RemoteStage::GENERATE);
generate_request.set_first_generate_token_id(t0);
generate_request.mutable_propose_token_ids()->CopyFrom({t0, d1});
transTensorPB(generate_request.mutable_propose_probs(), draft_probs);
transTensorPB(generate_request.mutable_propose_hidden(), draft_hidden);

client_stream->Write(generate_request);
```

三类 handoff 数据各有不同作用：

```text
t0
├── 补 Decode token history
├── 推进 Decode grammar/reasoning matcher
└── 告诉 Decode“Prefill 已经输出到哪里”

[t0, d1] + draft probs
├── 恢复 MTP proposal buffer
└── 让 Decode 第一轮不必丢掉 Prefill 已算的 draft continuation

draft hidden
└── 恢复下一步 MTP draft model 所需 hidden carrier
```

这一步相当于所有权提交：

```text
GENERATE 之前
  Prefill 是 logical generation owner
  Decode 只有 empty stream + loaded KV

GENERATE 之后
  Decode replay t0，成为 logical generation owner
  Prefill 只负责 relay Decode outputs
```

### 10.7 Decode 为什么要“重放 t0”，不能只拿 KV 开跑

KV cache 只包含模型 attention state，不包含所有 engine 状态：

```text
KV cache 包含
└── target/draft model 的历史计算结果

KV cache 不包含
├── CompleteTokenIds 的逻辑输出位置
├── GrammarMatcher parser state
├── Think state / token budget
├── stop/eos 状态
├── last_output_pos
└── MTP proposal metadata
```

因此 Decode 必须：

```cpp
new_tokens[0][0] = generate_request.first_generate_token_id();
generate_stream->incLastOutputPos();
generate_stream->update({new_tokens, 1, ...});
```

`update([t0])` 的效果：

```text
Decode CompleteTokenIds: prompt -> prompt+t0
Decode matcher:          S0     -> accept(t0) -> S1
Decode seqLength:        L      -> L+1
```

`incLastOutputPos()` 的效果：

```text
t0 已由 Prefill 返回给 Client
Decode output cursor 跳过 t0
后续只输出 t1, t2, ...
```

如果不重放：

```text
模型 KV：prompt history 已就绪，下一次 decode 输入应从 t0 接续
grammar state：若不 replay t0，仍停在输出起点

下一轮 mask 会基于错误 parser state 生成。
```

### 10.8 Decode 为什么到最后才 enqueue

`DecodeRpcServer::allocateResource()` 只创建 stream 和分配 blocks，不加入 scheduler。`localGenerate()` 完成重放和 MTP 恢复后才：

```cpp
generate_stream->setSPOutputBuffer(sp_output_buffer);
engine_->enqueue(generate_stream);
```

调度屏障：

```text
Decode stream created
      │
      ├─ KV not loaded       -> 禁止 enqueue
      ├─ t0 not replayed     -> 禁止 enqueue
      ├─ MTP state missing   -> 禁止 enqueue
      │
      ▼
LOAD ACK + GENERATE payload restored
      │
      ▼
engine_->enqueue
      │
      ▼
MtpExecutor::decodeStep
```

否则 scheduler 只知道“有个 runnable stream”，并不知道它的 KV/token/matcher/MTP 状态还处在半初始化状态。

### 10.9 三阶段之间真正的时间公式

简化时间线：

```text
time ─────────────────────────────────────────────────────────────────>

ALLOCATE: [Decode makeStream + reserve blocks]

Prefill :                    [target prefill + grammar sample t0 + d1]
Load    :                    [setup/wait-----------KV transfer-------]
Client  :                                      t0 ▲
Load ACK:                                                       ▲
Generate:                                                        [handoff]
Decode  :                                                         [decode rounds...]
```

两个用户可见时间：

```text
TTFT ≈ T_route + T_allocate + T_prefill_to_t0

Decode 真正开始时间
     ≈ T_route + T_allocate
       + max(T_prefill_handoff_ready, T_load_done)
       + T_generate_handoff
```

为什么用 `max` 而不是相加：Prefill 计算与 Decode 的 load setup/wait 是有重叠的。

### 10.10 为什么不能把三阶段合成一个 RPC 消息

假设只发一个 `START{input}`：

```text
Decode 收到 START
  ├─ allocate blocks
  ├─ 等 Prefill 产生 KV
  ├─ load KV
  └─ 等 t0/MTP state
```

问题是接收方无法区分：

```text
资源是否已经预留成功？
KV 是否已经完整到达？
Prefill 是否已经算出 t0？
源 KV 是否可以释放？
Decode 是否可以进入 scheduler？
```

拆成三个 ACK/barrier 后，每个阶段只有一个明确回答：

```text
ALLOCATE ACK：有地方
LOAD ACK：数据在地方里
GENERATE：状态完整，可以运行
```

这就是它像“三次握手”的根本原因：不是为了可靠网络本身，而是为了可靠地迁移分布式状态所有权。

### 10.11 当前实现的一个协议细节

proto 定义了：

```proto
enum RemoteStage {
    ALLOCATE = 0;
    LOAD = 1;
    GENERATE = 2;
}
```

但当前 `remoteLoadCacheStart()` 没有显式调用：

```cpp
load_request.set_stage(RemoteStage::LOAD);
```

Decode 的 `loadCacheFromPrefill()` 也没有校验第二条消息的 stage，而是依赖同一双向流上的固定读取位置：

```text
第 1 条 Read = ALLOCATE
第 2 条 Read = LOAD trigger
第 3 条 Read = GENERATE
```

ALLOCATE 和 GENERATE 都有显式 stage 校验，LOAD 当前主要靠“第二条消息”这个顺序协议识别。功能上可工作，但从协议自描述性和防错性看，更完整的实现应显式设置并校验 `RemoteStage::LOAD`。

### 10.12 `PrefillRpcServer::remoteGenerate` 的 payload

```cpp
vector<int> all_token = stream->currentExecuteTokens();
int first_token = all_token.back();

generate_request.set_first_generate_token_id(first_token);
generate_request.mutable_propose_token_ids()->CopyFrom(
    {stream->getProposeToken().begin(), stream->getProposeToken().end()});

QueryConverter::transTensorPB(
    generate_request.mutable_propose_probs(), sp_output_buffer->all_probs);
QueryConverter::transTensorPB(
    generate_request.mutable_propose_hidden(), sp_output_buffer->hidden_states);
```

传输内容：

```text
GenerateRequestPB::GENERATE
├── first_generate_token_id = t0
├── propose_token_ids       = [t0, d1]
├── propose_probs           = draft probability
├── propose_hidden          = draft hidden state
└── position_ids            = multimodal/MRoPE position state（如有）
```

matcher 本身没有序列化传输。

### 10.13 `DecodeRpcServer::allocateResource` 后的 matcher 状态

源码：[DecodeRpcServer.cc](../../rtp_llm/cpp/model_rpc/DecodeRpcServer.cc)

```cpp
auto input = QueryConverter::transQuery(&allocate_request.input());
auto generate_stream = engine_->makeStream(input);
generate_stream->reportEvent(StreamEvents::CanRun);
decode_context.setStream(generate_stream);
```

ALLOCATE ACK 刚返回时，Prefill 还没有产生 t0；两端 matcher 都在输出起点，但它们是两个独立对象：

```text
Matcher(P): S0，已接受 []
Matcher(D): S0，已接受 []
```

等 Prefill 产生 t0、LOAD 完成、即将发送 GENERATE 时：

```text
Matcher(P): S1，已接受 [t0]
Matcher(D): S0，已接受 []
```

### 10.14 `DecodeRpcServer::localGenerate`

```cpp
generate_stream->setIsContextStream(false);

auto new_tokens = torch::zeros({1, 1}, torch::kInt32);
new_tokens[0][0] = generate_request.first_generate_token_id();

generate_stream->incLastOutputPos();
generate_stream->update({new_tokens, 1, ...});

generate_stream->setMtpTokenIndex(generate_stream->seqLength() - 1);
generate_stream->setContainProposeToken(true);
generate_stream->setProposeToken(propose_tokens);
generate_stream->setSPOutputBuffer(sp_output_buffer);

engine_->enqueue(generate_stream);
```

讲解：

- `update([t0])` 不只是写 token history，还调用 Decode 端 processor 的 `updateStatus([t0])`；
- 这样 Decode matcher 被重放到和 Prefill matcher 等价的 committed 状态；
- `incLastOutputPos()` 防止 Decode 再把 Prefill 已返回的 `t0` 输出一次；
- `[t0,d1]` 与 draft probs/hidden 恢复 MTP 下一轮所需状态。

重放图：

```text
                 network payload
Matcher(P) ─────── token t0 ───────────────┐
 state S1                                  │
                                           ▼
                                    Matcher(D) state S0
                                    updateStatus(t0)
                                           │
                                           ▼
                                    Matcher(D) state S1
```

## 11. Decode 端一轮 MTP：完整过程

第 10 章结束时，Decode 端不是从空白状态开始，而是已经拿到了一个可以直接进入 MTP 的完整现场：

```text
第 10 章 PD GENERATE handoff 的输出

Decode GenerateStream
├── CompleteTokenIds = prompt + t0
├── matcher           = S1                  # 已 replay/commit t0
├── SPOutputBuffer
│   ├── tokens[0]     = t0                  # 上个 committed target token
│   ├── tokens[1]     = d1                  # Prefill 顺手算出的首个 draft
│   ├── all_probs     = P_draft(d1)
│   └── hidden_states = H_draft
└── KV cache          = 已从 Prefill 加载
                         │
                         ▼
                MtpExecutor::decodeStep
```

从第 11 章到第 16 章，实际上只是在追踪 **同一轮** `decodeStep()`。各章之间传递的接力棒如下：

```text
┌──────────────────────────────── 一轮 MTP ────────────────────────────────┐
│                                                                         │
│ 第 11 章：构造候选与 target verify                                      │
│   输入  SP=[Tprev,d1]                                                   │
│   输出  draft_tokens=[d1..dP] + target_logits=[L0..LP]                  │
│                         │                                               │
│                         ▼                                               │
│ 第 12 章：XGrammar provisional verify                                   │
│   输入  draft_tokens + committed matcher S                             │
│   输出  masks=[M0..MP] + grammar cap；matcher rollback 回 S             │
│                         │                                               │
│                         ▼                                               │
│ 第 13 章：mask target logits 并逐行采样                                 │
│   输入  [L0..LP] + [M0..MP]                                            │
│   输出  合法 target samples/probabilities                               │
│                         │                                               │
│                         ▼                                               │
│ 第 14 章：概率接受与 grammar cap 汇合                                   │
│   输入  draft proposal + target distribution + cap                     │
│   输出  本轮 actual accepted tokens=[a1..aN]                            │
│                         │                                               │
│                         ▼                                               │
│ 第 15 章：永久提交                                                      │
│   输入  [a1..aN]                                                        │
│   输出  stream history 追加；matcher S→S'；生成下一轮 SP                │
│                                                                         │
│ 第 16 章：不是额外阶段，而是展示上述 12→15 如何跨过 think boundary     │
└─────────────────────────────────────────────────────────────────────────┘
```

换成 `decodeStep()` 内部的调用顺序：

```text
MtpExecutor::decodeStep
│
├─ draftModelDecode                         ───── 第 11 章前半
│    └─ [Tprev,d1] 扩成 [Tprev,d1..dP]
│
├─ target model::forward(is_target_verify)  ───── 第 11 章后半
│    └─ 产生 [L0..LP]
│
├─ runSpecLogitsVerify                      ───── 第 12 章
│    └─ 用 draft + matcher 产生 [M0..MP], cap
│
├─ gatherSpecSamplerInput + Sampler         ───── 第 13 章
│    └─ [Li &= Mi] 后逐行采 target token
│
├─ SpeculativeSampler + applySpecVerify     ───── 第 14 章
│    └─ 概率结果与 cap 合并为 [a1..aN]
│
└─ dispatchDecode → specUpdate              ───── 第 15 章
     └─ 永久 append/commit，并准备 round N+1
```

后面每一小节都沿用这条主线，不重新定义一套局部流程。

先定义符号：

```text
P       = propose_step
Tprev   = 上轮最后一个 committed target token
d1..dP = draft model 提议 token
L0..LP = target model 并行产生的 P+1 行 logits
M0..MP = XGrammar 对应每个位置的 allow mask
cap     = grammar 允许的最长 draft 前缀长度，范围 [0,P]
```

以 `P=3` 为例：

```text
target verify input rows

row 0: prefix + Tprev            -> L0，预测 d1/fallback t1
row 1: prefix + Tprev + d1       -> L1，预测 d2/fallback t2
row 2: prefix + Tprev + d1+d2    -> L2，预测 d3/fallback t3
row 3: prefix + Tprev + d1+d2+d3 -> L3，预测 bonus t4

XGrammar masks

M0 = allowed(state S0)
M1 = allowed(state S0 + d1)
M2 = allowed(state S0 + d1 + d2)
M3 = allowed(state S0 + d1 + d2 + d3)
```

总体过程：

```text
          ┌──────────────────────┐
          │ committed state S0   │
          └──────────┬───────────┘
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
draft model                    target model
产生 d1,d2,d3                  算 L0,L1,L2,L3
       │                           │
       ▼                           │
XGrammar provisional verify       │
产生 M0,M1,M2,M3 + cap            │
并 rollback 回 S0                  │
       │                           │
       └─────────────┬─────────────┘
                     ▼
             Li &= Mi; target sample
                     │
                     ▼
          probability rejection sampling
                     │
                     ▼
              grammar cap 修正
                     │
                     ▼
        actual accepted tokens = a1..aN
                     │
                     ▼
         updateStatus(a1..aN): S0 -> SN
```

### 11.1 `MtpExecutor::decodeStep`

源码：[MtpExecutor.cc](../../rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc)

精简骨架：

```cpp
model_input = batch_stream_processor_->gatherDecodeModelInput(stream_groups);

prepareDecodeDraftModelInput(stream_groups, model_input);
draftModelDecode(model_input, stream_groups,
                 draft_probs_list, draft_token_ids_t);

model_input.is_target_verify = true;
model_output = model_->forward(model_input);

spec_logits_result = runSpecLogitsVerifyIfNeeded(
    streams, model_input, draft_sampler_output, draft_token_ids_t);

auto sampler_input = batch_stream_processor_->gatherSpecSamplerInput(
    stream_groups, model_input, model_output, spec_logits_result);
sampler_output = sampler_->forward(sampler_input);

auto spec_output = speculative_sampler_->forward(
    streams, draft_sampler_output, sampler_output);
applySpecVerifyResult(spec_logits_result, sampler_output,
                      spec_output, propose_step_);

batch_stream_processor_->dispatchDecode(
    stream_groups, spec_output, draft_prefill_output);
```

不要把上面看成一串平级函数。它内部形成两次“分叉—汇合”：

```text
第一次分叉：为 target verify 准备两类输入

stream committed history ── gatherDecodeModelInput ──> base model_input
SP buffer [Tprev,d1] ────── draftModelDecode ─────────> d1..dP
                                      │
                                      └──── merge ────> target verify input

第二次分叉：同时解释 target verify 的结果

target verify input ── target model ──> logits L0..LP ────────┐
draft d1..dP + matcher S ── XGrammar ──> masks M0..MP + cap ──┤
                                                               ▼
                                                          constrained sample
```

从这一层往下读时，可以把 `decodeStep()` 看作总导演；后面的函数只是在生产或消费其中一根箭头上的对象。

### 11.2 `MtpBatchStreamProcessor::prepareDecodeDraftModelInput`

```cpp
for (const auto& stream : stream_groups.allStreams()) {
    int propose_token = stream->getSPOutputBuffer()->tokens[1];
    combo_tokens[batch_idx] = propose_token;
}
model_input.combo_tokens = std::move(combo_tokens);
```

`SPOutputBuffer.tokens` 的语义：

```text
tokens[0] = 上轮最后 accepted target token
tokens[1] = 为本轮准备好的第一个 draft token
```

它消费的是上一轮（第一轮则是 PD Prefill）留下的 SP buffer，并把 `d1` 放进本轮 draft model input：

```text
上一轮 dispatch/PD handoff
       │
       ▼
SPOutputBuffer [Tprev,d1,prob,hidden]
       │
       ├─ tokens[1]=d1 ───────────────> model_input.combo_tokens
       ├─ hidden_states ──────────────> draft model hidden carrier
       └─ all_probs ──────────────────> rejection 所需 draft probability
                                                │
                                                ▼
                                      本轮 draftModelDecode
```

所以 `prepareDecodeDraftModelInput()` 不是在“提议 d1”；`d1` 已经由上一轮准备好了。它只是把上一轮的尾巴接成本轮的开头。

### 11.3 `MtpExecutor::draftModelDecode`

```cpp
draft_token_ids_list.push_back(pre_target_token);  // Tprev
draft_token_ids_list.push_back(pre_propose_token); // d1

for (int i = 0; i < propose_step_ - 1; ++i) {
    auto output = draft_model_->forward(model_input);
    auto sampled = fast_topk_sampler_->forward(output.logits, 1);
    draft_token_ids_list.push_back(sampled.token_ids); // d2 ... dP
    updateDecodeDraftModelInput(model_input, output, sampled.token_ids);
}

draft_token_ids_t = torch::cat(draft_token_ids_list, 1); // [B,P+1]
model_input.combo_tokens = draft_token_ids_t.reshape({B * (P + 1)});
```

图：

```text
SP buffer
[Tprev, d1]
          │
          ▼
draft forward #1 -> d2
          │
          ▼
draft forward #2 -> d3
          │
          ▼
[Tprev, d1, d2, d3]  (P=3)
          │
          ▼
target model 一次 verify forward
```

这里要区分两个方向：draft model 是自回归地串行产生 `d2..dP`，target model 随后才把完整 proposal chain 一次并行 verify：

```text
Host / draft loop                      token chain

从 SP 取出                                [Tprev,d1]
draft forward #1(d1)      ───────────>   [Tprev,d1,d2]
draft forward #2(d2)      ───────────>   [Tprev,d1,d2,d3]
                                               │
                                               ▼ reshape/flatten
                                      model_input.combo_tokens
```

**本函数交给后续两位消费者的是同一条 `draft_token_ids_t`：**

```text
draft_token_ids_t=[Tprev,d1,d2,d3]
             │
             ├─> target model：作为 P+1 个 verify positions
             │
             └─> XGrammar runner：跳过 Tprev，检查 d1,d2,d3
```

也就是说，target 与 grammar 不是各自拿了一套 proposal；它们对同一套 `d1..dP` 给出“概率判断”和“结构判断”。

### 11.4 Target model verify

```cpp
model_input.is_target_verify = true;
model_output = model_->forward(model_input);
model_input.is_target_verify = false;
```

Target model 把串行 decode 变成一次长度 `P+1` 的 verify/prefill：

```text
普通 decode：
GPU: [forward 1] -> [forward 1] -> [forward 1] -> [forward 1]

MTP target verify：
GPU: [----------- one forward over P+1 positions -----------]
```

verify 输出行与 draft token 的位置必须严格对齐：

```text
输入 position                    target 输出行       后续用途
────────────────────────────────────────────────────────────────
Tprev                            L0                 判断/替换 d1
Tprev + d1                       L1                 判断/替换 d2
Tprev + d1+d2                    L2                 判断/替换 d3
Tprev + d1+d2+d3                 L3                 bonus token

同一位置的 grammar mask：
S                                M0                 约束 L0
S + d1                           M1                 约束 L1
S + d1+d2                        M2                 约束 L2
S + d1+d2+d3                     M3                 约束 L3
```

因此第 11 章结束时，只有“候选”和“模型分数”，还不能采样：

```text
第 11 章输出
├── draft_token_ids_t = [Tprev,d1..dP]
└── model_output.logits = [L0..LP]
                              │
                              │ 还缺每一行对应的合法集合
                              ▼
                        第 12 章 XGrammar
```

## 12. MTP 与 XGrammar 的结合点

第 11 章留下两份数据：proposal chain `[d1..dP]` 和 target logits `[L0..LP]`。第 12 章先不碰 target logits，只消费 proposal chain，并从 committed matcher `S` 派生出与每个 `Li` 对齐的 mask `Mi`。

```text
第 11 章输出                            GenerateStream processor

draft_token_ids_t                        committed matcher S
[Tprev,d1,d2,...,dP]                            │
       │                                        │
       └──────────────┬─────────────────────────┘
                      ▼
            SpecLogitsVerifyRunner
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
packed masks [M0..MP]          grammar cap
          │                        │
          └───────────┬────────────┘
                      ▼
                第 13/14 章

matcher S ── provisional accept/rollback ──> matcher S
                                                    # 永久状态不变
```

本章内部也有一条清晰的调用链：

```text
MtpExecutor::runSpecLogitsVerify
    │ 收集 active processors + draft tensor
    ▼
SpecLogitsVerifyRunner::run
    │ D2H draft；分配/初始化 mask buffer
    ▼
mergeProcessorMasks
    │ 对每个 stream/processor 发 request
    ▼
GrammarLogitsProcessor::prepareSpeculative
    │ 锁住 matcher
    ▼
verifyDraftPrefixAndFillBitmask
    │ fill M0；try d1；fill M1；try d2；...
    ▼
RtpGrammarMatcher::acceptToken / rollback
    │
    └─ 返回 cap + masks，恢复 matcher
```

### 12.0 固定一个例子：后面一直追踪它

为避免每小节换符号，下面假设 `P=3`，并把若干字符串片段简化成单个 tokenizer token。Schema 只允许一个字段：

```json
{"answer": "..."}
```

请求已经 committed 的输出前缀是：

```text
{"answer"
```

因此 committed matcher `S0` 当前期待冒号。draft model 给出：

```text
d1 = :
d2 = "ok"
d3 = ,       # 对这个 schema 非法；值结束后应该是 }
```

第 11 章 target verify 同时已经产出 `L0..L3`。第 12 章要回答：每一行分别应该配哪个 grammar state？

```text
offset 的精确定义：已经 provisional accept 了多少个 draft token

offset=0
  parser state = S0 = after '{"answer"'
  next token mask M0 应允许 ':'
  M0 用来约束 target row L0，也用来检查 d1

offset=1
  parser state = S1 = after '{"answer":'
  next token mask M1 应允许 JSON value，例如 '"ok"'
  M1 用来约束 target row L1，也用来检查 d2

offset=2
  parser state = S2 = after '{"answer":"ok"'
  next token mask M2 应允许 '}'，不允许 ','
  M2 用来约束 target row L2，也用来检查 d3

offset=3
  只有 d1,d2,d3 全部合法时才到这里
  M3 用来约束 bonus row L3
```

把 proposal、target row、parser state 和 mask 横向对齐：

```text
offset       0                  1                    2                 3
─────────────┬──────────────────┬────────────────────┬─────────────────┬────────
prefix       committed          + d1                 + d2              + d3
state        S0                 S1                   S2                S3
mask         M0                 M1                   M2                M3
target row   L0                 L1                   L2                L3
draft check  d1=':'             d2='"ok"'            d3=','             bonus
result       allow              allow                deny
                                                      │
                                                      ▼
                                                    cap=2
```

这一张对齐图是理解 12～15 章的钥匙：`Mi` 永远描述“吃掉前 i 个 draft 后，下一个 token 的合法集合”；它既检查 `d(i+1)`，也 mask `Li` 的 fallback。

### 12.1 `MtpExecutor::runSpecLogitsVerify`

```cpp
LaunchTask task;
task.total_streams = streams.size();
task.propose_step  = propose_step_;
task.vocab_size    = vocab_size_;
task.draft_tokens  = draft_tokens;

for (const auto& stream : streams) {
    for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
        if (processor->mtpCapability().mode == MtpProcessorMode::SPEC_VERIFY) {
            task.active.push_back({processor, stream_idx});
        }
    }
}
return spec_logits_verify_runner_.run(task);
```

这里只收集显式支持 speculative verify 的 processor。

这一层的职责是把 engine 对象转换成 runner 能理解的扁平任务：

```text
engine 世界                              runner 世界

streams[0..B)                            LaunchTask
  ├─ stream id             ───────────>  total_streams=B
  ├─ processors            ───────────>  active[{processor,stream_idx}]
  └─ draft proposal        ───────────>  draft_tokens
executor propose_step=P    ───────────>  propose_step=P
model vocab_size=V         ───────────>  vocab_size=V
```

没有 `SPEC_VERIFY` capability 的 processor 不会被偷偷当成 grammar 执行；它要么仍走普通 processor 路径，要么由 admission rule 拒绝不支持的组合。

### 12.2 `SpecLogitsVerifyRunner::run`

源码：[SpecLogitsVerifyRunner.cc](../../rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.cc)

```cpp
ensureBuffersFit(shape);
std::fill_n(spec_cap_cpu_.data_ptr<int32_t>(), B, P);

materializeDraftTokensToCpu(task);
initializeCompactRows(layout, shape);
auto merge_result = mergeProcessorMasks(task, layout, shape);
auto result = makeResult(shape);
```

Runner 做四件事：

```text
GPU draft tokens
      │ D2H
      ▼
CPU draft_tokens [B,P]
      │
      ▼
每个 processor 构造 [P+1,W] packed masks + cap
      │ 多 processor 时按位 AND
      ▼
compact masks [active_streams*(P+1),W]
      │ H2D
      ▼
GPU mask kernel 只处理 active rows
```

这四步的产物是逐步累积的，不是四个互不相关的 helper：

```text
task.draft_tokens (GPU)
       │ materializeDraftTokensToCpu
       ▼
draft_tokens_cpu_ [B,P]
       │ initializeCompactRows
       ├──────────────────────────> row_indices [active rows]
       │
       │ mergeProcessorMasks
       ▼
merged_bitmask_cpu_ + spec_cap_cpu_
       │ makeResult / H2D
       ▼
LaunchResult
├── packed_allow_mask_gpu
├── logits_row_indices_gpu
├── spec_cap_cpu/gpu
└── processor_errors
```

为什么必须先把 draft token D2H：

```text
draft sampler 输出                         XGrammar matcher
CUDA tensor [B,P]                          CPU parser object
       │                                      ▲
       └────── D2H 到 pinned CPU buffer ──────┘
```

XGrammar 的 `AcceptToken()` 是 CPU 状态机调用，它不能直接解引用 CUDA tensor。target logits 不需要 D2H，因为它们只会在第 13 章通过 GPU mask kernel 处理；这里搬回 CPU 的只有小得多的 `[B,P]` token id。

尺寸直觉：

```text
必须 D2H：draft ids       B * P * sizeof(int32)
不应 D2H：target logits   B * (P+1) * V * sizeof(dtype)
```

### 12.3 `SpecLogitsVerifyRunner::mergeProcessorMasks`

```cpp
request.draft_tokens = draft_tokens_cpu_ + stream_idx * P;
request.propose_step = P;
request.bitmask_cpu_out = proc_mask.data_ptr<int32_t>();

auto cap_or = processor->prepareSpeculative(request);
cap = clamp(cap_or.value(), 0, P);

bitwiseAndBitmaskInplace(merged_row,
                         proc_mask.data_ptr<int32_t>(),
                         words_per_stream);
spec_cap_cpu_[stream_idx] = min(spec_cap_cpu_[stream_idx], cap);
```

如果未来同一 stream 有多个 MTP-aware processor：

```text
Grammar mask:       111100001111
Other hard mask:    110011001100
                    ──────────── AND
Merged mask:        110000001100

cap = min(grammar_cap, other_cap)
```

这里 `mask AND` 和 `cap min` 是同一件事的两种表示：

```text
逐 token 集合限制                     整条 draft 前缀限制

M_final[i] = M_A[i] & M_B[i]          cap_final = min(cap_A, cap_B)
     │                                      │
     └─ 保证 row i 采样合法                 └─ 保证非法 draft 不被保留
```

mask 交给第 13 章约束 target sampling，cap 交给第 14 章裁剪 speculative acceptance。两者缺一不可。

### 12.4 `GrammarLogitsProcessor::prepareSpeculative`

```cpp
std::lock_guard<std::mutex> lock(state_mutex_);
auto cap_or = verifySpecDraftAndFillBitmask(
    *matcher_, eos_token_id_, request);
return cap_or.value();
```

最重要的语义：它是“试走 + 回滚”，不是 commit。

从对象所有权看，这一层正好是 runner 的无状态 batch 逻辑与每个请求的有状态 matcher 之间的边界：

```text
SpecLogitsVerifyRunner                 GrammarLogitsProcessor
共享 batch buffers                    请求私有状态
无 grammar 语义                       matcher + committed length
         │                                      │
         └──── SpecLogitsProcessorRequest ─────>│
                                                │ mutex
         <──────── masks + cap/error ───────────┘
```

### 12.5 `verifyDraftPrefixAndFillBitmask`

```cpp
for (int offset = 0; offset <= P; ++offset) {
    row[offset] = fillSpecVerifyRow(matcher); // 当前状态的 next-token mask
    if (offset == P) return P;

    int32_t draft = request.draft_tokens[offset];
    if (!rowAllows(draft)) return offset;

    if (!matcher.acceptToken(draft)) return offset;
    provisional.recordAccepted();
}
```

循环内部的顺序一定是 **先 fill 当前状态的 mask，再检查并 provisional accept 当前 draft**：

```text
正确：

state Si ──fill──> Mi ──检查 d(i+1)──> accept ──> S(i+1)
          │
          └──────── Mi 同时给 Li 的 fallback 使用

错误：

state Si ──先 accept d(i+1)──> S(i+1) ──fill──> M(i+1)
                                                │
                                                └─ 拿它 mask Li，错位一格
```

为什么 `Li` 必须使用接受 draft 之前的 `Mi`？因为 `Li` 的职责就是在 `d(i+1)` 被拒绝时，从同一个位置生成 replacement。replacement 必须从 `Si` 出发，而不是从假设已经吃掉非法 draft 的 `S(i+1)` 出发。

套入 12.0 的固定例子，循环像一次带撤销日志的单步执行：

```text
进入：真实 matcher = S0，provisional_count=0

offset=0
  ① FillNextTokenBitmask(S0) -> M0，允许 ':'
  ② 检查 d1=':'：M0 bit=1
  ③ matcher.acceptToken(':') -> S1
  ④ provisional_count=1

offset=1
  ① FillNextTokenBitmask(S1) -> M1，允许 JSON string
  ② 检查 d2='"ok"'：M1 bit=1
  ③ matcher.acceptToken('"ok"') -> S2
  ④ provisional_count=2

offset=2
  ① FillNextTokenBitmask(S2) -> M2，允许 '}'，禁止 ','
  ② 检查 d3=','：M2 bit=0
  ③ 不调用 acceptToken(d3)
  ④ 返回 cap=2

离开前 guard rollback(provisional_count=2)：S2 -> S0
```

这里 `cap=2` 同时有三个等价含义：

```text
cap = 2
├── 合法 draft 前缀长度是 2：可以保留 d1,d2
├── 第一个非法 draft 的下标是 2：d3 非法
└── replacement 应从 target row 2 取：sample(L2 & M2)
```

边界值也按同一规则解释：

```text
cap=0   d1 就非法；输出最多是 target row0 的 1 个 replacement
cap=k   d1..dk 合法；第 k 行 target 可作为 replacement
cap=P   所有 P 个 draft 合法；第 P 行 target 可作为 bonus
```

以 `P=3`、`d1,d2` 合法、`d3` 非法为例：

```text
committed matcher = S0

offset 0: build M0 at S0; d1 allowed; provisional accept -> S1
offset 1: build M1 at S1; d2 allowed; provisional accept -> S2
offset 2: build M2 at S2; d3 denied;  cap = 2
offset 3: 不需要继续 provisional accept

rollback 2 tokens: S2 -> S0

result:
  masks = M0,M1,M2,(初始化 allow-all/合并后的安全行)
  cap   = 2
  committed matcher 仍为 S0
```

为什么 early stop 后未访问的后续 mask row 不会污染结果：`cap` 会在第 14 章把最终长度限制为 `cap+1`，因此 `M(cap+1)..MP` 对应的 target rows 不可能进入输出。runner 仍会把 buffer 初始化成安全值，以便 batch merge 和 GPU tensor shape 保持固定。

为什么必须 rollback：

```text
draft proposal != 最终接受结果

grammar 预演 d1,d2,d3
             │
             ├─ probability rejection 可能只接受 d1
             └─ grammar 本身也可能在 d3 拒绝

所以 verify 阶段绝不能永久推进 matcher。
只有 specUpdate(actual accepted tokens) 才能 commit。
```

把 provisional verify 类比成数据库事务：

```text
BEGIN       保存进入时 matcher token count
READ        fill M0
WRITE?      provisional accept d1
READ        fill M1
WRITE?      provisional accept d2
DECIDE      得到 masks + cap
ROLLBACK    撤销所有 provisional accepts

真正 COMMIT 不在这里，而在第 15 章 specUpdate(actual tokens)
```

### 12.6 `RtpGrammarMatcher::acceptToken/rollback`

```cpp
bool ok = matcher_->AcceptToken(token_id);
if (ok) ++num_accepted_;

matcher_->Rollback(n);
num_accepted_ = std::max<int64_t>(0, num_accepted_ - n);
```

这层 wrapper 把 XGrammar exception 转成 `ErrorInfo`，并维护 RTP 侧 token 计数。

把第 12 章的调用和返回合在一张栈图里：

```text
向下调用                                      向上返回
─────────────────────────────────────────────────────────────────
runSpecLogitsVerify(task)
  └─ runner.run
      └─ mergeProcessorMasks
          └─ prepareSpeculative
              └─ verify draft prefix
                  ├─ fill M0 at S
                  ├─ accept d1 -> S1
                  ├─ fill M1 at S1
                  ├─ accept d2 -> S2
                  └─ rollback(2) -> S
              <──── masks M0.. + cap
          <──────── processor mask/cap
      <──────────── merged masks/min cap
  <──────────────── LaunchResult

永久 matcher：S ───────────────────────────────────────────────> S
```

**交给下一章的接力棒：**

```text
SpecLogitsVerifyRunner::LaunchResult
├── packed_allow_mask_gpu   ──> 第 13 章，修改 target logits
├── logits_row_indices_gpu  ──> 第 13 章，mask row 对齐 logits row
├── spec_cap                ──> 第 14 章，限制最终 accept_len
└── processor_errors        ──> 第 15/21 章，随 stream update 传播
```

## 13. 把 grammar mask 应用到 target logits

第 13 章是第 11、12 两条支路的第一次汇合：

```text
第 11 章 target model                   第 12 章 XGrammar runner
logits [L0,L1,...,LP]                   masks [M0,M1,...,MP]
          │                                      │
          └──────────────────┬───────────────────┘
                             ▼
                    gatherSpecSamplerInput
                             │
             constrained_logits[i] = Li & Mi
                             │
                             ▼
                       Sampler::forward
                             │
                             ▼
            target sampled tokens/probabilities
```

这里的 `&` 是语义表示；实际实现是把 `Mi=0` 的 `Li` 写成 `-inf`。

继续第 12.0 节的固定例子。XGrammar 已得到：

```text
M0：当前位置只允许 ':' 等合法分隔 token
M1：当前位置允许 schema 对应的 JSON value
M2：值结束后只允许 '}'，不允许 draft d3=','
cap=2
M3：因为 offset 2 已停止，后续行不会进入最终输出
```

target model 的 raw logits 可能仍然更偏爱非法 token：

```text
row L0：最高候选 '='，次高 ':'
row L1：最高候选 '"ok"'
row L2：最高候选 ','，次高 '}'
row L3：若干 bonus 候选
```

第 13 章做完后的效果：

```text
row 0        L0 + M0        '=' -> -inf        sample ':'
row 1        L1 + M1        合法 value 保留    sample '"ok"'
row 2        L2 + M2        ',' -> -inf        sample '}'
row 3        后续 cap 会截掉，不参与最终输出
```

这里可以提前看到第 12 章为什么必须生成 `M2`：`d3=','` 不合法时，第 14 章会用 row 2 的 target sample 替换它；若 row 2 没有被 `M2` 约束，replacement 仍可能采回同一个非法逗号。

### 13.1 compact row 映射

假设 batch 中 3 个 stream，只有 stream 0 和 2 有 MTP-aware processor，`P=2`：

```text
完整 target logits rows（每 stream P+1=3 行）

stream 0: row 0,1,2   <- active
stream 1: row 3,4,5   <- no processor
stream 2: row 6,7,8   <- active

compact masks:
mask row 0 -> logits row 0
mask row 1 -> logits row 1
mask row 2 -> logits row 2
mask row 3 -> logits row 6
mask row 4 -> logits row 7
mask row 5 -> logits row 8

row_indices = [0,1,2,6,7,8]
```

`row_indices` 是把“只为 active stream 生成的紧凑 mask”重新接回“包含所有 stream 的完整 target logits”的桥：

```text
compact mask row k
       │ row_indices[k]
       ▼
full logits row r

没有它：mask row 3 会误打到 stream 1 的 logits row 3
有了它：mask row 3 正确打到 stream 2 的 logits row 6
```

### 13.2 `MtpBatchStreamProcessor::gatherSpecSamplerInput`

```cpp
sampler_inputs.logits = model_output.logits.clone();

if (spec_logits_result.has_active_processor) {
    SpecLogitsVerifyRunner::applyMaskToLogits(
        sampler_inputs.logits,
        spec_logits_result.packed_allow_mask_gpu,
        spec_logits_result.logits_row_indices_gpu,
        sampler_inputs.vocab_size);
}
```

注意这里不再走 normal `LogitsProcessorStates::batchProcess()`：MTP 的 `P+1` 行状态不同，必须使用前面基于 provisional matcher 生成的逐位置 mask。

与第 8 章 normal sampling 对比：

```text
normal 首 token                          MTP verify
────────────────────────────────────────────────────────────────
一条 logits row                         每 stream 有 P+1 rows
matcher 当前状态生成一个 mask           provisional 状态链生成 P+1 masks
processor::process 当场造 mask           第 12 章预先造好 artifact
Sampler preprocess 中应用               gatherSpecSamplerInput 中应用
```

区别在“mask 怎样准备”，不在 correctness：二者最终都必须让 Sampler 只看到被 hard-mask 后的 logits。

此处 clone 也形成清晰的所有权边界：

```text
model_output.logits                  sampler_inputs.logits
target verify 原始输出   ──clone──>  可被 processor/mask 原地修改的工作副本
       │                                     │
       │ 保留给其他后处理/调试               └─> Sampler::forward
       └────────────────────────────────────────────────────────
```

### 13.3 GPU mask 后的采样

```text
target logits [B*(P+1), V]
             │
             ▼ packed mask kernel
grammar-illegal logits = -inf
             │
             ▼ Sampler::forward
target sampled tokens [B*(P+1)]
target probabilities   [B,P+1,V]
```

以 `P=3` 的单 stream 为例，位置对齐关系贯穿始终：

```text
position       draft 候选      target logits/mask      target sample
────────────────────────────────────────────────────────────────────
0              d1              L0 & M0                 t1
1              d2              L1 & M1                 t2
2              d3              L2 & M2                 t3
3              bonus           L3 & M3                 t4
```

注意 `t1..t4` 还不是马上全部输出。它们是 target 为每个可能拒绝位置准备的合法 replacement/bonus，下一章才决定实际用到哪一行。

固定例子此刻的数据台账：

```text
draft proposal         [ ':', '"ok"', ',' ]
grammar cap            2
target sampled rows    [ ':', '"ok"', '}', unused ]
target probabilities   [ Q0,   Q1,     Q2,  Q3     ]
matcher                仍为 S0
```

注意 target row 0/1 的 sampled token 即使恰好等于 draft d1/d2，也不是“已经接受 draft”的意思；是否接受由 target/draft 概率比和随机数在第 14 章决定。

```text
第 13 章输出
├── target_token_probs [B,P+1,V]
├── target_sampled_tokens [B,P+1]
└── grammar cap（原样随 artifact 继续向后）
                         │
                         ▼
             第 14 章 probability rejection
```

## 14. Probability rejection 与 grammar cap 如何共同决定输出

第 14 章同时拿到三张“票”：

```text
draft model 的票                     target model 的票             grammar 的票
P_draft(di)                          P_target(di) / sampled ti      cap + Mi
      │                                      │                         │
      └──────────────────────┬───────────────┴─────────────────────────┘
                             ▼
                    SpeculativeSampler
                             │
                概率上能接受多少 draft？
                             ▼
                    applySpecVerifyResult
                             │
                语法上最多允许多少 draft？
                             ▼
              actual accepted tokens [a1..aN]
```

概率 rejection 和 grammar cap 不是先后重复检查同一个条件：前者维护 speculative sampling 的分布正确性，后者维护 structural correctness。

先把三类数据的职责钉死：

```text
draft token/probability      决定 proposal 是什么，以及 proposal 分布 q
target probability          决定 target 分布 p 下是否接受 proposal
target sampled row          proposal 被拒时，提供 replacement
grammar mask                保证每个 target sampled row 合法
grammar cap                 保证非法 draft 本身不会被保留
```

### 14.1 `SpeculativeSampler::forward/batchSample`

源码：[SpeculativeSampler.cc](../../rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.cc)

```cpp
execChainSpeculativeSampling({
    draft_token_probs,
    draft_token_ids,
    uniform_samples,
    target_token_probs,
    output_token_ids,
    output_accepted_token_num,
    output_emitted_token_num});

accept_len = output_emitted_token_num[stream_idx];
accept_tokens = output_token_ids[stream_idx][0:accept_len];

// 最后一位总使用 target token
accept_tokens[accept_len - 1] = target_sampled_token;
```

概率 rejection 回答的是：“draft token 在 target 分布下能接受多少个？”

XGrammar cap 回答的是：“从语法看，draft 前缀最多能走多远？”

两者取更严格结果。

标准 speculative sampler 暂时不知道 grammar cap，它先给出概率结果：

```text
draft             [d1, d2, d3]
target verdict     A   A   R
target fallback                t3
                            │
                            ▼
probability output [d1,d2,t3], old_len=3
```

这份 output 中的每个 target fallback 已经在第 13 章被对应的 `Mi` mask 过，因此 fallback 本身合法；但被保留的 draft 前缀还要受 cap 限制。

### 14.2 `applySpecVerifyResult`

```cpp
const int token_cap = clamp(spec_cap[i], 0, P);
const int old_len = output.accept_len[i];

if (token_cap < P && old_len > token_cap) {
    output.accept_tokens[i][token_cap] =
        target_token_at_row(token_cap);
}

const int new_len = std::min(old_len, token_cap + 1);
output.accept_len[i] = new_len;
output.accept_tokens[i] = narrow(0, new_len);
```

为什么是 `cap + 1`：

```text
cap = 合法 draft token 数

cap=0: draft d1 不合法
       仍可输出 grammar-masked target token t1
       => emitted len = 1

cap=2: d1,d2 合法，d3 不合法
       可输出 [d1,d2,target fallback t3]
       => emitted len = 3
```

例子，`P=3`：

```text
draft:                    [ d1, d2, d3 ]
grammar cap:                       2
probability result(old):  [ d1, d2, d3, t4 ]  len=4
                                      ▲
                                      d3 grammar invalid

修正后:                  [ d1, d2, t3 ]      len=3
                                      ▲
                         row 2 在 M2 下采出的合法 target token
```

把两个裁决画成坐标会更直观：

```text
                    grammar 最多保留的 draft 数 cap
                    0          1          2          3
概率结果 old_len
1                   1          1          1          1
2                   1          2          2          2
3                   1          2          3          3
4                   1          2          3          4

单元格 = final_len = min(old_len, cap+1)
```

`+1` 的那个 token永远来自 target row `cap`：

```text
合法 draft prefix                  replacement/bonus
[d1 ... d_cap]                     target_sample[row cap]
        │                                   │
        └────────────────┬──────────────────┘
                         ▼
                 accept_tokens[0:final_len]
```

**交给下一章的接力棒已经不再是 proposal，而是唯一真实结果：**

```text
SpeculativeSamplerOutput
├── accept_tokens = [a1..aN]
├── accept_len    = N
└── processor_errors
                         │
                         ▼
                 第 15 章永久 commit
```

继续固定例子，假设概率 sampler 原本愿意接受全部三个 draft，并给出一个 bonus：

```text
probability output before cap
    [ ':', '"ok"', ',', t4 ]
      d1     d2    d3  bonus
    old_len = 4

grammar result
    cap = 2
    d1,d2 合法；d3 非法
    target_sample[row2] = '}'
```

`applySpecVerifyResult()` 做两步：

```text
第一步 replacement
    position cap=2：',' -> target_sample[row2]='}'

第二步 truncate
    new_len=min(4,2+1)=3

最终 actual accepted tokens
    [ ':', '"ok"', '}' ]
```

完整变形图：

```text
draft/prob result       :   | : | "ok" | , | t4 |
grammar valid prefix    :   |======= 2 =======|
target fallback at cap  :                  | } |
                                        replacement
final                   :   | : | "ok" | } |
```

再看概率更严格的情况。若 probability rejection 在 `d2` 处就停止：

```text
probability output = [d1, target_sample[row1]], old_len=2
grammar cap        = 2，最多允许 3 个输出 token

final_len = min(2,3) = 2
```

此时 grammar cap 不会强行多输出 token；它只提供上界，不提供下界。

## 15. 最终 accepted tokens 如何真正推进 matcher

前四章都可以丢弃或回滚，只有第 15 章会改变请求的 authoritative state：

```text
临时世界（11～14）                         永久世界（15）

draft proposals                            CompleteTokenIds
target logits                              output queue
provisional matcher states                 think_info_
mask/cap                                   grammar matcher
probability result                              ▲
        │                                       │
        └──── actual [a1..aN] ──────────────────┘
```

完整 commit 调用栈：

```text
MtpExecutor::decodeStep
  └─ MtpBatchStreamProcessor::dispatchDecode
      ├─ prepareDecodeSpecUpdateInfo
      │    └─ 把 accept tensor 切成每个 stream 的 [a1..aN]
      ├─ updateProposeTokens
      │    └─ 保存下一轮 d1_next
      └─ StreamGroups::updateStreams
          └─ GenerateStream::specUpdate
              ├─ CompleteTokenIds::update
              ├─ updateLogitProcessorStatus
              │    └─ GrammarLogitsProcessor::updateStatus
              │         └─ acceptCommittedLocked
              ├─ 更新 SPOutputBuffer
              └─ updateOutput / finish / enqueue output
```

继续固定例子。第 14 章的唯一真实结果是：

```text
actual = [ ':', '"ok"', '}' ]
```

第 12 章虽然曾让 matcher 临时走过 `:` 和 `"ok"`，但已经 rollback；所以第 15 章必须从原 committed `S0` 重新接受 actual：

```text
进入 commit
history = '{"answer"'
matcher = S0

accept ':'      S0 -> S1
accept '"ok"'   S1 -> S2
accept '}'      S2 -> TERMINATED

离开 commit
history = '{"answer":"ok"}'
matcher = TERMINATED
committed_output_len += 3
```

这里实际提交的第三个 token 是 target replacement `}`，不是 draft d3=`,`。这正说明不能把第 12 章 provisional draft 状态直接当成最终状态：最终 token 序列可能在任意 rejection/cap 位置分叉。

### 15.1 `MtpBatchStreamProcessor::prepareDecodeSpecUpdateInfo`

```cpp
StreamSpecUpdateInfo info{
    accept_tokens[batch_idx],
    accept_len[batch_idx],
    -1,
    draft_hidden_states,
    propose_all_probs};

info.error_info = spec_decode_output.processor_errors[stream_idx];
```

它把 batch 坐标重新还原成请求坐标：

```text
batch tensors                         per-stream update

accept_len[B]       ─┐
accept_tokens[B,P+1] ├─ slice stream i ─> StreamSpecUpdateInfo i
processor_errors[B] ─┘                    ├─ new_tokens=[a1..aN]
                                          ├─ num_new_tokens=N
                                          └─ error_info
```

### 15.2 `dispatchDecode -> StreamGroups::updateStreams`

```cpp
prepareDecodeSpecUpdateInfo(..., spec_update_infos);
updateProposeTokens(..., spec_update_infos);
stream_groups.updateStreams(spec_update_infos);
```

到这里 engine batch 开始拆回独立请求；从此每个 stream 分别 commit，不再共享 batch 行号：

```text
spec_update_infos[0] ──> stream 0::specUpdate
spec_update_infos[1] ──> stream 1::specUpdate
spec_update_infos[2] ──> stream 2::specUpdate
```

### 15.3 `GenerateStream::specUpdate`

```cpp
complete_token_ids_->update(new_tokens, ..., num_new_tokens, ...);

int accept_token_num = nxt_cached_len - cur_cached_len;
updateLogitProcessorStatus(new_tokens, accept_token_num);

sp_output_buffer_->tokens[0] = target_last_token;
sp_output_buffer_->tokens[1] = next_draft_token;

updateOutput(...);
```

顺序形成一条不能跳跃的状态链：

```text
[a1..aN]
    │
    ▼
token history append
    │ append 成功多少，才知道实际 committed 数
    ▼
processor updateStatus(actual committed tokens)
    │ grammar/think state 与 history 对齐
    ▼
SP buffer 写入 [本轮末 token, 下一轮 d1]
    │
    ▼
output/finish 对外可见
```

也就是说，下一轮 draft 起点和客户端输出都必须建立在同一批已提交 token 上。

### 15.4 `GrammarLogitsProcessor::updateStatus`

```cpp
std::lock_guard<std::mutex> lock(state_mutex_);
return acceptCommittedLocked(new_tokens.data_ptr<int32_t>(),
                             num_new_tokens);
```

### 15.5 `acceptCommittedLocked`

```cpp
const int64_t old_matcher_len = matcher_->numAcceptedTokens();
const int64_t old_output_len  = committed_output_len_;

for (size_t i = 0; i < n; ++i) {
    if (matcher_->isTerminated()) {
        require(tokens[i] == eos_token_id_);
        break;
    }
    require(matcher_->acceptToken(tokens[i]));
}

committed_output_len_ = old_output_len + n;
decode_mask_builder_->refreshAfterCommit(
    *matcher_, committed_output_len_);
```

失败时函数回滚整批，而不是留下半批状态：

```text
before commit: S0, committed_len=N
try tokens:    a1 -> S1, a2 -> S2, a3 rejected
rollback:      S2 -> S0
restore len:   N
restore mask:  mask(S0,N)
return error
```

这里的 rollback 和第 12 章 rollback 目的不同：

```text
第 12 章 rollback                       第 15 章失败 rollback
─────────────────────────────────────────────────────────────────
正常流程必定发生                        只有 commit 出错时发生
撤销 proposal 预演                      撤销半批 actual commit
返回 masks/cap                          返回 ErrorInfo
matcher 必须回到进入时状态              matcher/len/mask 原子恢复
```

最终状态不变量：

```text
GenerateStream::outputTokenLen()
             ==
GrammarLogitsProcessor::committedOutputLen()
```

这也是最终 token 即使结束 stream，也仍要调用 `updateStatus()` 的原因。

固定例子中 `}` 很可能让 stream 立刻结束，但它仍必须先完成：

```text
append '}'
   │
   ▼
matcher.acceptToken('}') -> TERMINATED
   │
   ▼
committed length 对齐
   │
   ▼
needFinish / publish final output
```

若因为“反正 stream 已结束”跳过最后一个 token，客户端文本虽然看似正确，内部却会变成：

```text
stream history：... '}'
matcher state ：仍停在等待 '}' 的 S2
```

这就是前文讨论“看起来只差一个 token，能不能丢”的核心答案：对输出字符串可能暂时不可见，对 stateful processor invariant 则是确定性的破坏。

一轮 MTP 到这里才真正闭环，并把状态接给下一轮：

```text
round N 进入时                         round N commit 后

history = H                            history = H + [a1..aN]
matcher = S                            matcher = S'
SP = [Tprev,d1]                        SP = [aN,d1_next]
KV/seq position = K                    KV/seq position = K'
                                              │
                                              ▼
                                      round N+1 第 11.2 节
```

循环图：

```text
      ┌─────────────────────────────────────────────────────────┐
      │                                                         │
      ▼                                                         │
[Tprev,d1] -> propose/verify -> mask/sample -> reject/cap -> commit
                                                           │
                                                           ▼
                                                   [aN,d1_next]
      ▲                                                         │
      └─────────────────────────────────────────────────────────┘
```

## 16. Think boundary 如何在一轮 MTP 内跨过去

第 16 章不是插在第 15 章之后执行的新函数。它是在同一条 12→13→14→15 链里，把普通 `GrammarLogitsProcessor` 换成带 think 状态的 processor 后，观察一次 proposal 跨状态边界时发生什么：

```text
同一轮控制流

第 12 章 provisional verify
  Reasoning/Grammar processor 的临时状态副本
        │ 可能跨 THINK → CLOSING → FINAL_GRAMMAR
        ▼
第 13/14 章 mask + reject + cap
        │ 只留下 actual accepted tokens
        ▼
第 15 章 updateStatus
  永久 think_info_ / matcher 按 actual tokens 重放同一路径
```

Reasoning + grammar processor 逻辑上拼接了两个自动机：

```text
Reasoning state machine                         Grammar state machine

IN_THINK -> CLOSING_THINK -> AFTER_THINK  ───>  JSON S0 -> S1 -> ...
     │             │                    边界          │
任意 reasoning     强制 end-think 剩余 token         structural masks
```

每个 offset 只由其中一侧产生 mask：

```text
当前 phase=IN_THINK/CLOSING_THINK   -> think rules 生成 Mi
当前 phase=AFTER_THINK              -> XGrammar matcher 生成 Mi
```

假设 grammar 当前仍在 THINK，剩余 budget 只够 1 token，`P=3`：

```text
draft = [reasoning_token, think_end, '{']

row 0 / M0: THINK_ANY_TEXT
             reasoning_token allowed
             provisional accept

row 1 / M1: THINK budget exhausted / 等待 end
             think_end allowed
             provisional accept

row 2 / M2: FINAL_JSON_SCHEMA start
             '{' allowed
             provisional accept

row 3 / M3: JSON object internal state
             只允许合法 key/value/'}' token
```

逐 offset 台账：

```text
offset  临时 phase/state        mask 来源         检查 draft          预演后
────────────────────────────────────────────────────────────────────────────
0       IN_THINK                think rule        reasoning_token ✓   budget-1
1       EXPECT/CLOSING_END      think rule        think_end ✓         AFTER_THINK
2       AFTER_THINK + grammar S0 XGrammar          '{' ✓               grammar S1
3       AFTER_THINK + grammar S1 XGrammar          bonus row           不再吃 draft
```

这里 `M1` 与 `M2` 可能来自完全不同的状态机，但对第 13 章来说它们仍是统一布局的相邻两行 mask；GPU sampler 不需要知道某一行来自 think rule 还是 XGrammar。

所以 MTP 并不需要一个单独的“think 阶段 special case”。逐位置 provisional matcher 自然处理跨状态边界：

```text
THINK ──d1──> THINK_END ──d2──> JSON ──d3──> JSON_NEXT
  │             │                 │             │
 M0            M1                M2            M3
```

同一批 token 在 verify 与 commit 中会走两遍，但性质不同：

```text
第 12 章：预演轨                              第 15 章：提交轨

copy(think_state), matcher=S                   real think_info_, matcher=S
          │                                             │
try d1 -> THINK_END                            commit actual a1
try d2 -> JSON                                 commit actual a2
try d3 -> JSON_NEXT                            commit actual a3
          │                                             │
产生 masks/cap                                  永久状态 -> S'
          │
rollback/copy 丢弃 -> S
```

只有当 probability rejection 和 grammar cap 最终保留了 `[d1,d2,d3]`，提交轨才会真的走完三步；若最终只输出 `[d1,target_fallback]`，提交轨只接受这两个真实 token。

把跨 boundary 的整轮再串一次：

```text
第 12 章
  copy think state / provisional grammar
  [M_think, M_force_end, M_json_start, M_json_next] + cap
                              │
                              ▼
第 13 章
  四行 target logits 分别套对应 mask，准备合法 fallback
                              │
                              ▼
第 14 章
  probability result 与 cap 合并，只留下 actual tokens
                              │
                              ▼
第 15 章
  real think_info_ 接受 actual think/end token
  real matcher 只接受 actual boundary 之后的 grammar token
```

如果 draft 在边界处给错 token：

```text
THINK ──合法 reasoning──> EXPECT_END ──错误普通文本──X
                                              │
                                              cap=1
                                              │
                                              ▼
                              使用 target row1 的合法 think_end 替换
```

到第 16 章为止，讲的是同步语义上的完整 correctness 链。第 17、18 章不会改变这些数据依赖，只会把可以重叠的工作移到 worker/CUDA auxiliary stream：

```text
同步逻辑依赖：proposal -> masks -> sample -> accepted -> commit
                         │
                         ▼
异步实现目标：保持箭头不变，把独立方框横向重叠
```

## 17. `dsv4_on_dev` 已实现的 MTP + XGrammar 异步链

这一节不再描述建议方案，而是读取本地分支：

```text
branch : feat/dsv4_on_dev
commit : dd20403c3466bb510e7ae60b9d6935dc527c0247
```

不切分支也可以复现文中的源码：

```bash
git show feat/dsv4_on_dev:rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc
git show feat/dsv4_on_dev:rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.cc
git show feat/dsv4_on_dev:rtp_llm/cpp/models/logits_processor/ReasoningGrammarLogitsProcessor.cc
```

### 17.0 先把第 12～15 章映射到异步实现

异步版没有改变算法，只改变“谁在什么时候执行”。先做一张一一对应表：

```text
同步语义章节                    dsv4_on_dev 执行位置
──────────────────────────────────────────────────────────────────────
第 11 章 draft proposal         main engine thread + main CUDA stream
第 11 章 target verify          main thread enqueue 到 main CUDA stream
第 12 章 draft D2H              grammar worker + copy_stream_
第 12/16 章 matcher 预演         grammar worker CPU
第 12 章 mask/cap H2D           grammar worker 发起，copy_stream_ 执行
第 13 章 apply mask/sample       main CUDA stream
第 14 章 rejection/cap           main CUDA stream
第 15 章 accepted D2H/commit     bookkeeping worker + its CUDA stream
```

依赖箭头不能变：

```text
draft ready ───────> grammar verify ───────> mask ready ──┐
       │                                                   │
       └──────────> target verify/logits ready ────────────┤
                                                           ▼
                                                   mask + sample
                                                           │
                                                           ▼
                                                  reject + grammar cap
                                                           │
                                                           ▼
                                                    actual tokens
                                                           │
                                                           ▼
                                                        commit
```

异步优化做的是把 DAG 中没有直接依赖的节点横向展开：

```text
同步排队的直觉：

draft -> target verify -> grammar D2H/CPU/H2D -> sample -> commit

dsv4 实际调度：

draft ->┬-> target verify GPU ----------------┐
        └-> D2H -> XGrammar CPU -> mask H2D --┤
                                              ├-> sample/reject
                                              │
                                              └-> next draft work --┐
                                                   commit worker ----┴-> next round
```

继续沿用第 12.0 节的 `P=3` 例子，异步链传递的“接力棒”如下：

```text
Artifact A：draft_token_ids_gpu = [d1=':', d2='"ok"', d3=',']
     │ E_draft_ready
     ▼
Artifact B：mask/cap
     ├── mask rows：M0,M1,M2,...
     ├── cap=2
     └── E_mask_ready
     │
     ▼
Artifact C：constrained target sample
     └── row2 fallback='}'
     │
     ▼
Artifact D：actual tokens=[':','"ok"','}']
     ├── E_rejection
     └── accepted-token D2H
     │
     ▼
Artifact E：committed stream/matcher state
```

读 17.1～17.16 时，只需要追踪 A→B→C→D；第 18 章再追踪 D→E 以及 E 如何约束下一轮。

### 17.1 先分清四种“异步”

`dsv4_on_dev` 里至少有四条不同的异步支路，不能统称为一个 async：

```text
                         MtpExecutor::decodeStep
                                  │
          ┌───────────────────────┼────────────────────────┐
          │                       │                        │
          ▼                       ▼                        ▼
  A. attention prepare     B. XGrammar spec verify   C. stream bookkeeping
  两个 AsyncRunner         一个 AsyncRunner          一个 AsyncRunner
  + 独立 CUDA stream       + worker thread           + worker thread
          │                + 内部 copy stream        + 独立 CUDA stream
          │                       │                        │
          └───────────────┬───────┴──────────────┬─────────┘
                          │                      │
                          ▼                      ▼
                  D. metrics D2H          主线程继续下一轮
                  独立 CUDA stream        使用 GPU device state
```

三个环境开关控制的是不同层次：

```cpp
bool MtpExecutor::useStreamAsync() const {
    return readEnvFlagOnce("RTP_LLM_STREAM_ASYNC", ...);
}

bool MtpExecutor::useDropBroadSync() const {
    return readEnvFlagOnce("RTP_LLM_DROP_BROAD_SYNC", ...);
}

bool MtpExecutor::useAsyncPrepare() const {
    return readEnvFlagOnce("RTP_LLM_MTP_ASYNC_PREPARE", ...);
}
```

开关矩阵：

```text
RTP_LLM_MTP_ASYNC_PREPARE=1
  └─ target/draft attention input prepare 放到辅助 runner

RTP_LLM_STREAM_ASYNC=1
  └─ accepted token D2H、specUpdate、processor commit 放到 bookkeeping worker

RTP_LLM_DROP_BROAD_SYNC=1
  └─ 下一轮尽量不在入口等上一轮 bookkeeping；只在真正读 host/grammar state 前窄同步

有 SpecLogitsProcessor 且 P>1
  └─ spec_logits_verify_async_runner_ 执行 XGrammar verify
     这条分支本身不由 RTP_LLM_STREAM_ASYNC 开关决定
```

换句话说：

```text
“XGrammar verify 异步” ≠ “stream output commit 异步” ≠ “attention prepare 异步”
```

把四种异步放回刚才的 artifact 链：

```text
A draft ready
├─ attention prepare async ───────> 帮 target/draft forward 提前备 metadata
├─ grammar verify async ──────────> 生产 B mask/cap
└─ target verify on main stream ──> 生产 target logits
                                      │
                                      ▼
                              C/D sample/reject
                                      │
                                      ├─ bookkeeping async -> E commit
                                      └─ metrics async -> 仅观测，不参与正确性
```

其中只有 grammar verify、bookkeeping 两条会触碰 structural 状态链；attention prepare 和 metrics 不改变 matcher/token history。

### 17.2 `AsyncRunner` 是什么

四条支路的基础设施都是 `AsyncRunner`。核心代码非常短：

```cpp
void AsyncRunner::launch(std::function<void()> fn) {
    at::ThreadLocalState tls_state;
    {
        std::unique_lock<std::mutex> lk(mutex_);
        cv_done_.wait(lk, [this] { return task_done_; });
        rethrowPendingExceptionIfAny(lk);
        pending_task_ = Task{std::move(fn), std::move(tls_state)};
        task_done_ = false;
    }
    cv_task_.notify_one();
}

void AsyncRunner::sync(const torch::Stream& wait_stream) {
    std::unique_lock<std::mutex> lk(mutex_);
    cv_done_.wait(lk, [this] { return task_done_; });
    rethrowPendingExceptionIfAny(lk);
    lk.unlock();
    event_.block(wait_stream);
}
```

worker 线程：

```cpp
void AsyncRunner::workerLoop() {
    ...
    GraphStreamGuard stream_guard(toGraphStream(stream_));
    task.fn();
    event_.record(stream_);
    ...
}
```

它同时解决两类依赖：

```text
Host 依赖：
launch ──把 lambda 交给 worker──> task.fn()
sync   <──等待 task_done_─────────┘

CUDA 依赖：
worker CUDA stream ──record event_──┐
                                    ├─ wait_stream.wait(event_)
main CUDA stream  <─────────────────┘
```

注意它是 single-slot runner：

```text
task N  : [RUNNING----------------DONE]
task N+1:                         [launch 才能成功][RUNNING----]

同一个 AsyncRunner 不会并行执行两个 task。
```

这对 `SpecLogitsVerifyRunner` 的成员 scratch buffer 很重要：同一 executor 的两个 grammar task 不会同时覆盖它们。

把 `launch()`/`sync()` 放到一次调用里理解：

```text
main thread                      worker thread                    CUDA streams
────────────────────────────────────────────────────────────────────────────
runner.launch(fn)
   │ enqueue Task ─────────────> wake / take Task
   │ return                         │
   │ main 继续                      ├─ GraphStreamGuard(worker stream)
   │                               ├─ fn()
   │                               └─ record runner.event
   │
runner.sync(main_stream)
   ├─ host wait task_done <────── worker done
   └─ main_stream wait(event) ────────────────────────────────> CUDA happens-before
```

所以 `sync()` 同时补齐 host 对象依赖和 CUDA 数据依赖；只等其中一个都不够。

### 17.3 `MtpExecutor` 中有哪些 runner

构造函数直接创建四个辅助 stream/runner：

```cpp
MtpExecutor::MtpExecutor(...):
    collect_metrics_stream_(graphGetStreamFromPool(true)),
    target_verify_prepare_runner_(graphGetStreamFromPool(true)),
    draft_prefill_prepare_runner_(graphGetStreamFromPool(true)),
    spec_logits_verify_async_runner_(graphGetStreamFromPool(true)),
    spec_logits_verify_runner_(std::make_unique<SpecLogitsVerifyRunner>()),
    spec_bookkeeping_runner_(graphGetStreamFromPool(true)) {
    ...
}
```

对象关系图：

```text
MtpExecutor
├── main engine thread
├── current/default CUDA stream
├── target_verify_prepare_runner_
│   ├── worker thread
│   └── prepare CUDA stream
├── draft_prefill_prepare_runner_
│   ├── worker thread
│   └── prepare CUDA stream
├── spec_logits_verify_async_runner_
│   ├── worker thread
│   └── runner CUDA stream
│       └── SpecLogitsVerifyRunner
│           └── 自己还有一个 copy_stream_
├── spec_bookkeeping_runner_
│   ├── worker thread
│   └── bookkeeping CUDA stream
└── collect_metrics_stream_
```

这些 runner 不是层层互相调用，而是都由 `MtpExecutor::decodeStep()` 这个中心调度者 launch/join：

```text
                         decodeStep main thread
                                  │
             launch ┌─────────────┼──────────────┐ launch
                    ▼             ▼              ▼
             prepare runner  grammar runner  bookkeeping runner
                    ▲             ▲              ▲
              join  │       join before      join only where
              before forward     sampler      host/grammar state needed
                    └─────────────┴──────────────┘
```

因此阅读源码时不要沿某个 runner 寻找“下一个 runner”；真正的接力发生在 `decodeStep()` 保存的 event、tensor 和 `LaunchResult` 上。

### 17.4 一轮 decode 的总时间线

先看全图，再逐段展开。设 `P=3`：draft 提议 `d1,d2,d3`，target verify 产生 4 行 logits。

```text
time ───────────────────────────────────────────────────────────────────────────────>

Engine host:
  gather ─ draft propose ─ enqueue target verify ─ launch grammar worker ─ join ─ sample/reject ─ draft prefill ─ dispatch

Main CUDA stream:
  [gather/H2D]
       [draft d1][draft d2][draft d3] E_draft
                                         [--------- target verify 4 rows ---------]
                                                                                wait E_mask
                                                                                [mask][sample][reject]
                                                                                                     E_reject
                                                                                                     [draft prefill/sample] E_draft2
                                                                                                                               [publish GPU state]

Grammar worker thread + copy stream:
                                          wait E_draft
                                          [D2H d1..d3]--sync(copy only)
                                                        [CPU XGrammar try/rollback]
                                                                 [CPU expand mask]
                                                                 [H2D mask/cap] E_mask

Bookkeeping worker:
                                                                                                                               wait E_reject/E_draft2
                                                                                                                               [D2H ready]
                                                                                                                               [specUpdate]
                                                                                                                               [commit matcher]
```

关键 overlap 是：

```text
                         ┌──── target verify GPU kernels ────┐
                         │                                   │
draft ready ─────────────┼─ D2H ─ CPU XGrammar ─ H2D mask ──┤
                         │                                   │
                         └──────── max(两条支路) ────────────┘
```

采样是 join point，必须同时满足：

```text
target logits ready
        AND
grammar mask/cap ready
```

按 artifact 看，这张时间线只有五个关键里程碑：

```text
E_draft   ：A 已完成，grammar copy stream 可以读取 draft ids
E_mask    ：B 已在 GPU 可见，sampler 可以读取 mask/cap
E_reject  ：D 的 accept_len/accept_tokens 已由 GPU 产生
E_draft2  ：下一轮首个 draft/prob/hidden 已产生
commit done：E 已写入 host stream + matcher
```

事件不是“通知一下”的日志点，而是跨 stream 的内存可见性边界：

```text
producer 在 event 前写 tensor
consumer wait/block event
consumer 在 wait 后读 tensor
```

### 17.5 draft proposal 与 `E_draft_ready`

`decodeStep()` 先产生 proposal：

```cpp
if (propose_step_ > 1) {
    draftModelDecode(model_input,
                     stream_groups,
                     draft_probs_list,
                     draft_token_ids_t,
                     model_forward_us);
}

auto draft_tokens_ready_event =
    std::make_shared<torch::Event>(makeGraphEvent());
draft_tokens_ready_event->record(graphGetCurrentStream());
```

`draft_token_ids_t` 的布局不是只有 draft token；构建 target verify 输入时是：

```text
每个请求一行，P=3：

column       0       1       2       3
          ┌───────┬───────┬───────┬───────┐
tokens    │ t_prev│  d1   │  d2   │  d3   │
          └───────┴───────┴───────┴───────┘
```

`SpecLogitsVerifyRunner` 会检测列数大于 `P`，从 offset 1 开始取真正的 draft：

```cpp
const int64_t draft_cols   = task.draft_tokens.numel() / B;
const int64_t draft_offset = draft_cols > P ? 1 : 0;
auto draft = task.draft_tokens.reshape({B, draft_cols})
                              .narrow(1, draft_offset, P);
```

`E_draft_ready` 刚好记录在 draft tensor 最后一次 GPU 写入之后、target verify kernels 之前：

```text
main stream queue

[write d1][write d2][write d3] | record E_draft | [target K1][target K2]...
                                      │
                                      ├─ grammar 支路只需要等到这里
                                      └─ 不需要等 target verify 完成
```

这一步交出 Artifact A：

```text
Artifact A
├── tensor owner：draft_token_ids_t
├── logical slice：[d1..dP]，跳过 column 0 的 Tprev
└── readiness：E_draft_ready
        │
        ├─ target verify 已直接使用完整 [Tprev,d1..dP]
        └─ grammar worker 下一节 D2H 使用 [d1..dP]
```

### 17.6 为什么 worker 在 `forward()` 调用之后 launch，仍能与 target GPU 重叠

源码顺序是：

```cpp
draft_tokens_ready_event->record(current_stream);

model_output = runTargetVerifyForward(model_input, stream_groups);

spec_logits_verify_async_runner_.launch([...] {
    *spec_logits_result = buildSpecLogitsVerifyInline(...);
});
```

只看 C++ 行号容易误判成完全串行。PyTorch/CUDA `forward()` 通常是向 main CUDA stream 入队 kernel，host 返回时 GPU 可能仍在计算：

```text
Host 顺序：

record E_draft
      │
      ├─ enqueue target kernel 1
      ├─ enqueue target kernel 2
      ├─ enqueue target kernel 3
      │
      └─ launch grammar worker

CUDA main stream 队列：

... draft writes ... | E_draft | target K1 | target K2 | target K3
                           ▲
                           │
copy stream 只 wait E_draft，不 wait K1/K2/K3
```

因此 copy stream 可以从 `E_draft` 之后开始搬 draft token：

```text
main stream:  E_draft [target K1][target K2][target K3]──────────────>
                 │
copy stream:     └ wait E_draft [D2H]───────────────────────────────>
worker CPU:                         [XGrammar verify]────────────────>
```

这是这条异步链最容易漏掉的原理：

> launch 的 C++ 位置在 target `forward()` 之后，不代表 GPU 执行必须等 target 完成；决定依赖的是 CUDA event 记录在哪里，以及 copy stream wait 哪个 event。

用“发快递”类比：host 调用 `forward()` 更像把一批 target kernels 放进 main stream 的传送带，而不是站在原地等它们全部跑完。worker 随后启动时，只要求 copy stream 等到传送带上的 `E_draft` 标记；标记后面的 target 包裹仍可继续与 D2H 并行移动。

### 17.7 `materializeDraftTokensToCpu`：D2H 只阻塞 grammar worker

真实代码：

```cpp
GraphStreamGuard stream_guard(toGraphStream(copy_stream_));
if (task.draft_tokens_ready_event) {
    task.draft_tokens_ready_event->block(copy_stream_);
}
dst.copy_(draft_i32, /*non_blocking=*/true);
copy_stream_.synchronize();
```

依赖图：

```text
main CUDA stream                  grammar copy stream                grammar worker CPU
────────────────                 ───────────────────                ──────────────────
write draft tokens
record E_draft ────────────────> wait E_draft
enqueue target verify             async D2H draft ────────────────> synchronize(copy_stream)
继续跑 target verify                                                   │
                                                                       ▼
                                                               可以安全读 pinned CPU draft
```

这里确实有 `synchronize()`，但同步的是：

```text
被阻塞：grammar worker thread
未被阻塞：engine main thread
未要求完成：main stream 上排在 E_draft 后面的 target verify kernels
```

所以它仍然能形成 CPU/GPU overlap。

D2H 完成后的接力关系：

```text
draft_token_ids_t (GPU)
      │ E_draft + async copy
      ▼
draft_tokens_cpu (pinned)
      │ copy_stream_.synchronize() 只保证这一小块可读
      ▼
grammar worker 进入 buildInline/tryAcceptAndFillBitmask

同时：main CUDA stream 仍在计算 L0..LP
```

### 17.8 `buildSpecLogitsVerifyInline` 收集 processor snapshot

`MtpExecutor` 为每个 stream 中实现 `SpecLogitsProcessor` 的 processor 建任务：

```cpp
for (const auto& stream : streams) {
    size_t processor_idx = 0;
    for (const auto& processor : stream->getAllLogitsProcessorPtr()) {
        auto spec = std::dynamic_pointer_cast<SpecLogitsProcessor>(processor);
        if (spec) {
            task.active.push_back({spec,
                                   stream_idx,
                                   processor_idx,
                                   stream->streamId(),
                                   stream->seqLength(),
                                   stream->outputTokenLen()});
        }
        ++processor_idx;
    }
}
```

任务结构：

```text
LaunchTask
├── total_streams = B
├── propose_step  = P
├── vocab_size    = V
├── draft_tokens  = CUDA [B, P] 或 [B, P+1]
├── draft_tokens_ready_event
└── active[]
    ├── processor ptr
    ├── stream_idx
    ├── processor_idx
    ├── stream_id
    ├── base_seq_len
    └── base_output_len
```

`stream_id + processor_idx` 后面构成精确 processor identity，避免“batch 里只要有一个 grammar，就把所有 processor 都跳过”的错误。

snapshot 中的长度字段还承担版本检查作用：

```text
launch 时 snapshot
├── stream_id
├── base_seq_len
└── base_output_len
         │
         ▼
worker 必须基于这一轮看到的 committed state 构造 mask
```

如果上一轮 commit 尚未完成却启动下一轮 grammar worker，即使 processor 指针仍有效，matcher 版本也可能落后一轮；第 18.9 节的窄同步就是为这条跨轮依赖服务。这里先记住：**异步 task 可以晚执行，但不能基于错误版本执行。**

### 17.9 `SpecLogitsVerifyRunner::buildInline` 的 CPU 数据流

主过程可以压成：

```cpp
ensureBuffersFit(B, P, V, W);
materializeDraftTokensToCpu(task);

fillAllAllow(merged);
for (const auto& item : task.active) {
    fillAllAllow(proc_mask);
    int cap = item.processor->tryAcceptAndFillBitmask(request);
    bitwiseAndInplace(merged_row, proc_mask, (P + 1) * W);
    spec_cap[item.stream_idx] = min(old_cap, cap);
    result.applied_processors.push_back({stream_id, processor_idx});
}
```

形状图：

```text
B 个 stream，P=3，W=ceil(V/32)

processor_bitmask_cpu_：单 processor 临时结果
┌─────────┬─────────────────────────────┐
│ row 0   │ W 个 int32 packed bits      │  state S0 的合法 token
│ row 1   │ W 个 int32 packed bits      │  S0 + d1
│ row 2   │ W 个 int32 packed bits      │  S0 + d1+d2
│ row 3   │ W 个 int32 packed bits      │  S0 + d1+d2+d3
└─────────┴─────────────────────────────┘

merged_bitmask_cpu_：batch 总结果
┌──────────────┬─────────────────────────┐
│ stream 0 row0│ processors 按位 AND     │
│ stream 0 row1│ processors 按位 AND     │
│ stream 0 row2│ processors 按位 AND     │
│ stream 0 row3│ processors 按位 AND     │
├──────────────┼─────────────────────────┤
│ stream 1 row0│ ...                     │
│ ...          │ ...                     │
└──────────────┴─────────────────────────┘
```

多个约束组合时是集合交集：

```text
allowed_final = allowed_processor_A ∩ allowed_processor_B ∩ ...

packed bits：
11101111  A
11011111  B
────────
11001111  A & B
```

当前实现并没有按 stream 开 worker pool 并行 XGrammar：

```text
一个 spec_logits_verify_async_runner_ worker
      │
      ├─ stream 0 / processor 0
      ├─ stream 1 / processor 0
      ├─ stream 2 / processor 0
      └─ ... 顺序循环
```

“异步”指它与 engine/main GPU 并行，不表示 batch 内的 matcher 彼此并行。

继续固定例子，在 worker CPU 内 Artifact A 会变成：

```text
draft_cpu = [ ':', '"ok"', ',' ]
matcher entry state = S0

tryAcceptAndFillBitmask
├─ row0 -> M0；':' allowed；临时 S0->S1
├─ row1 -> M1；'"ok"' allowed；临时 S1->S2
├─ row2 -> M2；',' denied；cap=2
└─ rollback(2)：S2->S0

CPU result
├─ packed masks M0,M1,M2,...
└─ cap=2
```

这正是第 12 章同步语义的同一段算法；区别只是执行者从 main thread 换成了 grammar worker。

### 17.10 dsv4 的 think + grammar processor

`dsv4_on_dev` 中，如果同时有 grammar constraint 与 `in_think_mode`，factory 创建的是：

```cpp
if (config->in_think_mode) {
    return std::make_shared<ReasoningGrammarLogitsProcessor>(
        matcher,
        eos_token_id,
        config->max_thinking_tokens,
        config->begin_think_token_ids,
        config->end_think_token_ids,
        generate_input->inputLength(),
        error_reporter);
}
```

它内部有两个状态对象：

```text
ReasoningGrammarLogitsProcessor
├── StreamThinkInfo think_info_
│   ├── IN_THINK
│   ├── CLOSING_THINK
│   ├── AFTER_THINK
│   └── current_output_length
└── RtpGrammarMatcher matcher_
    └── 只负责 think 结束后的 JSON/regex/structural grammar
```

状态机：

```text
                       达到 max_thinking_tokens
                               │
                               ▼
IN_THINK ──看到 end-think 前缀──> CLOSING_THINK ──完整 end-think──> AFTER_THINK
   │                                  │                                  │
   │ allow reasoning                  │ force 剩余 end token             │ XGrammar mask
   │ mask EOS                         │ mask EOS                          │ JSON/structural
   └──────────────────────────────────┴──────────────────────────────────┘
```

因此 17.9 的 `item.processor->tryAcceptAndFillBitmask(request)` 在运行时可能分派到两类实现：

```text
普通 structural request  ──> Grammar processor：只预演 matcher
think + structural request ──> ReasoningGrammar processor：
                                copy think state + 预演 matcher
```

runner 不关心 processor 内部有一个还是两个状态机；它只消费统一的 `mask rows + cap + error` 接口。

### 17.11 `tryAcceptAndFillBitmask` 是一个可回滚事务

Reasoning + grammar 的 spec verify 核心：

```cpp
std::lock_guard<std::mutex> lock(mutex_);

auto think_state = think_info_.copy();
int grammar_accepted_prefix = 0;

for (int offset = 0; offset <= P; ++offset) {
    fill_row(mask[offset]);
    if (offset == P) break;

    int32_t draft_token = request.draft_tokens[offset];
    if (!bitmaskAllowsToken(mask[offset], draft_token)) {
        cap = offset;
        break;
    }

    if (token_belongs_to_grammar) {
        if (!matcher_->acceptToken(draft_token)) {
            cap = offset;
            break;
        }
        ++grammar_accepted_prefix;
    } else {
        advanceThinkStateForSpec(think_state, draft_token);
    }
}

matcher_->rollback(grammar_accepted_prefix);
return cap;
```

事务图：

```text
永久状态（进入函数前）
┌────────────────────────────────────────────┐
│ think_info_ = IN_THINK / matcher = S       │
└──────────────────────┬─────────────────────┘
                       │ copy think state
                       ▼
临时状态
┌────────────────────────────────────────────┐
│ think_state(copy)                          │
│ matcher: S -> S+d1 -> S+d1+d2 ...          │
└──────────────────────┬─────────────────────┘
                       │ 生成 mask/cap
                       ▼
rollback(grammar_accepted_prefix)
                       │
                       ▼
永久状态恢复为 S；think_info_ 从未被改写
```

跨 think 边界的例子：

```text
起点：CLOSING_THINK，P=3
draft： [end_2, '{', '"answer"']

offset 0 mask: 只允许 end_2
    accept end_2 on copied think_state
    copied think_state -> AFTER_THINK

offset 1 mask: XGrammar(S) 的 JSON 起始 mask，允许 '{'
    matcher provisional accept '{'

offset 2 mask: XGrammar(S+'{')，允许 JSON key
    matcher provisional accept '"answer"'

offset 3 mask: XGrammar 下一位置

函数结束：matcher rollback 2；真实 matcher 仍是 S
```

draft 在边界处非法：

```text
offset 0: end_2      合法
offset 1: 普通文本   JSON 起点不合法
                     │
                     ▼
                   cap=1

输出最多保留：前 1 个合法 draft + target row1 的合法 replacement
```

第 16 章与第 17 章在这里完成对应：

```text
第 16 章的逻辑动作                    第 17 章的异步载体
──────────────────────────────────────────────────────────────
copy think state                      grammar worker stack/local object
provisional matcher accept            processor mutex 内 CPU 调用
fill per-position masks               pinned processor/merged buffers
rollback                              worker 返回前完成
actual commit                         不在本 worker；交给第 18 章 bookkeeping
```

所以 grammar worker 返回时，真实 think/matcher 状态必须仍是进入时版本；否则 main stream 还没完成概率 rejection，worker 已经猜测性地污染了永久状态。

### 17.12 packed bitmask 为什么被展开成 bool vocab mask

XGrammar 输出 packed allow bitmask：

```text
1 bit / token
shape = [B*(P+1), ceil(V/32)] int32
bit 1 = allowed
```

当前 dsv4 runner 在 CPU 展开为 sampler 通用 bool mask：

```cpp
for (size_t token = 0; token < vocab_size; ++token) {
    bool allowed = word & (1u << (token % 32));
    row_mask[token] = !allowed;
}
```

展开后：

```text
1 byte / token
shape = [B*(P+1), V] bool
true = masked/disallowed
```

转换图：

```text
XGrammar packed allow bits
       [.... int32 ....]
                 │ CPU unpack + invert
                 ▼
Sampler bool disallow mask
       [false true false ...]
```

这让 sampler 与 grammar backend 解耦，但代价也很明确：

```text
packed mask 大小 ≈ rows * V / 8 bytes
bool mask 大小   ≈ rows * V bytes

约放大 8 倍，然后整块 H2D。
```

如果 profile 中 XGrammar 支路仍偏长，这一段 CPU 展开和 bool-mask H2D 是首要观察点。

从 Artifact B 的角度，representation 连续变化但语义不变：

```text
允许集合 Mi
   │ XGrammar 输出
   ▼
packed allow bits CPU        bit=1 means allowed
   │ unpack + invert
   ▼
bool disallow mask CPU       true means masked
   │ H2D
   ▼
bool disallow mask GPU
   │ masked_fill_(..., -inf)
   ▼
constrained target logits
```

读代码时最容易在这里把 0/1 语义看反：packed 阶段是 allow-mask，bool 阶段是 disallow-mask。

### 17.13 mask/cap H2D 与 `ready_event`

runner 使用本轮独立分配的 pinned owner：

```cpp
auto mask_cpu = torch::empty({rows, V}, pinned_bool);
auto cap_cpu  = torch::empty({B}, pinned_i32);
auto mask_gpu = torch::empty({rows, V}, cuda_bool);
auto cap_gpu  = torch::empty({B}, cuda_i32);

GraphStreamGuard guard(toGraphStream(copy_stream_));
mask_gpu.copy_(mask_cpu, /*non_blocking=*/true);
cap_gpu.copy_(cap_cpu, /*non_blocking=*/true);
ready->record(copy_stream_);
```

返回的 artifact：

```text
LaunchResult
├── spec_vocab_mask_gpu       GPU consumer 数据
├── spec_cap_gpu              GPU consumer 数据
├── ready_event               producer 已经入队完成的边界
├── consumed_event            consumer 已读完的边界
├── applied_processors[]      哪些 processor 已被 artifact 覆盖
├── spec_vocab_mask_cpu_owner pinned H2D source 保活
└── spec_cap_cpu_owner        pinned H2D source 保活
```

producer/consumer 生命周期：

```text
CPU fill mask/cap
      │
      ▼
copy stream H2D
      │
      ├─ record ready_event ───────────────────┐
      │                                        ▼
      │                                sampler stream wait
      │                                        │
      │                                masked_fill + cap read
      │                                        │
      └──────────────────── record consumed_event
```

两个 event 夹住了 GPU consumer 的生命周期：

```text
CPU owners 必须保活
    │
    ├─ H2D enqueue ── ready_event ──> GPU mask/cap 可读
    │                                  │
    │                                  ├─ apply mask
    │                                  ├─ sample
    │                                  └─ cap/replacement
    │                                             │
    └──────────────────────── consumed_event <────┘
```

`ready_event` 防止“数据还没传完就读”，`consumed_event` 防止“消费者还在读就复用/释放 storage”。

### 17.14 `AsyncRunner::sync` 是采样前的 join point

main thread 在采样前显式 join：

```cpp
if (spec_logits_async_launched) {
    spec_logits_verify_async_runner_.sync(graphGetCurrentStream());
}
```

这里先做两件事：

```text
1. Host wait：等 CPU XGrammar task 构造出 LaunchResult
2. CUDA wait：让 main stream 等 runner 的 event_
```

但 `SpecLogitsVerifyRunner` 内部又使用了独立的 `copy_stream_`。因此 runner 自己的 `event_` 不能替代 artifact 的 `ready_event`：

```text
spec runner stream: task.fn 返回 ── record runner event
                         │
                         │ task.fn 只保证 H2D 已经入队
                         ▼
copy stream       : [mask/cap H2D] ── record ready_event
                                              │
                                              ▼
main sampler      : wait ready_event ──真正读取 mask/cap
```

也就是说：

```text
AsyncRunner::sync  = LaunchResult 对象/CPU task 的 join
artifact ready     = mask/cap GPU 数据的 join
```

因此本轮耗时不是：

```text
target verify + XGrammar verify
```

两层 join 都满足后，本轮 sampler 前的耗时更接近：

```text
max(target verify GPU,
    draft D2H + CPU XGrammar + mask H2D)
+ sampler/rejection
```

理想与退化：

```text
Case A：target 慢，grammar 藏住

target : [==============================]
grammar:     [D2H][CPU][H2D]
join   :                                ▲ 几乎不等 grammar

Case B：grammar 慢，target 先结束

target : [==============]
grammar:     [D2H][========CPU========][H2D]
join   :              target done ───────▲ 等 grammar 尾巴
```

join 后，第 11 章的 target 分支与第 12/16 章的 grammar 分支终于重新汇合：

```text
main stream 已有 [L0..LP]
          AND
Artifact B 已有 mask/cap + ready_event
          │
          ▼
gatherSpecSamplerInput / LogitsProcessorStates
```

join 只保证“可以开始消费”，并不在 host 上把整个 copy stream 全局 synchronize；真正的 GPU read 仍通过 `ready_event->block(current_stream)` 建立依赖。

### 17.15 sampler 只跳过确实已经合入 artifact 的 processor

`gatherSpecSamplerInput()` 把 artifact 接到 sampler：

```cpp
sampler_inputs.phase                    = LogitsProcessorPhase::MTP_VERIFY;
sampler_inputs.spec_vocab_mask_gpu      = result.spec_vocab_mask_gpu;
sampler_inputs.spec_cap_gpu             = result.spec_cap_gpu;
sampler_inputs.spec_mask_ready_event    = result.ready_event;
sampler_inputs.spec_mask_consumed_event = result.consumed_event;
sampler_inputs.spec_applied_processors  = result.applied_processors;
```

`LogitsProcessorStates::batchProcess()`：

```cpp
if (has_spec_mask) {
    inputs.spec_mask_ready_event->block(current_stream);
    inputs.logits.masked_fill_(inputs.spec_vocab_mask_gpu, neg_inf);
}

for (size_t i = 0; i < logits_processors_.size(); ++i) {
    if (has_spec_mask
        && isSpecProcessor(logits_processors_[i])
        && isProcessorApplied(inputs, processor_ids_[i])) {
        continue;
    }
    logits_processors_[i]->process(...);
}
```

正确语义：

```text
processor A 已写入 artifact ──> sampler 不再执行 A，避免重复 mask/改状态
processor B 未写入 artifact ──> sampler 仍执行 B
```

而不是：

```text
batch 中存在任意 spec mask ──X──> 跳过所有 SpecLogitsProcessor
```

processor identity：

```text
(stream_id, processor_idx)

stream A processor 0 != stream B processor 0
stream A processor 0 != stream A processor 1
```

这一节完成 Artifact B→C：

```text
Artifact B                         target logits
mask_gpu [rows,V]                  [L0..LP]
cap_gpu [B]                             │
applied_processor ids                   │
       │                                │
       └────────────┬───────────────────┘
                    ▼
      wait ready -> masked_fill(-inf)
                    │
       未包含的 processor 再补执行
                    │
                    ▼
              Sampler::forward
                    │
                    ▼
Artifact C：target tokens/probabilities
```

固定例子中，row2 的逗号 logit 会在这里变成 `-inf`，所以 sampler 为 row2 产生合法 fallback `}`。

### 17.16 probability rejection 与 grammar cap 的汇合

先执行标准 speculative rejection：

```cpp
speculative_sampler_output = speculative_sampler_->forward(
    streams, draft_sampler_output, sampler_output);
```

再执行 grammar cap：

```cpp
output.accept_len = min(output.accept_len, spec_cap_gpu + 1);
```

为什么是 `cap + 1`：

```text
cap = 第一个非法 draft 的 offset

合法 draft 前缀：d0 ... d(cap-1)       共 cap 个
replacement：    target row[cap]         再加 1 个

最多输出长度 = cap + 1
```

二维决策图：

```text
probability rejection 给出的长度 = L_prob
grammar 第一个非法位置        = cap

最终长度：L_final = min(L_prob, cap + 1)
```

例子：

```text
P=3
draft                 d0      d1      d2
grammar               OK      BAD      -
probability           accept  accept  reject

grammar cap = 1
prob length  = 3（假设含 bonus/replacement）
final length = min(3, 2) = 2

accepted tokens：
  [d0, target_sample_at_row1]
```

非法位置 replacement：

```cpp
auto replacement = target_tokens.gather(1, cap_index.unsqueeze(1));
output.accept_tokens = torch::where(
    replace_mask,
    replacement,
    output.accept_tokens);
```

mask/cap 的消费边界在 replacement 完成后记录：

```cpp
if (sampler_input.spec_mask_consumed_event) {
    sampler_input.spec_mask_consumed_event->record(current_stream);
}
```

继续固定例子：

```text
Artifact A draft        [ ':', '"ok"', ',' ]
Artifact B grammar      cap=2，row2 只允许 '}'
Artifact C target       target_sample[row2]='}'
probability result      假设 old_len=4
                              │
                              ▼
replacement at 2        ',' -> '}'
truncate                 new_len=min(4,3)=3
                              │
                              ▼
Artifact D actual        [ ':', '"ok"', '}' ]
```

到此 mask/cap 的 GPU consumer 已经全部完成，所以记录 `consumed_event`。但 Artifact D 仍只存在于 speculative output/device state，真实 stream/matcher 尚未永久更新：

```text
GPU 已知道 actual tokens                 CPU committed state
[':','"ok"','}']                         仍是旧 history / matcher S0
         │                                      │
         └──────── 第 18 章 async commit ───────┘
```

### 17.17 当前异步链的准确评价

已经做到：

```text
✓ draft D2H 使用独立 copy stream
✓ CPU XGrammar 在 worker thread 执行
✓ 能与已入队的 target verify GPU kernels overlap
✓ mask/cap H2D 有 producer-ready event
✓ sampler 对 ready event 建 CUDA 依赖
✓ exact applied processor identity
✓ consumed event 标记 GPU consumer 边界
✓ worker exception 在 sync/下一次 launch 时重新抛到主线程
```

还没有做到：

```text
△ batch 内不同 stream 的 matcher 仍串行循环
△ CPU packed->bool 展开是 O(B*(P+1)*V)
△ bool mask H2D 比 packed mask 大约 8 倍
△ main host 在 sampler 前仍要 join grammar worker
△ 同一 request 的下一轮 grammar 依赖上一轮 committed state，不能无限跨轮流水
△ consumed_event 已记录，但当前每轮 mask/cap 是新分配，并没有完整的复用 slot pool
```

一句话：

> 这是“本轮 target GPU 与本轮 CPU XGrammar 重叠”的异步，不是“grammar 可以落后若干轮再提交”的异步。

## 18. `dsv4_on_dev` 的异步 commit 与下一轮衔接

XGrammar artifact 解决的是“本轮怎么合法采样”；stream bookkeeping 解决的是“采样后怎样不阻塞主线程，同时又让下一轮拿到状态”。

它直接接住第 17 章的 Artifact D：

```text
第 17 章结束
GPU actual tokens = [a1..aN]
GPU next draft state = [d1_next, probs, hidden]
CPU stream/matcher = 仍是 round N 进入时的旧状态
              │
              ├─────────────┬────────────────────┐
              ▼             ▼                    ▼
      publish device state  bookkeeping worker   round N+1 prepare
              │             commit actual             │
              │             CPU state S->S'            │
              └─────────────┴──── narrow join ─────────┘
```

第 18 章内部可以分成“快轨”和“提交轨”：

```text
快轨（main/GPU）
Artifact D
  └─ publish MtpAsyncDeviceState
       └─ round N+1 model input/资源预留可提前开始

提交轨（bookkeeping worker/CPU）
Artifact D + next draft artifact
  └─ D2H accepted tokens
       └─ dispatchDecode
            └─ specUpdate
                 └─ stream history + matcher permanent commit

汇合点
  └─ round N+1 在再次读取 stateful grammar matcher 之前
```

继续固定例子：

```text
Artifact D actual = [':','"ok"','}']

快轨先发布：accept_len=3、last token='}'、next seq len、next draft state
提交轨稍后执行：
  CompleteTokenIds append [':','"ok"','}']
  matcher S0 -> TERMINATED
  committed_output_len += 3
```

快轨能替代 GPU/model 所需数据，但不能伪装成 matcher 已提交；这条边界贯穿 18.2～18.10。

### 18.1 rejection 与 draft 两个 ready event

rejection 结果刚可用时：

```cpp
if (useStreamAsync()) {
    rejection_event = std::make_shared<torch::Event>(makeGraphEvent());
    rejection_event->record(graphGetCurrentStream());
}
```

下一轮 draft prefill/sample 结果刚可用时：

```cpp
if (useStreamAsync()) {
    draft_event = std::make_shared<torch::Event>(makeGraphEvent());
    draft_event->record(graphGetCurrentStream());
}
```

两份数据的生产者不同：

```text
E_rejection 保护：
├── accept_len_gpu
├── accept_tokens_gpu
├── accept_len_cpu 的异步 D2H
└── accept_tokens_cpu 的异步 D2H

E_draft 保护：
├── 下一轮 propose token
├── draft all_probs
└── draft hidden states
```

### 18.2 为什么先发布 `MtpAsyncDeviceState`

如果把一切都等 CPU `specUpdate` 完成，下一轮无法提前准备。dsv4 在 main stream 上先构造 device-resident state：

```cpp
GenerateStream::MtpAsyncDeviceState state;
state.accept_len_gpu         = accept_len_gpu_all.narrow(...);
state.accept_tokens_gpu      = accept_tokens_gpu_all.narrow(...);
state.propose_tokens_gpu     = propose_tokens_gpu_all.narrow(...);
state.next_seq_len_gpu       = next_seq_len_all.narrow(...);
state.last_hidden_states_gpu = last_hidden_all.narrow(...);
state.draft_all_probs_gpu    = draft_probs_all.narrow(...);
state.last_real_seq_len      = stream->seqLength();
state.next_real_seq_len      = state.last_real_seq_len + propose_step_ + 1;
stream->setMtpAsyncDeviceState(std::move(state));
```

Host 与 device 双轨状态：

```text
                 本轮 sampler/rejection
                         │
          ┌──────────────┴───────────────┐
          │                              │
          ▼                              ▼
GPU fast path                       CPU committed path
MtpAsyncDeviceState                 GenerateStream fields
├ accept_len/tokens                 ├ complete_token_ids
├ next_seq_len                      ├ seqLength
├ propose token                     ├ sp_output_buffer
├ hidden state                      ├ finish status
└ draft probs                       └ Grammar matcher committed state
          │                              │
          ▼                              ▼
下一轮 model input 可先准备          bookkeeping worker 稍后更新
```

device state 能替代的读取：

```text
✓ 下一轮 target/draft token
✓ 下一轮 sequence length tensor
✓ 下一轮 hidden state
✓ 下一轮 draft probability
✓ scheduler KV 预留长度上界
```

它不能替代：

```text
✗ XGrammar matcher 的 committed parser state
✗ stop/eos 的最终 host 状态
✗ 对外 output queue
```

这就是后面仍需“窄同步”的原因。

### 18.3 `next_real_seq_len` 为什么用上界

main thread 尚不知道 CPU 最终会如何把 host stream 状态收束，但 KV 分配不能少：

```text
旧真实 seq_len = S
一轮最多输出   = P+1

异步发布时：next_real_seq_len = S + P + 1
```

即使最终只接受 2 个 token：

```text
真实需要：S+2
提前预留：S+P+1

多预留可回收；少预留可能越过 block boundary 后缺块。
```

`GenerateStateMachine::handleRunning()` 优先读这个 override：

```text
MtpAsyncDeviceState.next_real_seq_len
                 │
                 ▼
StreamCacheResource::incrKVBlock(reserve_step, seq_len_override)
```

### 18.4 `dispatchDecodeAsync` 把 bookkeeping 交给 worker

发布 GPU state 后，主线程只做引用计数并 launch：

```cpp
for (auto& stream : streams) {
    stream->incPendingAsyncBookkeeping();
}

spec_bookkeeping_runner_.launch([
    processor,
    stream_groups_copy,
    spec_decode_copy,
    draft_prefill_copy,
    rejection_event,
    draft_event]() mutable {

    if (rejection_event) rejection_event->block(current_stream);
    if (draft_event)     draft_event->block(current_stream);

    processor->dispatchDecode(
        stream_groups_copy,
        spec_decode_copy,
        draft_prefill_copy);
});
```

主线程与 worker 的分叉：

```text
main thread
  │ publish MtpAsyncDeviceState
  │ inc pending count
  ├──────────────────────────────> bookkeeping worker
  │                                  wait E_rejection
  │                                  wait E_draft
  │ return decodeStep                prepareDecodeSpecUpdateInfo
  │                                  updateProposeTokens
  ▼                                  StreamGroups::updateStreams
可进入调度/下一轮                         │
                                         ▼
                                   GenerateStream::specUpdate
```

### 18.5 bookkeeping worker 中发生了哪些同步

`prepareDecodeSpecUpdateInfo()`：

```cpp
spec_decode_output.transfer_done_event->synchronize();
const auto& accept_len    = spec_decode_output.accept_len_cpu;
const auto& accept_tokens = spec_decode_output.accept_tokens_cpu;
```

然后逐 stream 读取 CPU scalar 并切 accepted token：

```cpp
int cur_accept_len = accept_len[batch_idx].item<int>();
auto accepted = accept_tokens.narrow(0, batch_idx, next_batch_size)
                             .narrow(1, 0, cur_accept_len)
                             .contiguous();
```

同步发生的位置：

```text
main engine thread       bookkeeping worker       GPU/D2H
──────────────────       ──────────────────       ─────────────
继续返回/调度             wait transfer event <── accept CPU copy
                         读 accept_len
                         组装 update info
```

昂贵的 D2H 等待和 `.item()` 没消失，而是被移出了 main engine thread 的关键路径。

### 18.6 `specUpdate` 是唯一的永久 commit 点

worker 最终调用：

```text
MtpBatchStreamProcessor::dispatchDecode
  ├─ prepareDecodeSpecUpdateInfo
  ├─ updateProposeTokens
  └─ StreamGroups::updateStreams
       └─ GenerateStream::specUpdate
```

`specUpdate()` 的核心顺序：

```cpp
int old_seq_length = seqLength();

complete_token_ids_->update(new_tokens, ..., num_new_tokens, ...);
updateOutput(...);

int committed = std::max(0, seqLength() - old_seq_length);
if (committed > 0) {
    updateLogitProcessorStatus(
        new_tokens, committed, torch::Tensor(), /*stateful_only=*/true);
}
validateStatefulLogitsProcessorState();
```

永久状态推进图：

```text
accepted token tensor
        │
        ▼
complete_token_ids_->update
        │  可能因为 max token / stop 截断实际提交数量
        ▼
updateOutput / finish check
        │
        ▼
committed = new_seq_len - old_seq_len
        │
        ▼
ReasoningGrammarLogitsProcessor::updateStatus(committed tokens)
        │
        ├─ think_info_ 永久推进
        └─ matcher_->acceptToken 永久推进
```

这也回答了最初“结束时少更新一个 token 是否可以”的问题：

```text
stream output length = N
grammar committed len = N-1
```

即使 response 已经结束，它仍会破坏：

```text
validateStatefulLogitsProcessorState()
PD/异步 worker 的状态一致性
terminal EOS/think-end 的完成语义
以后复用或诊断时的 committed invariant
```

所以 dsv4 也改成了：先计算实际 committed 数量，再无条件提交这批实际 token；是否 finished 不再决定“要不要更新 grammar”，只可能决定是否跳过非 stateful processor。

把第 17→18 章的状态变化放在一张表里：

```text
时刻                         GPU device state              CPU stream/matcher
──────────────────────────────────────────────────────────────────────────────
round N verify 前             proposal d1..dP              history H, matcher S
provisional grammar 后        masks/cap                    仍是 H,S（已 rollback）
sample/rejection 后           actual a1..aN                仍是 H,S
publish async state 后         actual + next-draft 可读      仍是 H,S
bookkeeping specUpdate 后      同一批 device tensor          history H+a, matcher S'
round N+1 grammar 前           next proposal                 必须已经看到 S'
```

### 18.7 为什么 async worker 需要 pending 引用计数

危险时序：

```text
worker 捕获 stream/KV block
       │
request 被 cancel/finish
       │
scheduler release KV block
       │
block 被别的请求复用
       │
旧 worker 醒来写旧 block  -> 内存/结果破坏
```

保护协议：

```text
dispatch worker 前：count++
worker 退出 guard：count--

releaseResource：
  count == 0 -> 立即释放
  count > 0  -> defer_release=true

最后一个 worker count-- 到 0：
  如果 defer_release -> worker 尾部执行 release
```

代码骨架：

```cpp
void GenerateStream::releaseResource() {
    waitPendingAsyncBookkeeping();
    ... release ...;
}

void GenerateStream::decPendingAsyncBookkeepingAndMaybeRelease() {
    if (count.fetch_sub(1) == 1) {
        cv.notify_all();
        if (defer_release.exchange(false)) {
            releaseResource();
        }
    }
}
```

生命周期图：

```text
ALIVE ──launch worker──> ALIVE + pending=1
  │                            │
  └─finish/cancel──────────────┤ mark deferred release
                               │
                         worker completes
                               │ pending=0
                               ▼
                         RELEASE KV SAFELY
```

### 18.8 下一轮为什么大部分计算可以不等 commit

下一轮 draft/model prepare 优先从 `MtpAsyncDeviceState` 取数据：

```text
round N main stream 产生 GPU state
           │
           ├──────────────> round N bookkeeping：host commit 仍在跑
           │
           └──────────────> round N+1 gather/draft：读取 GPU state
```

例如上一轮最后接受 token 可以直接在 GPU gather：

```text
accept_tokens_gpu [B, P+1]
accept_len_gpu    [B]
          │
          ▼
index = accept_len - 1
          │
          ▼
pre_target_token_gpu [B]
```

不需要先等待 CPU 把 `sp_output_buffer->tokens[0]` 写好。

### 18.9 但 XGrammar 下一轮必须看到上一轮 commit

Grammar matcher 是 CPU stateful parser：

```text
round N spec verify：基于 committed state S_N
round N accepted：a1..ak
round N commit：S_N -> S_N+k
round N+1 spec verify：必须从 S_N+k 开始
```

错误时序：

```text
round N bookkeeping 尚未 commit a1..ak
                     │
round N+1 grammar worker 读取旧 matcher S_N
                     │
生成旧位置 mask
                     ▼
约束错误
```

所以当开启 stream async 且 drop broad sync 时，代码在启动下一轮 spec logits 前做窄同步：

```cpp
if (useStreamAsync() && useDropBroadSync()) {
    spec_bookkeeping_runner_.sync(graphGetCurrentStream());
    stream_groups = StreamGroups(streams);
    prev_bookkeeping_synced_for_spec_logits = true;
}
```

依赖图：

```text
round N bookkeeping worker
  specUpdate
  grammar updateStatus
  record runner event
          │
          ▼
round N+1 pre-spec-logits narrow sync
          │
          ▼
round N+1 XGrammar tryAcceptAndFillBitmask
```

因此当前跨轮结构是“部分流水”：

```text
可以越过 commit 的：GPU device-state 驱动的 model input/部分 prepare
不能越过 commit 的：下一轮 stateful XGrammar verify
```

### 18.10 同一个 processor 上 mutex 与 runner 顺序各保护什么

`ReasoningGrammarLogitsProcessor` 的 `mutex_` 保护同一瞬间的互斥：

```text
tryAcceptAndFillBitmask ─┐
                        ├─ mutex_，不能同时改 matcher
updateStatus             ┘
```

但 mutex 本身不保证先后语义：

```text
错误：round N+1 verify 抢先拿锁，基于旧 state 完成；round N commit 随后拿锁
```

因此还需要 runner sync 保证 happens-before：

```text
round N updateStatus 完成
          happens-before
round N+1 tryAcceptAndFillBitmask
```

两层保护：

```text
mutex       = 不并发破坏 matcher 内存
runner sync = 不以错误的版本顺序读取 matcher
```

### 18.11 attention prepare 的另一条 overlap

`RTP_LLM_MTP_ASYNC_PREPARE=1` 时，target verify 的 attention metadata 也异步准备：

```cpp
input_ready_event->record(current_stream);
target_verify_prepare_runner_.launch([input_ready_event, input] {
    input_ready_event->block(current_stream);
    model_->prepareAttentionInputs(input);
});
```

target 真正 forward 前 join：

```cpp
target_verify_prepare_runner_.sync(current_stream);
model_->forward(model_input);
```

时间线：

```text
prepare runner: [target attention prepare----------------]
main stream   :      [draft propose----------------------]
                                                      join
                                                       │
                                                       ▼
                                               target verify forward
```

draft prefill prepare 类似，但它提前于 target verify 启动，在 draft prefill forward 前 join：

```text
prepare runner: [draft-prefill attention prepare-------------]
main stream   :      [target verify][grammar][sample/reject]
                                                           join
                                                            │
                                                            ▼
                                                   draft prefill forward
```

### 18.12 PD 分离下这条异步链放在哪里

PD 不会把一个 matcher 跨进程共享；Decode 节点创建自己的 processor/matcher，并重放 Prefill 首 token。之后上述异步链全部发生在 Decode engine 内：

```text
Prefill node
  target prefill + constrained first token t0
  draft first proposal d1
       │
       │ RPC: t0 + MTP payload
       ▼
Decode node
  makeStream -> create its own Reasoning/Grammar matcher
  update([t0]) -> commit t0
       │
       ▼
MtpExecutor::decodeStep round 1
  ├─ draft propose
  ├─ target verify
  ├─ async XGrammar artifact
  ├─ sample/reject/cap
  └─ async bookkeeping commit
       │
       ▼
round 2 ...
```

PD、MTP、XGrammar 三者的状态所有权：

```text
PD RPC owns transport
  └─ 首 token / KV / draft payload

MTP executor owns per-round GPU tensors
  └─ proposals / target logits / accept_len / device state

GenerateStream + processor own committed request state
  └─ output tokens / finish / think state / grammar matcher
```

### 18.13 真实关键路径与性能判断

把所有等待压成一张图：

```text
round N

draft proposal
     │
     ├──────────────────────────────────────────┐
     ▼                                          ▼
target verify GPU                    D2H + CPU XGrammar + H2D
     │                                          │
     └──────────────── join before sampler ─────┘
                           │
                           ▼
                  mask + sample + rejection
                           │
                           ▼
                draft prefill/sample for N+1
                           │
             ┌─────────────┴──────────────┐
             ▼                            ▼
 publish next GPU state            async host bookkeeping
             │                            │
             ▼                            │
 round N+1 GPU preparation                │
             │                            │
             └── before stateful grammar ─┘ narrow join
```

判断优化收益时应分别看：

```text
T_target       = target verify GPU 时间
T_grammar      = draft D2H + CPU matcher + CPU unpack + mask H2D
T_sample       = mask + target sample + rejection + cap
T_bookkeeping  = accept D2H + specUpdate + processor commit

本轮 sampler 前约为：max(T_target, T_grammar) + T_sample
跨轮暴露尾巴约为：下一轮到 grammar join 时尚未完成的 T_bookkeeping
```

### 18.14 当前实现最值得继续优化的三个点

第一，提前启动 worker host task。当前 worker 在 target `forward()` 已入队后 launch，GPU overlap 成立，但 host 调度稍晚：

```text
当前：draft ready -> enqueue target -> launch grammar worker
可比较：draft ready -> launch grammar worker -> enqueue target
```

需 profile 证明 host launch 顺序是否真的影响 overlap，不能只看源码行序下结论。

第二，保持 packed mask 到 GPU：

```text
当前：packed CPU -> bool CPU -> bool H2D -> masked_fill
候选：packed CPU -> packed H2D -> CUDA packed-mask kernel
```

第三，batch 内 matcher 并行：

```text
当前一个 worker：A -> B -> C -> D
候选 worker pool：A || B || C || D -> merge
```

前提是同一 stream/processor 的 verify 与 commit 仍严格串行。

### 18.15 用一句伪代码串起真实实现

```cpp
// round N
draft_tokens = draftModelDecode();
record(E_draft_ready);

enqueueTargetVerify();  // main CUDA stream

grammar_runner.launch([&] {
    copy_stream.wait(E_draft_ready);
    drafts_cpu = asyncD2H(draft_tokens);
    copy_stream.synchronize();       // only worker waits
    artifact_cpu = xgrammarTryAndRollback(drafts_cpu);
    artifact_gpu = asyncH2D(artifact_cpu);
    record(E_mask_ready);
});

grammar_runner.sync(main_stream);    // join target + grammar before sampler
applyExactProcessorMasks();
targetSample();
probabilityReject();
applyGrammarCapAndReplacement();
record(E_rejection);

draftPrefillAndSampleNextProposal();
record(E_draft_next);

publishMtpAsyncDeviceState();        // next round GPU fast path
bookkeeping_runner.launch([&] {
    wait(E_rejection, E_draft_next);
    accepted_cpu = waitD2H();
    stream.specUpdate(accepted_cpu); // permanent token + matcher commit
});
```

最短记忆版：

```text
Draft
  ├─ Target GPU verify ───────────┐
  └─ XGrammar CPU verify/rollback ├─> Mask + Reject + Cap
                                  │
                                  └─> GPU state 立即发布
                                      CPU commit 异步补齐
                                             │
                                             └─ 下一轮 grammar 前必须有序
```

至此第 11～18 章形成完整循环：

```text
round N
  proposal ─> target/grammar parallel verify ─> sample/reject/cap
       │                                           │
       │                                           ▼
       │                                  publish GPU next state
       │                                           │
       │                        ┌──────────────────┴─────────────┐
       │                        ▼                                ▼
       │                 CPU commit actual                 round N+1 prepare
       │                        │                                │
       └────────────────────────┴──── join before grammar ───────┘
```

## 19. 为什么不能把 XGrammar 直接放进 draft model sampling

这个问题现在可以直接用第 12～14 章的数据流回答。grammar correctness 有两个入口：

```text
入口 1：draft token di 是否能保留？             -> grammar cap
入口 2：di 被拒后，target replacement 是否合法？ -> target row mask Mi
```

只在 draft sampling 处加 XGrammar，只覆盖入口 1，覆盖不了入口 2。

看似可以让 draft 本身只提合法 token，但这不能替代 target verify mask：

```text
仅约束 draft：
draft d1,d2 合法
target probability rejection 在 d2 处拒绝
fallback target token 如果未 mask，仍可能非法

正确做法：
draft 可选地约束（提升 acceptance）
+ target P+1 每一行必须约束（保证 correctness）
```

如果未来也约束 draft，可以增加一条性能优化路径：

```text
committed matcher snapshot
       │ clone/replay-only state
       ▼
draft step 1 mask -> sample d1
draft step 2 mask -> sample d2
...

但 target verify 的 M0..MP 仍然必须保留。
```

固定例子中，即使 draft sampler提前禁止了逗号，也仍可能发生概率 rejection：

```text
draft 已受约束，提出合法 d1=':'
target probability rejection 拒绝 d1
             │
             ▼
必须从 target row0 采 replacement
             │
             └─ 若 L0 没套 M0，replacement 仍可能是非法 '='
```

所以“约束 draft”是提高 acceptance 的可选优化，“约束 target 的每一行”才是最终 correctness gate。

## 20. 输出与终止

第 20 章接的是第 15/18 章的 permanent commit，而不是第 14/17 章刚算出的 GPU speculative result：

```text
actual tokens on GPU
      │
      ▼
specUpdate / token history / processor commit
      │
      ├─ running  -> streaming output + 下一轮
      └─ finished -> final output + GenerateDone
```

### 20.1 `NormalGenerateStream::updateOutput`

每轮 `specUpdate` 最后进入这里：

```cpp
finished_ = needFinish();
if (finished_) {
    reportEventWithoutLock(StreamEvents::GenerateDone);
    fillSubGenerateStatus(StreamState::FINISHED);
}

if (isStreaming() || finished_) {
    enqueueGenerateOutput(prepareGenerateOutput(update_info));
}
```

终止条件包括：

```text
seqLength >= maxTokenNum
        OR
all sequences matched EOS/stop words
```

### 20.2 最终 token 仍需 commit

正确顺序：

```text
append accepted final token
        │
        ▼
Grammar updateStatus(final token)
        │
        ▼
check EOS/stop/max length
        │
        ▼
publish finished output
```

不能写成：

```text
publish/check finished
        │
        ├─ finished -> skip processor update   X
        └─ running  -> update processor
```

因为 PD Prefill 的 local done 不是 request done，而且统一 committed state 能防止 MTP/normal 生命周期分叉。

终止不是绕开 commit 的旁路，而是 commit 后的一个分支：

```text
                  actual token batch
                         │
                         ▼
                 authoritative commit
                         │
                         ▼
                    needFinish()
                    /          \
                  no            yes
                  │              │
             next round      final publish/release
```

## 21. 错误传播图

错误路径与成功路径使用同一组接力对象，只是某一步把 token/mask 产物替换成 `ErrorInfo`：

```text
成功：processor -> mask -> sample -> StreamUpdateInfo(tokens) -> commit/output
失败：processor -> error ─────────> StreamUpdateInfo(error)  -> report/stop
```

### 21.1 Normal/prefill mask 错误

```text
GrammarLogitsProcessor::process error
        │
        ▼
LogitsProcessorStates::processor_errors[row]
        │
        ▼
SamplerOutput.processor_errors
        │
        ▼
NormalOutputDispatcher::collectStreamSamplerError
        │
        ▼
StreamUpdateInfo.error_info
        │
        ▼
GenerateStream::update -> report Error -> 不发布 token
```

### 21.2 MTP verify 错误

```text
prepareSpeculative error
        │
        ▼
SpecLogitsVerifyRunner::processor_errors[stream]
        │
        ▼
applySpecVerifyResult / SpeculativeSamplerOutput
        │
        ▼
StreamSpecUpdateInfo.error_info
        │
        ▼
GenerateStream::specUpdate -> report Error -> return
```

把错误点叠加到第 11～18 章主链：

```text
draft/target forward
      │ model error
      X
XGrammar provisional verify
      │ parser/mask error ───────────────> processor_errors[stream]
      X
mask/sample/rejection
      │ kernel/sampler error ────────────> sampler/spec output error
      X
specUpdate actual commit
      │ accept/rollback error ───────────> stream Error
      X
output
```

原则是：错误跟随原 stream identity 穿过 batch，不能退化成“去掉约束继续采样”。

## 22. 阅读时应始终抓住的五个不变量

前面大量函数和异步 event 最终都服务于下面这条状态环：

```text
committed S
   │
   ├─ 只读/可回滚地产生候选约束
   ▼
legal actual tokens
   │
   ├─ 恰好一次永久 commit
   ▼
committed S'
   │
   └─ 成为下一轮唯一合法起点
```

```text
Invariant 1
stream history 是唯一 authoritative token history。

Invariant 2
每次成功 append 的 token batch，所有 processor 恰好 updateStatus 一次，
包括最终 batch 和 PD Prefill 的首 token。

Invariant 3
prepareSpeculative 只允许 provisional accept；返回前 matcher 必须 rollback。

Invariant 4
MTP 最终 accept_len 同时受 probability rejection 和 grammar cap 限制。

Invariant 5
异步 bookkeeping 完成后，processor committed length 必须等于 stream output length；
terminal resource release 不能越过仍在执行的 bookkeeping worker。
```

用一张图收束：

```text
                        ┌─────────────────────┐
                        │ committed matcher S │
                        └──────────┬──────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              │                                         │
         normal process                          MTP provisional verify
         mask next logits                        masks P+1 rows + rollback
              │                                         │
              └────────────────────┬────────────────────┘
                                   ▼
                             legal sampling
                                   │
                                   ▼
                       actual accepted token batch
                                   │
                                   ▼
                    append stream tokens / output bookkeeping
                                   │
                                   ▼
                         updateStatus / commit
                                   │
                                   ▼
                  validate output_len == committed_len
                                   │
                                   ▼
                  terminal resource may be safely released
```

## 23. 推荐的源码阅读顺序

按下面顺序打开文件，最容易把状态串起来：

```text
请求/PD 建流
  response_format_builder -> PrefillRpcServer -> DecodeRpcServer
             │
             ▼
state owner
  GenerateStream -> LogitsProcessorFactory -> XGrammarBackend
             │
             ▼
一轮 MTP
  MtpExecutor -> SpecLogitsVerifyRunner -> GrammarLogitsProcessor
             │
             ▼
merge/commit/output
  MtpBatchStreamProcessor -> SpeculativeSampler -> NormalGenerateStream
```

1. [response_format_builder.py](../../rtp_llm/config/response_format_builder.py)：think 如何被包进 structural grammar；
2. [PrefillRpcServer.cc](../../rtp_llm/cpp/model_rpc/PrefillRpcServer.cc)：PD 编排与 handoff payload；
3. [DecodeRpcServer.cc](../../rtp_llm/cpp/model_rpc/DecodeRpcServer.cc)：Decode stream 创建和首 token 重放；
4. [GenerateStream.cc](../../rtp_llm/cpp/engine_base/stream/GenerateStream.cc)：processor 创建、normal update、spec update；
5. [LogitsProcessorFactory.cc](../../rtp_llm/cpp/models/logits_processor/LogitsProcessorFactory.cc)：constraint one-of 与 grammar processor 安装；
6. [XGrammarBackend.cc](../../rtp_llm/cpp/engine_base/grammar/XGrammarBackend.cc)：compile 与 matcher 创建；
7. [MtpExecutor.cc](../../rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc)：MTP prefill/decode 主流程；
8. [SpecLogitsVerifyRunner.cc](../../rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.cc)：draft D2H、mask merge、H2D；
9. [GrammarLogitsProcessor.cc](../../rtp_llm/cpp/models/logits_processor/GrammarLogitsProcessor.cc)：normal mask、provisional verify、commit/rollback；
10. [MtpBatchStreamProcessor.cc](../../rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.cc)：target logits mask、dispatch accepted tokens；
11. [SpeculativeSampler.cc](../../rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.cc)：概率接受；
12. [NormalGenerateStream.cc](../../rtp_llm/cpp/normal_engine/NormalGenerateStream.cc)：输出与终止。

第 17～18 节的 `dsv4_on_dev` 实现不在当前 checkout 中，按下面顺序用 `git show` 阅读：

```bash
git show feat/dsv4_on_dev:rtp_llm/cpp/normal_engine/AsyncRunner.cc
git show feat/dsv4_on_dev:rtp_llm/cpp/normal_engine/speculative/MtpExecutor.cc
git show feat/dsv4_on_dev:rtp_llm/cpp/models/logits_processor/SpecLogitsVerifyRunner.cc
git show feat/dsv4_on_dev:rtp_llm/cpp/models/logits_processor/LogitsProcessorStates.cc
git show feat/dsv4_on_dev:rtp_llm/cpp/models/logits_processor/ReasoningGrammarLogitsProcessor.cc
git show feat/dsv4_on_dev:rtp_llm/cpp/normal_engine/speculative/MtpBatchStreamProcessor.cc
git show feat/dsv4_on_dev:rtp_llm/cpp/engine_base/stream/GenerateStream.cc
git show feat/dsv4_on_dev:rtp_llm/cpp/engine_base/stream/GenerateStateMachine.cc
```

## 24. 对应测试入口

测试也可以按同一条主链定位，而不是按文件孤立地找：

```text
字段跨进程          QueryConverterTest
      │
processor/matcher   GrammarLogitsProcessorTest
      │
MTP masks/cap       MtpExecutorTest
      │
final commit        GenerateStreamTest
      │
端到端              Smoke q_r_mtp_grammar*
```

```text
QueryConverterTest
  └─ grammar/think 字段跨 proto 后仍可创建 processor

GenerateStreamTest
  └─ final token 在 finished output 前仍提交 processor

GrammarLogitsProcessorTest
  ├─ normal mask
  ├─ committed token
  ├─ MTP draft cap
  ├─ provisional rollback
  ├─ terminal EOS committed length
  └─ reasoning budget -> final grammar

MtpExecutorTest
  ├─ compact mask row mapping
  ├─ grammar cap 修正 accept_len
  ├─ processor error propagation
  └─ unsupported processor admission rejection

Smoke q_r_mtp_grammar*
  └─ reasoning + grammar + MTP 的端到端输出
```

最后可以用这条一句话检查自己是否读通：

> Prefill 用 normal grammar mask 产生首个 target token，PD 把首 token 与 MTP draft 状态送到 Decode；Decode 每轮让 XGrammar 对 draft 链做可回滚的逐位置预演，给 target 的 `P+1` 行 logits 加硬 mask，再由概率 rejection 与 grammar cap 共同确定实际输出，最后只把实际接受 token commit 到 matcher。
