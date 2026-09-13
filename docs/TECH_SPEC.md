# 技术规格：叁鼎芯供应链商品知识助手

版本：0.1（商城目录资料阶段）
更新日期：2026-09-13

## 1. 设计原则与参考边界

本工程采用“规格先行、核心类型明确、摄取与查询分离、外部模型可替换”的思路；参考了 `jerry-ai-dev/MODULAR-RAG-MCP-SERVER` 的 `clean-start` 工程骨架及 dev 分支从工程/配置、可替换基础设施、摄取、检索逐步交付的节奏。

实现从零编写，仅保留本项目需要的轻量目录与接口；不复制参考仓库的代码、提示词、配置、业务逻辑或 Skill 文件。v0.1 旧电子产品样例已从业务资料、测试题与本地索引迁出，当前资料是商城目录事实。

## 2. 范围

### 本期包含

1. Markdown Loader：解析 YAML 风格 front matter 与正文，校验五个必填 Metadata。
2. `SemanticRecursiveSplitter`：先按标题、段落、列表与句子寻找语义边界；超长片段再递归降级为字符窗口，保留少量 overlap。
3. Embedding 抽象：生产使用 Sentence Transformers 中文模型；测试使用确定性向量，避免测试依赖网络。
4. `ChromaStore`：使用本地持久化 Chroma collection，按 `source` 幂等替换、按向量 Top-K 查询，并可读取当前 chunk 构建内存 BM25。
5. 可配置检索：默认 Dense（BGE + Chroma）；确定性中文 BM25 保留商品名、数字、单位和规格；Hybrid 使用 rank-only RRF 融合两路名次。
6. 问答编排：硬安全门 → Top-K 检索与阈值门 → 请求级证据 ID → 上下文受限 LLM → 引用验证 → 统一用户来源（含 `source_url`）输出。
7. CLI、商城目录资料、18 题回归验收、32 题安全保留评估，以及 27 题检索压力集脚本。
8. 显式真实 LLM 生成质量评估：26 道独立题、逐题回答/证据/耗时追踪、规则化逐维检查和人工复核工作清单；普通测试与 CI 不调用 API。

### 明确不做

Rerank、多模态、Dashboard、MCP、异步队列、复杂观察体系、Ragas/Golden set 回归框架、正式权限体系与交易系统集成。BM25 和 RRF 仅用于可重复对照实验，均不是默认生产路径。

## 3. 数据契约

每个 Markdown 文档的 front matter 需有：

```yaml
---
source: catalog/marketplace-cork-painting-sanfangqixiang.md
source_url: https://txs.wyfdev.com/product/%e4%b8%89%e5%9d%8a%e4%b8%83%e5%b7%b7%e4%b8%bb%e9%a2%98%e7%a6%8f%e5%b7%9e%e8%bd%af%e6%9c%a8%e7%94%bb/
product_id: marketplace-cork-painting-sanfangqixiang
document_type: product
updated_at: 2026-09-13
---
```

`product_name` 是用于显式商品名匹配的可选扩展 Metadata；`product_id` 在平台资料中为 `GENERAL`。入库 chunk 额外写入 `chunk_id`、`chunk_index`、`heading` 和字符偏移。`updated_at` 被按字符串保存，便于 Chroma filter 与可追溯展示。`product_id` 只用于后端筛选、来源映射和测试，绝不写入 LLM prompt、最终答案或用户可见的 `sources`。

## 4. 数据流

```text
Markdown + front matter
  -> MetadataValidator / MarkdownLoader
  -> SemanticRecursiveSplitter
  -> EmbeddingProvider
  -> ChromaStore (vectors + metadata)

question
  -> SafetyGate
  -> Retriever(dense | bm25 | hybrid_rrf) -> Top-K -> relevance gate
  -> explicit-product / explicit-fact guard
  -> request-scoped evidence {S1, S2, ...} (no product_id)
  -> Context-bound LLM returns {answer, used_source_ids}
  -> validate IDs against this request's evidence -> returned_sources
  -> {answer, sources, handoff_required}
```

## 5. 接口与运行配置

生产 Embedding 是 `SentenceTransformerEmbedder`（默认 `BAAI/bge-small-zh-v1.5`）；LLM 是 `OpenAICompatibleLLM`，通过 `LLM_API_BASE`、`LLM_API_KEY`、`LLM_MODEL` 对接任一 Chat Completions 兼容端点。测试模式通过依赖注入提供 `HashingTestEmbedder` 和 `ContextEchoLLM`，绝不把测试后端伪装成生产语义模型。

Chroma collection 使用 cosine 空间，Dense 分数为 `1 - distance`。默认 `TOP_K=4`、`MIN_RELEVANCE=0.60`、`RETRIEVAL_MODE=dense`，由环境变量调节；`RRF_K=60` 与 `RRF_CANDIDATE_DEPTH=12` 只在 Hybrid RRF 生效；`EVALUATION_THRESHOLDS` 配置离线候选值。BM25 原始分只用于 BM25 排序，映射到 0–1 置信度接入同一相关性门；Hybrid 的排序只按 RRF 名次融合，置信度不参与融合。当前 0.60 来自独立 32 题保留集的比较，不使用 18 题回归集调节；详见 `docs/EVALUATION_REPORT.md`。三模式对照没有显示 Hybrid 质量收益，详见 `docs/HYBRID_RETRIEVAL_EXPERIMENT.md`。生产上线前仍需以审核后的真实问答记录再次验证。

真实生成评估脚本只从本地 `.env` 读取 `LLM_API_BASE`、`LLM_API_KEY` 和 `LLM_MODEL`，且必须显式执行，不能由单元测试或 CI 触发。它固定验证 Dense 与 0.60 阈值；对每题记录答案、完整 Top-K `retrieval_sources`、经模型引用和后端验证的 `returned_sources`、模型名、耗时、硬安全门和是否到达 LLM 边界。实际运行的原始回答与 trace 默认写入 gitignore 的 `data/runtime/`；`--prepare-only` 才更新无原始回答的透明 docs 报告。自动检查仅基于问题、检索证据、最终回答与预期要点，分别报告忠实度、相关性、完整性、来源正确性和安全合规；它不调用 LLM Judge，也不是绝对真相。所有自动失败题和确定性抽取的至少 20% “已调用 LLM 且自动通过”题必须人工复核；九道硬安全门题由独立安全测试保证，不进入生成答案的人工抽样。模型回答在写入本地追踪前会掩盖常见 API Key、邮箱和手机号模式。详见 `docs/GENERATION_EVALUATION_REPORT.md`。

## 6. 安全控制

`SafetyGate` 在检索和 LLM 前识别库存/现货、绕开关键词的“还能下单”、实时或指定日期交期/出库、订单级/批量/经销报价，以及食品保质期、配料和过敏原。供应商/经销商的资质、审核、门槛、区域等未公开规则同样在检索前转人工。命中后直接返回固定人工文案；低相关性同样返回该文案。查询还会优先限定问题中明确出现的 `product_name`，而“微波炉”、材质、洗碗机、认证、食品安全、定制范围等需明确资料支持的属性在目标商品 chunk 没有对应文字时固定转人工。

离线评估通过 `trace_retrieval()` 记录硬安全门决定与原始 Top-K，随后记录最终 answer/handoff；追踪不要求 LLM 暴露推理过程。

LLM 的系统提示只允许使用所给上下文、禁止填补缺失事实，并要求输出严格的 `{answer, used_source_ids}` 结构。证据片段只使用本次请求稳定分配的 `S1`、`S2` 等 ID；后端拒绝无效、伪造或缺失引用，并只将已验证 ID 映射成用户来源。比较多个明确商品时，若各商品资料均在上下文中，提示要求分别回答，不得因“比较”本身转人工。通用的“语义相近但证据不足”阈值保护仍未完成，故产品侧规定资料未覆盖的信息一律转人工。
