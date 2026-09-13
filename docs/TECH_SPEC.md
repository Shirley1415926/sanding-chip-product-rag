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
4. `ChromaStore`：使用本地持久化 Chroma collection，按 `source` 幂等替换、按向量 Top-K 查询。
5. 问答编排：硬安全门 → Top-K 检索与阈值门 → 上下文受限 LLM → 统一来源（含 `source_url`）输出。
6. CLI、商城目录资料、18 题回归验收脚本，以及独立保留集的阈值比较与逐题追踪脚本。

### 明确不做

BM25、RRF、Rerank、多模态、Dashboard、MCP、异步队列、复杂观察体系、Ragas/Golden set 回归框架、正式权限体系与交易系统集成。

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

`product_name` 是用于显式商品名匹配的可选扩展 Metadata；`product_id` 在平台资料中为 `GENERAL`。入库 chunk 额外写入 `chunk_id`、`chunk_index`、`heading` 和字符偏移。`updated_at` 被按字符串保存，便于 Chroma filter 与可追溯展示。

## 4. 数据流

```text
Markdown + front matter
  -> MetadataValidator / MarkdownLoader
  -> SemanticRecursiveSplitter
  -> EmbeddingProvider
  -> ChromaStore (vectors + metadata)

question
  -> SafetyGate
  -> EmbeddingProvider -> Chroma Top-K -> relevance gate
  -> explicit-product / explicit-fact guard
  -> Context-bound LLM
  -> {answer, sources, handoff_required}
```

## 5. 接口与运行配置

生产 Embedding 是 `SentenceTransformerEmbedder`（默认 `BAAI/bge-small-zh-v1.5`）；LLM 是 `OpenAICompatibleLLM`，通过 `LLM_API_BASE`、`LLM_API_KEY`、`LLM_MODEL` 对接任一 Chat Completions 兼容端点。测试模式通过依赖注入提供 `HashingTestEmbedder` 和 `ContextEchoLLM`，绝不把测试后端伪装成生产语义模型。

Chroma collection 使用 cosine 空间，分数为 `1 - distance`。默认 `TOP_K=4`、`MIN_RELEVANCE=0.60`，由环境变量调节；`EVALUATION_THRESHOLDS` 配置离线候选值。当前 0.60 来自独立 32 题保留集的比较，不使用 18 题回归集调节；详见 `docs/EVALUATION_REPORT.md`。生产上线前仍需以审核后的真实问答记录再次验证。

## 6. 安全控制

`SafetyGate` 在检索和 LLM 前识别库存/现货、绕开关键词的“还能下单”、实时或指定日期交期/出库、订单级/批量/经销报价，以及食品保质期、配料和过敏原。供应商/经销商的资质、审核、门槛、区域等未公开规则同样在检索前转人工。命中后直接返回固定人工文案；低相关性同样返回该文案。查询还会优先限定问题中明确出现的 `product_name`，而“微波炉”、材质、洗碗机、认证、食品安全、定制范围等需明确资料支持的属性在目标商品 chunk 没有对应文字时固定转人工。

离线评估通过 `trace_retrieval()` 记录硬安全门决定与原始 Top-K，随后记录最终 answer/handoff；追踪不要求 LLM 暴露推理过程。

LLM 的系统提示只允许使用所给上下文、禁止填补缺失事实，并由编排层追加机器生成来源，避免模型伪造来源字段。通用的“语义相近但证据不足”阈值保护仍未完成，故产品侧规定资料未覆盖的信息一律转人工。
