# 技术规格：叁鼎芯供应链商品知识助手

版本：0.1（第一阶段 MVP）
更新日期：2026-09-13

## 1. 设计原则与参考边界

本工程采用“规格先行、核心类型明确、摄取与查询分离、外部模型可替换”的思路；参考了 `jerry-ai-dev/MODULAR-RAG-MCP-SERVER` 的 `clean-start` 工程骨架及 dev 分支从 A（工程/配置）、B（可替换基础设施）、C（摄取）、D（检索）逐步交付的节奏。

实现从零编写，仅保留本项目需要的轻量目录与接口；不复制参考仓库的代码、提示词、配置、业务逻辑或 Skill 文件。

## 2. 范围

### 本期包含

1. Markdown Loader：解析 YAML 风格 front matter 与正文，校验四个必填 Metadata。
2. `SemanticRecursiveSplitter`：先按标题、段落、列表与句子寻找语义边界；超长片段再递归降级为字符窗口，保留少量 overlap。
3. Embedding 抽象：生产使用 Sentence Transformers 中文模型；测试使用确定性向量，避免测试依赖网络。
4. `ChromaStore`：使用本地持久化 Chroma collection，按 `source` 幂等替换、按向量 Top-K 查询。
5. 问答编排：硬安全门 → Top-K 检索与阈值门 → 上下文受限 LLM → 统一来源输出。
6. CLI、样例资料和十题验收脚本。

### 明确不做

BM25、RRF、Rerank、多模态、Dashboard、MCP、异步队列、复杂观察体系、Ragas/Golden set 回归框架、正式权限体系与交易系统集成。

## 3. 数据契约

每个 Markdown 文档的 front matter 需有：

```yaml
---
source: sample/product-sdx-iso-485a.md
product_id: SDX-ISO-485A
document_type: product
updated_at: 2026-09-13
---
```

`product_id` 在规则/FAQ 中可为 `GENERAL`。入库 chunk 额外写入 `chunk_id`、`chunk_index`、`heading` 和字符偏移。`updated_at` 被按字符串保存，便于 Chroma filter 与可追溯展示。

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
  -> Context-bound LLM
  -> {answer, sources, handoff_required}
```

## 5. 接口与运行配置

生产 Embedding 是 `SentenceTransformerEmbedder`（默认 `BAAI/bge-small-zh-v1.5`）；LLM 是 `OpenAICompatibleLLM`，通过 `LLM_API_BASE`、`LLM_API_KEY`、`LLM_MODEL` 对接任一 Chat Completions 兼容端点。测试模式通过依赖注入提供 `HashingTestEmbedder` 和 `ContextEchoLLM`，绝不把测试后端伪装成生产语义模型。

Chroma collection 使用 cosine 空间，分数为 `1 - distance`。默认 `TOP_K=4`、`MIN_RELEVANCE=0.45`，由环境变量调节。生产上线前需以真实脱敏问答记录调整阈值。

## 6. 安全控制

`SafetyGate` 在检索和 LLM 前识别库存/现货、实时或指定日期交期、最终报价和具体价格。命中后直接返回固定人工文案；低相关性同样返回该文案。LLM 的系统提示只允许使用所给上下文、禁止填补缺失事实，并由编排层追加机器生成来源，避免模型伪造来源字段。
