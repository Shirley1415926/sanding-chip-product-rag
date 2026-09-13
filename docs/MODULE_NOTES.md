# 模块设计说明

## 1. Markdown 导入与 Metadata 校验

解决什么：让可审核的商品/规则/FAQ 资料以统一格式进入知识库，并能在回答后追溯。
输入 → 输出：带 front matter 的 `.md` 文件 → `SourceDocument(text, metadata)`。
为什么：第一阶段避免 PDF/OCR 的不确定性；在入口强制 `source`、`product_id`、`document_type`、`updated_at`，比后置补齐更可审计。

## 2. 自研语义/递归切分

解决什么：避免把型号参数、MOQ 与限制语句拆散，造成检索命中却缺少条件。
输入 → 输出：一个 `SourceDocument` → 一组带标题路径与字符位置的 `Chunk`。
为什么：优先按 Markdown 标题、段落和中英文句末符切分；只有单元仍超长才降级切字符窗口。该实现不依赖 LangChain，业务规则可直接测试和调整。

## 3. Embedding

解决什么：将中文自然语言问题与资料片段映射至可比较的向量空间。
输入 → 输出：文本列表 → 同维 `list[list[float]]`。
为什么：生产默认使用本地 Sentence Transformers 中文模型以获得语义检索；通过协议隔离模型依赖，验收时可注入确定性测试向量而不请求外网。

## 4. Chroma 向量库

解决什么：持久化 chunk、Metadata 和 embedding，并完成 Top-K 近邻查找。
输入 → 输出：`Chunk + vector` 批量 upsert / `query vector + k` → `RetrievedChunk`。
为什么：Chroma 本地持久化适合 MVP；同一 `source` 先删后写，使再次导入不会积累旧版本 chunk。

## 5. 检索与安全门

解决什么：区分“有证据可回答”与“必须由人工确认”的请求。
输入 → 输出：问题 → 风险判定或按分数排序的 Top-K 证据。
为什么：库存、实时交期与最终报价不能从静态知识库推断；阈值门能在语义相近但没有依据时阻断幻觉。

## 6. 上下文受限 LLM 回答与来源

解决什么：把多个片段组织成简洁回答，同时保留可核验出处。
输入 → 输出：问题 + Top-K context → `AnswerPayload(answer, sources, handoff_required)`。
为什么：LLM 只接收编号后的检索片段和禁止补全指令；来源由程序从真实检索结果生成，不交由模型臆造。

## 7. CLI、样例与十题验收

解决什么：让非 Web 环境也可重复跑通闭环并检查业务边界。
输入 → 输出：目录/问题/十题 fixture → JSON 结果与退出码。
为什么：先用小而可复现的命令行闭环验证资料结构与安全策略，复杂测试体系留到下一阶段。
