# 叁鼎芯供应链商品知识助手（RAG MVP）

这是围绕商城目录商品咨询重新设计的第一阶段 RAG 项目。它采用独立的 Markdown 知识资料、可替换 Embedding 和本地 Chroma，回答已公开的商品信息与平台合作入口，并返回可核查来源。

v0.1 早期的 SDX 电子产品资料仅用于验证技术链路；它们不是商城业务资料，已从当前知识库、测试集和本地索引中移除。当前版本使用商城目录资料，未被资料覆盖的信息一律转人工，不把推断当作商品事实或平台承诺。

## 当前 RAG 链路

`Markdown 导入 → 自研结构/递归切分 → Metadata → Embedding → Chroma → Top-K 检索 → LLM 受控生成 + 引用 ID 验证 → 用户来源`

第一批知识覆盖 10 个商城商品与 1 份平台合作说明：产地、包装、已公开展示价、已写明规格、起订量、定制或微波炉适用性等。资料只有在目录明确展示时才写入；不能据此回答库存、具体发货日期、订单级/批量/经销报价、食品保质期/配料/过敏原，或其他未公开的材质、认证与售后承诺。

平台合作资料只说明：有供应商入驻入口、有经销商申请入口，以及批量采购可申请经销价与专属采购入口。入驻材料、审批时间和实际批量价没有公开依据，固定转人工。

## 快速运行

在本目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

编辑 `.env`，填写可用的 OpenAI Chat Completions 兼容 LLM 配置；然后运行：

```bash
PYTHONPATH=src python -m sanding_rag.cli ingest data/sample
PYTHONPATH=src python -m sanding_rag.cli ask "景德镇青花龙纹瓷盘的起订量是多少？"
```

安装为 editable package 后也可使用 `sanding-rag` 命令。`PYTHONPATH=src` 写法可直接从检出目录运行，适用于包含空格或同步盘路径的本地环境。

首次使用 `sentence_transformers` 时会下载 `BAAI/bge-small-zh-v1.5`。若仅需离线演示与验收（不是生产语义模型），可使用确定性测试模式：

```bash
python scripts/run_acceptance.py
```

## 命令

```bash
# 摄取 Markdown 文件或目录；front matter 必须包含五个业务 Metadata 字段
sanding-rag ingest path/to/markdown-or-directory

# 问答；输出 JSON，其中 sources 只包含用户可见的目录来源 URL 与文档信息
sanding-rag ask "问题"

# 18 题可重复验收，采用测试 Embedding 与测试 LLM，不需要 API Key
python scripts/run_acceptance.py

# 以真实本地中文语义 Embedding 跑同一组题目（模型已下载时可离线运行）
python scripts/run_acceptance.py --semantic

# 用独立的 32 题保留集比较 MIN_RELEVANCE，并写入逐题追踪 JSON
python scripts/run_offline_evaluation.py

# 运行独立的 27 题检索压力集，对照 Dense、BM25 和 rank-only RRF
python scripts/run_retrieval_stress.py

# 真实 LLM 回答质量评估：仅生成不调用 API 的透明准备报告（写入 docs）
python scripts/run_generation_evaluation.py --prepare-only

# 显式真实评估：仅从本地 .env 读取 LLM_API_BASE、LLM_API_KEY、LLM_MODEL
# 不属于普通单元测试或 CI；原始回答与 trace 默认写入 gitignore 的 data/runtime/，
# 首个 API、网络、认证、超时或提供方响应错误会 fail-fast，以非 0 状态写入
# status=aborted 的本地诊断，不会伪装成 completed 评估。
python scripts/run_generation_evaluation.py

# 完全离线地写入人工复核：只读取上一条命令的本地 trace 与 review JSON，
# 不读取 .env、不初始化 Embedding、不调用 LLM 或网络。
# review JSON 只能填写脚本已选中的 case ID，格式：{"G01":{"decision":"pass","notes":"..."}}
python scripts/apply_generation_manual_review.py \
  --review-file data/runtime/generation_evaluation/manual_review.json

# 基础单元测试（标准库 unittest）
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 测试集与当前结果

`data/test_questions.json` 含 18 道商城目录回归验收题：10 个商品的产地、包装、公开展示价或已写明规格/起订量，平台合作入口，未公开经销价、食品资料缺口、非玲珑瓷的微波炉问题、库存、具体发货日期和订单级报价。当前离线验收与 `BAAI/bge-small-zh-v1.5` 语义验收均为 **18/18 通过**，基础单元测试为 **37/37 通过**。

阈值不使用这 18 题调节，而使用独立的 32 题保留集。实际语义评估后，暂用 `MIN_RELEVANCE=0.60`：Source Hit@1/Hit@3 为 17/17，错误回答率为 0/32，正确转人工率为 15/15，串商品错误为 0。阈值对比和失败用例见 [EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md)，完整逐题记录见 [EVALUATION_TRACES.json](docs/EVALUATION_TRACES.json)。

验收使用真实 Chroma 接口、样例 Markdown 和完整的导入/切分/向量检索/安全门/来源返回链路；离线模式注入 `HashingTestEmbedder` 与 `ContextEchoLLM`，所以它验证闭环、来源和安全策略，不构成真实 LLM 的回答质量基准。

另有独立的 27 题检索压力集，使用真实 BGE 比较 Dense、确定性中文 BM25 和 rank-only RRF：三者均为 Hit@1/Hit@3/MRR 100%、串商品 0，未出现 Hybrid 的明确质量收益，因此 `RETRIEVAL_MODE` 默认继续为 `dense`。完整方法、延迟和局限见 [HYBRID_RETRIEVAL_EXPERIMENT.md](docs/HYBRID_RETRIEVAL_EXPERIMENT.md)。

`data/generation_evaluation_questions.json` 是独立的 26 题最终回答质量集，不修改也不参与 18 题回归、32 题安全或 27 题检索压力集。它同时检查公开商品回答与必须转人工的问题；显式脚本记录最终答案、完整 Top-K `retrieval_sources`、模型验证后的 `returned_sources`、模型名、耗时和逐维规则结果。`product_id` 仅保留在后端匹配与调试 trace，绝不进入模型上下文、最终回答或用户可见来源。模型的无效 JSON 或无效 `used_source_ids` 是可评估的单题 badcase；API、网络、认证、超时或提供方响应故障则会中止整轮运行，绝不会写成 `completed` 或生成质量结论。

一次本地 DeepSeek 预修复运行已真实执行，但原始回答与报告只保存在 gitignore 的本地证据目录，未进入提交。该运行发现 G11 的多商品比较误转人工，以及 G13/G14/G15/G17 的无关来源绑定问题；脱敏摘要见 [GENERATION_EVALUATION_BADCASE_SUMMARY.md](docs/GENERATION_EVALUATION_BADCASE_SUMMARY.md)。本次修复后尚未重新调用真实模型，仓库内报告仍明确为 **not_run**，不含虚构的模型指标或人工结论；见 [GENERATION_EVALUATION_REPORT.md](docs/GENERATION_EVALUATION_REPORT.md)。真实运行完成后，再使用离线人工复核命令在同一 `data/runtime/generation_evaluation/latest/` 中更新 trace/report 与演示门槛。人工复核覆盖所有自动失败题，加上至少 20% “已调用 LLM 且自动通过”的题；九道硬安全门题不计入生成抽样，由独立安全门测试保证。

## 已知限制

- `MIN_RELEVANCE=0.60` 是当前小型保留集上的暂用值，不是长期固定阈值；资料、模型或流量分布变化后需要重跑保留评估。明确属性保护仍是保守词表，而非通用字段级事实验证。
- BM25 与 RRF 仅作为可配置的离线对照实现，未因本次实验改为默认；尚未实现 Rerank、MCP、多模态、Dashboard 或复杂评测体系。
- 真实 LLM 生成评估只能在本地 `.env` 已配置时显式运行；自动规则不是 LLM Judge 或绝对真相。模型必须以结构化 `used_source_ids` 引用当次证据，后端只返回验证过的来源；格式或引用无效会转人工。完成全部失败题和 20% 有模型回答的通过题的人工复核前，不能判定达到独立演示门槛。
- 当前硬规则会把库存、具体/实时交期、订单级/批量/经销报价，以及食品保质期、配料、过敏原转人工；“微波炉”只在命中明确写有该事实的商品资料时回答。
- 公开展示价不是经销价、批量价或订单级最终报价。任何未被当前资料覆盖的信息一律转人工。

## 目录

```text
src/sanding_rag/      核心代码（无参考项目业务代码）
data/sample/          10 份商城商品资料 + 平台合作说明
data/test_questions.json  18 道商城目录验收题
data/holdout_evaluation_questions.json  32 道独立离线保留题
data/retrieval_stress_questions.json  27 道独立检索压力题
data/generation_evaluation_questions.json  26 道独立真实生成质量题
docs/                 产品、技术规格、模块设计与验收报告
scripts/              可重复验收脚本
tests/                轻量单元测试
```

详细约束与边界见 [PRODUCT_SPEC.md](docs/PRODUCT_SPEC.md)、[TECH_SPEC.md](docs/TECH_SPEC.md)、[MODULE_NOTES.md](docs/MODULE_NOTES.md) 和 [EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md)。
