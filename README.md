# 叁鼎芯供应链商品知识助手（RAG MVP）

这是围绕商城目录商品咨询重新设计的第一阶段 RAG 项目。它采用独立的 Markdown 知识资料、可替换 Embedding 和本地 Chroma，回答已公开的商品信息与平台合作入口，并返回可核查来源。

v0.1 早期的 SDX 电子产品资料仅用于验证技术链路；它们不是商城业务资料，已从当前知识库、测试集和本地索引中移除。当前版本使用商城目录资料，未被资料覆盖的信息一律转人工，不把推断当作商品事实或平台承诺。

## 当前 RAG 链路

`Markdown 导入 → 自研结构/递归切分 → Metadata → Embedding → Chroma → Top-K 检索 → LLM 受控回答 + 来源`

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

# 问答；输出 JSON，其中 sources 包含目录来源 URL
sanding-rag ask "问题"

# 18 题可重复验收，采用测试 Embedding 与测试 LLM，不需要 API Key
python scripts/run_acceptance.py

# 以真实本地中文语义 Embedding 跑同一组题目（模型已下载时可离线运行）
python scripts/run_acceptance.py --semantic

# 基础单元测试（标准库 unittest）
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 测试集与当前结果

`data/test_questions.json` 含 18 道商城目录验收题：10 个商品的产地、包装、公开展示价或已写明规格/起订量，平台合作入口，未公开经销价、食品资料缺口、非玲珑瓷的微波炉问题、库存、具体发货日期和订单级报价。当前离线验收与 `BAAI/bge-small-zh-v1.5` 语义验收均为 **18/18 通过**，基础单元测试为 **6/6 通过**；详细记录见 [TEST_RESULTS.md](docs/TEST_RESULTS.md)。

验收使用真实 Chroma 接口、样例 Markdown 和完整的导入/切分/向量检索/安全门/来源返回链路；离线模式注入 `HashingTestEmbedder` 与 `ContextEchoLLM`，所以它验证闭环、来源和安全策略，不构成真实 LLM 的回答质量基准。

## 已知限制

- 尚未完成“语义近似但证据不足”的通用阈值保护校准；生产上线前应补充拒答测试集和更严格的证据充分性判定。
- 尚未实现 BM25、RRF、Rerank、MCP、多模态、Dashboard 或复杂评测体系。
- 当前硬规则会把库存、具体/实时交期、订单级/批量/经销报价，以及食品保质期、配料、过敏原转人工；“微波炉”只在命中明确写有该事实的商品资料时回答。
- 公开展示价不是经销价、批量价或订单级最终报价。任何未被当前资料覆盖的信息一律转人工。

## 目录

```text
src/sanding_rag/      核心代码（无参考项目业务代码）
data/sample/          10 份商城商品资料 + 平台合作说明
data/test_questions.json  18 道商城目录验收题
docs/                 产品、技术规格、模块设计与验收报告
scripts/              可重复验收脚本
tests/                轻量单元测试
```

详细约束与边界见 [PRODUCT_SPEC.md](docs/PRODUCT_SPEC.md)、[TECH_SPEC.md](docs/TECH_SPEC.md) 和 [MODULE_NOTES.md](docs/MODULE_NOTES.md)。
