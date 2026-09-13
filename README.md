# 叁鼎芯供应链商品知识助手（RAG MVP）

这是围绕供应链商品咨询重新设计的第一阶段 RAG 项目。业务场景覆盖采购客户的商品参数、定制、起订量、交期说明与售后，以及供应商入驻和经销商合作咨询。它借鉴了“规格先行、模块分层、按闭环迭代”的工程方法，但没有复制参考项目的业务实现、MCP、BM25、RRF、Rerank、多模态或 Dashboard 代码。

## 已实现范围

`Markdown 导入 → 自研结构/递归切分 → Metadata → Embedding → Chroma → Top-K 检索 → LLM 受控回答 + 来源`

知识范围覆盖商品咨询、定制、起订量、交期说明、售后、供应商入驻与经销商合作。所有涉及库存、实时交期或最终报价的请求，都会在进入 LLM 前固定转人工。没有检索结果或低于相关性阈值时也会转人工；但“语义相近、分数偏高而实际证据不足”的阈值保护仍在完善，详见已知限制。

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
PYTHONPATH=src python -m sanding_rag.cli ask "SDX-ISO-485A 适用什么供电电压？"
```

安装为 editable package 后也可使用 `sanding-rag` 命令。上面的 `PYTHONPATH=src` 写法可直接从检出目录运行，适用于包含空格或同步盘路径的本地环境。

首次使用 `sentence_transformers` 时会下载 `BAAI/bge-small-zh-v1.5`。若仅需离线演示与验收（不是生产语义模型），可使用确定性测试模式：

```bash
python scripts/run_acceptance.py
```

## 命令

```bash
# 摄取任一 Markdown 文件或目录；front matter 必须包含四个业务 Metadata 字段
sanding-rag ingest path/to/markdown-or-directory

# 问答；输出 JSON，其中 sources 为来源列表
sanding-rag ask "问题"

# 本地可重复的 10 题验收，采用测试 Embedding 与测试 LLM，不需要 API Key
python scripts/run_acceptance.py

# 以真实本地中文语义 Embedding 跑同一组十题（模型已下载时可离线运行）
python scripts/run_acceptance.py --semantic

# 基础单元测试（标准库 unittest）
PYTHONPATH=src python -m unittest discover -s tests -v
```

## 测试集与当前结果

`data/test_questions.json` 包含 10 道脱敏代表性题目：3 道商品规格、1 道定制、1 道 MOQ、1 道售后、供应商/经销商各 1 道，以及库存/报价与实时到货各 1 道风险题。

- `python scripts/run_acceptance.py`：Chroma + 离线确定性测试 Embedding + 测试 LLM，**10/10 通过**。
- `python scripts/run_acceptance.py --semantic`：Chroma + `BAAI/bge-small-zh-v1.5`，**10/10 通过**；生成边界仍使用测试 LLM，因为仓库不含也不应包含真实 API Key。
- `PYTHONPATH=src python -m unittest discover -s tests -v`：**6/6 通过**。

完整记录见 [TEST_RESULTS.md](docs/TEST_RESULTS.md)。这些结果验证摄取、检索来源和安全门；不代表真实 LLM 的回答质量基准。

## 已知限制

- 尚未完成“语义近似但证据不足”的阈值保护校准；生产上线前应引入拒答测试集和更严格的证据充分性判定。
- 尚未实现 BM25、RRF、Rerank、MCP、多模态、Dashboard 或复杂评测体系。
- 静态资料不能给出库存、实时交期或最终报价；这些请求固定转人工。
- 当前样例均为模拟或脱敏资料，不能作为真实商品、合作条款或售后承诺。

## 目录

```text
src/sanding_rag/      核心代码（无参考项目业务代码）
data/sample/          3 份脱敏商品资料 + 规则 + FAQ
docs/                 产品、技术规格、模块设计与验收报告
scripts/              可重复的 10 题验收脚本
tests/                轻量单元测试
```

详细约束与边界见 [PRODUCT_SPEC.md](docs/PRODUCT_SPEC.md)、[TECH_SPEC.md](docs/TECH_SPEC.md) 和 [MODULE_NOTES.md](docs/MODULE_NOTES.md)。
