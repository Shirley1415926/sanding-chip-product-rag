# 商城目录资料验收结果

执行日期：2026-09-13

v0.1 早期的 SDX 电子产品资料只用于 RAG 技术链路的初始样例。本次验收已经使用商城目录的 10 个商品与 1 份平台合作说明，并在重新摄取前清空本地 Chroma collection。重新摄取后集合包含 **11 份资料、33 个 chunk**；所有 chunk 均有 `source_url`，没有 SDX 来源。

## 执行结果

| 命令 | 结果 | 说明 |
| --- | --- | --- |
| `.venv/bin/python scripts/run_acceptance.py` | **18 / 18 通过** | Chroma + `HashingTestEmbedder` + `ContextEchoLLM`，可离线重复 |
| `.venv/bin/python scripts/run_acceptance.py --semantic` | **18 / 18 通过** | Chroma + `BAAI/bge-small-zh-v1.5`；生成边界仍使用测试 LLM，不使用或保存任何 API Key |
| `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v` | **6 / 6 通过** | Metadata 校验、切分保留、来源返回与既有安全门基础检查 |
| `PYTHONPATH=src .venv/bin/python -m sanding_rag.cli ingest data/sample` | **11 份资料 / 33 chunks** | 已清空旧 collection 后的实际本地摄取 |

验收使用真实 Markdown 导入、切分、Chroma 接口、Top-K 检索、安全门和来源返回。离线与语义验收的 LLM 使用 `ContextEchoLLM`，因此结果验证的是证据、来源与安全策略，不构成生产 LLM 文案质量或订单系统承诺。

## 18 道测试题

| 题号 | 场景 | 预期 | 实际 |
| --- | --- | --- | --- |
| Q01 | 软木画产地、包装、公开展示价 | 命中软木画目录资料 | 通过，来源为 `marketplace-cork-painting-sanfangqixiang` |
| Q02 | 乌龙茶产区、规格、包装、公开展示价 | 命中乌龙茶目录资料 | 通过，来源为 `marketplace-oolong-tea-gift-box` |
| Q03 | 竹荪产地、袋装规格、整箱包装 | 命中古田竹荪目录资料 | 通过，来源为 `marketplace-gutian-bamboo-fungus` |
| Q04 | 高足杯产地、包装、定制 | 命中高足杯目录资料 | 通过，来源为 `marketplace-jingdezhen-blue-white-high-foot-cup` |
| Q05 | 龙纹瓷盘产地、包装、起订量 | 命中龙纹瓷盘目录资料 | 通过，返回 10 件起订证据 |
| Q06 | 玲珑瓷产地、规格、材质 | 命中玲珑瓷目录资料 | 通过，返回 56 头套装证据 |
| Q07 | 漆器花瓶产地、包装 | 命中漆器目录资料 | 通过，返回单件礼盒证据 |
| Q08 | 火锅底料产地、包装、规格 | 命中火锅底料目录资料 | 通过，返回每袋 200g 证据 |
| Q09 | 沙茶酱产地、规格 | 命中沙茶酱目录资料 | 通过，返回 300g/瓶证据 |
| Q10 | 寿山石摆件产地、包装 | 命中寿山石目录资料 | 通过，返回福州晋安区寿山乡证据 |
| Q11 | 批量采购可申请的入口 | 命中平台合作资料 | 通过，返回经销价申请与专属采购入口证据 |
| Q12 | 乌龙茶经销价 | 转人工 | 通过，`non_public_price`，不把展示价当经销价 |
| Q13 | 竹荪保质期、配料、过敏原 | 转人工 | 通过，`unpublished_food_facts` |
| Q14 | 龙纹瓷盘能否用于微波炉 | 转人工 | 通过，`explicit_fact_not_found`；不串用玲珑瓷资料 |
| Q15 | 沙茶酱当前库存 | 转人工 | 通过，`inventory` |
| Q16 | 竹荪具体发货日期 | 转人工 | 通过，`real_time_lead_time` |
| Q17 | 寿山石摆件订单级报价 | 转人工 | 通过，`non_public_price` |
| Q18 | 玲珑瓷是否适用微波炉 | 命中玲珑瓷目录资料 | 通过，返回“微波炉适用”证据 |

## 仍缺失的业务资料

当前资料没有食品保质期、配料、过敏原或营养信息；没有库存、发货日期、承诺交期、订单级报价、批量价或经销价；没有除已写明“支持定制”外的定制范围；也没有供应商入驻材料、审批时间、经销商门槛、售后承诺、认证或其他商品属性。这些问题均不能通过当前商城目录资料回答，应转人工。

通用的“语义近似但证据不足”阈值保护还在校准中；上线前应以审核后的真实问答补充拒答用例，并将每个新事实与可访问的单品来源 URL 一同纳入资料。
