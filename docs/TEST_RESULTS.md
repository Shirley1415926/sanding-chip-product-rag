# 第一阶段十题验收结果

执行日期：2026-09-13
执行命令：`.venv/bin/python scripts/run_acceptance.py`
结果：**10 / 10 通过**。

验收使用真实 Chroma 持久化接口、样例 Markdown 和完整的导入/切分/向量检索/安全门/来源返回链路；为了使离线验收可重复，不依赖 API Key，Embedding 与 LLM 边界分别注入 `HashingTestEmbedder`、`ContextEchoLLM`。因此它验证的是闭环、来源和安全策略，不构成生产语义模型或真实 LLM 的质量基准。

另执行 `.venv/bin/python scripts/run_acceptance.py --semantic`：使用已下载的 `BAAI/bge-small-zh-v1.5` 与真实 Chroma 接口，**10 / 10 通过**；生成边界仍使用测试 LLM，因为本次环境未配置用户的 LLM API Key。三条独立的生产语义检索冒烟测试也分别命中 ISO-485A 商品资料（0.6781）、CONN-16P 商品资料（0.5936）和供应商入驻规则资料（0.6663），且四个必填 Metadata 均存在。

| 题号 | 问题类型 | 结果 | 关键验证 |
| --- | --- | --- | --- |
| Q01 | SDX-ISO-485A 逻辑侧供电范围 | 通过 | 检索到 `product-sdx-iso-485a.md`，返回来源 |
| Q02 | SDX-DC5A-24V 输入与输出 | 通过 | 检索到 `product-sdx-dc5a-24v.md`，返回来源 |
| Q03 | SDX-CONN-16P 间距与线规 | 通过 | 检索到 `product-sdx-conn-16p.md`，返回来源 |
| Q04 | 定制电源资料 | 通过 | 检索到 `service-and-cooperation-policy.md`，返回来源 |
| Q05 | SDX-CONN-16P MOQ | 通过 | 检索到对应商品资料，返回来源 |
| Q06 | 功能异常的售后材料 | 通过 | 检索到规则资料，返回来源 |
| Q07 | 供应商入驻材料 | 通过 | 检索到规则资料，返回来源 |
| Q08 | 经销商合作申请信息 | 通过 | 检索到规则资料，返回来源 |
| Q09 | 当前库存与当天发货 | 通过 | 在检索前因 `inventory` 固定转人工，无来源伪造 |
| Q10 | 最终报价与次日到货 | 通过 | 在检索前因 `real_time_lead_time` 固定转人工，无来源伪造 |

附加基础检查：`.venv/bin/python -m unittest discover -s tests -v`，**6 / 6 通过**（Metadata 必填校验、切分 Metadata 保留、库存/报价安全门、普通规格问题与无依据固定转人工）。
