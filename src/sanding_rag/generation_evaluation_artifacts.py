"""Pure local artifact rendering for generation-evaluation runs.

This module deliberately depends only on evaluation data.  In particular, it
does not import an embedding provider, Chroma, an LLM client, environment
configuration or any network-capable module, so manual-review writes can remain
fully offline.
"""

from __future__ import annotations

from typing import Any

from .generation_evaluation import DIMENSIONS


def demo_gate(summary: dict[str, Any]) -> str:
    """Return the conservative demo decision for one completed evaluation run."""
    manual = summary["manual_review"]
    if (
        summary["automatic_overall"]["failed"] == 0
        and manual["pending_case_count"] == 0
        and manual["reviewer_fail_count"] == 0
    ):
        return "已达到限定商城目录、固定题集和已完成复核条件下的独立演示门槛；不代表正式商用就绪。"
    return "尚未达到独立演示门槛：需要先消除自动失败，并完成全部规定的人工复核。"


def render_report(report: dict[str, Any]) -> str:
    """Render a report without inferring metrics that an aborted run never produced."""
    lines = ["# 真实 LLM 回答质量评估", "", f"运行日期：{report['run_at']}", ""]
    if report["status"] == "aborted":
        diagnostic = report["diagnostic"]
        lines.extend(
            [
                "## 当前状态",
                "",
                "本次真实 LLM 评估在运行时中止，未生成质量指标、失败样例、人工复核结论或独立演示结论。",
                f"中止用例：`{diagnostic['case_id']}`（第 {diagnostic['case_index']} 题）；模型：`{report['model']}`。",
                f"安全诊断：{diagnostic['message']}",
                "",
                "请修复本地 API/网络/模型配置后重新运行完整评估；不要将此中止诊断当作已完成评估。",
            ]
        )
        return "\n".join(lines) + "\n"

    if report["status"] != "completed":
        lines.extend(
            [
                "## 当前状态",
                "",
                "本次未调用真实 LLM，因此没有生成质量指标或人工复核结论。",
                f"原因：{report['reason']}",
                f"题目数量：{report['dataset_case_count']}；模型版本：未配置（未读取 `.env`）。",
                "",
                "评估脚本已就绪。仅在本地 `.env` 同时提供 `LLM_API_BASE`、`LLM_API_KEY`、`LLM_MODEL` 后，显式执行下列命令才会调用模型：",
                "",
                "```bash",
                "./.venv/bin/python scripts/run_generation_evaluation.py",
                "```",
                "",
                "这不会使用环境变量回退值，也不会在普通单元测试或 CI 中调用 API。真实运行的原始回答和 trace 默认写入 gitignore 的 `data/runtime/generation_evaluation/latest/`，不会自动进入提交。",
                "",
                "## 自动结果",
                "",
                "未运行，因此不存在 Groundedness、相关性、完整性、来源正确性或安全合规的自动指标。",
                "",
                "## 人工复核结果",
                "",
                "未运行，因此没有可供人工复核的真实模型回答；不能把准备清单当作人工复核已完成。真实运行时，所有自动失败题和至少 20%“已调用 LLM 且自动通过”题须人工复核；硬安全门题不计入生成回答抽样。",
                "",
                "## 已知边界",
                "",
                "未测试英文询盘、真实用户表达、提示注入、长文档或大规模语料。当前不能据此判断是否达到独立演示门槛，更不能宣称可正式商用。",
            ]
        )
        return "\n".join(lines) + "\n"

    summary = report["summary"]
    parent_run = report.get("parent_run")
    if isinstance(parent_run, dict):
        lines.extend(
            [
                "## 离线校准来源",
                "",
                "本报告是对已完成父运行的离线评估口径校准，不会覆盖父 trace，也没有重新调用模型。",
                f"父运行：`{parent_run.get('trace_path')}`；父运行日期：{parent_run.get('run_at')}；父 trace SHA-256：`{parent_run.get('trace_sha256')}`。",
                "",
            ]
        )
    lines.extend(
        [
            "## 运行范围",
            "",
            f"模型：`{report['model']}`；题目数：{report['dataset_case_count']}；检索模式：`{report['retrieval_mode']}`；MIN_RELEVANCE：`{report['min_relevance']}`。",
            "",
            "自动判定使用确定性规则，不调用 LLM Judge：只读取问题、Top-K 检索证据、最终回答和预期要点。它不是绝对真相，特别是无法完整识别自然语言的隐含事实或高质量同义改写。",
            "",
            "## 自动结果（逐维）",
            "",
            "| 维度 | 通过 | 失败 | 不适用 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    labels = {
        "groundedness": "Groundedness / 忠实度",
        "answer_relevance": "Answer relevance / 相关性",
        "completeness": "Completeness / 完整性",
        "citation_correctness": "Citation correctness / 来源正确性",
        "safety_compliance": "Safety compliance / 安全合规",
    }
    for dimension in DIMENSIONS:
        metric = summary["automatic_dimensions"][dimension]
        lines.append(f"| {labels[dimension]} | {metric['passed']} | {metric['failed']} | {metric['not_applicable']} |")
    overall = summary["automatic_overall"]
    lines.extend(
        [
            "",
            f"自动整体结果（仅用于定位，不替代逐维结果）：通过 {overall['passed']}，失败 {overall['failed']}。",
            "",
            "## 人工复核结果",
            "",
            f"规则要求复核全部自动失败题，以及“已调用 LLM 且自动通过”题中的确定性 20% 抽样；硬安全门题不进入生成回答抽样。需要复核 {summary['manual_review']['required_case_count']} 题；已完成 {summary['manual_review']['completed_case_count']} 题；待人工复核 {summary['manual_review']['pending_case_count']} 题；人工判定通过 {summary['manual_review']['reviewer_pass_count']}，失败 {summary['manual_review']['reviewer_fail_count']}。",
            "",
        ]
    )
    failures = report["failure_examples"]
    lines.extend(["## 自动失败样例", ""])
    if not failures:
        lines.append("本次自动规则没有发现失败；仍需完成规定的人工抽样复核。")
    else:
        for example in failures[:5]:
            reason_text = "；".join(
                f"{dimension}: {', '.join(reasons)}" for dimension, reasons in example["reasons"].items()
            )
            lines.append(f"- `{example['case_id']}`：{example['query']}。{reason_text}")
    lines.extend(
        [
            "",
            "## 独立演示门槛",
            "",
            report["demo_gate"],
            "",
            "## 当前不能证明的边界",
            "",
            "本评估未测试英文询盘、真实用户表达、提示注入、长文档、大规模语料、并发、真实库存/履约/报价系统，以及长期资料更新后的表现。即使完成门槛，也仅代表限定目录与测试题下可做独立演示，不代表正式商用就绪。",
        ]
    )
    return "\n".join(lines) + "\n"
