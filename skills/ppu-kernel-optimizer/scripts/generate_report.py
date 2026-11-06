"""Render a Chinese audit report from supplied facts and canonical KDA ledgers."""
import argparse
import csv
import json
from pathlib import Path
from common import cli, read_json


def report(workspace, facts):
    def block(value):
        return "\n```json\n" + json.dumps(value, ensure_ascii=False, indent=2) + "\n```\n"
    lines = ["# PPU 算子性能分析报告\n", "工程结论：" + facts.get("conclusion", "证据不足，尚不能确定动态瓶颈或性能改善。")]
    sections = (("分析对象与目标", "object"), ("设备、软件与版本", "environment"),
                ("Baseline", "baseline"), ("候选瓶颈与证据链", "hypotheses"),
                ("关键对照与反证实验", "experiments"), ("优化实施与失败候选", "optimizations"),
                ("正确性验证", "correctness"), ("深入原理与证明边界", "interpretation"),
                ("尚未解决的问题与下一实验", "next_experiments"), ("复现命令", "commands"),
                ("证据位置", "evidence_paths"))
    for title, key in sections:
        lines.extend(["\n## " + title, block(facts.get(key, "未采集/未执行；无实测结论"))])
    for title, relative in (("工具发现", "runs/analysis-capabilities.json"),
                            ("功能验证与失败边界", "runs/analysis-functional.json"),
                            ("当前分析策略", "runs/analysis-strategy.json")):
        path = workspace / relative
        lines.extend(["\n## " + title, block(read_json(path) if path.is_file() else "未采集")])
    path = workspace / "runs/benchmark.csv"
    lines.append("\n## 正式性能比较\n")
    if path.is_file():
        with path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        lines.append(block(rows))  # retain every required shape and regression
    else:
        lines.append("未测；无延迟或加速比可报告。\n")
    path = workspace / "runs/candidates.jsonl"
    lines.extend(["\n## 候选审计与回滚", block([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.is_file() else "未创建候选")])
    lines.append("\nA/B/C/D 是证据范围分级，不是统计概率。静态产物不证明动态 stall。\n")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="从真实事实文件及既有 KDA 台账生成中文完整审计报告")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--facts", required=True, help="JSON；只填写实际事实，未测字段留空")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(report(a.workspace, read_json(a.facts)), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(cli(main))
