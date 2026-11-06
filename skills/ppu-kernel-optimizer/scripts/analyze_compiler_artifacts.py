"""Attributable static findings and lossless text diffs, not dynamic counters."""
import argparse
import ast
import difflib
import re
from pathlib import Path
from common import cli, digest, read_json, write_json

PATTERNS = {"layout_conversion": r"convert_layout|layout_conversion",
            "memory": r"\b(load|store)\b|vmem\.", "dot_matrix": r"\bdot\b|mma|matrix_mad",
            "shared": r"shared|\btsm\b|local_alloc", "wait_sync": r"wait|barrier|sync",
            "loop_branch": r"\b(for|while|br|cond_br)\b", "spill_hint": r"spill|scratch"}


def analyze(manifest_path):
    path = Path(manifest_path)
    manifest = read_json(path)
    result = {"provenance": manifest["provenance"], "resources": manifest["resources"],
              "missing": manifest["missing"], "static_features": {},
              "limitation": "文本出现次数不是动态计数；不能据此确定 stall/occupancy/重叠。"}
    for key, info in manifest["artifacts"].items():
        artifact = (path.parent / info["path"]).resolve()
        artifact.relative_to(path.parent.resolve())
        if digest(artifact) != info["sha256"]:
            raise ValueError("产物摘要不匹配：" + str(artifact))
        if info["binary"]:
            continue
        text = artifact.read_text(encoding="utf-8")
        result["static_features"][key] = {name: len(re.findall(pattern, text, re.I))
                                           for name, pattern in PATTERNS.items()}
    return result


def source_analysis(path):
    text = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(text)
    calls = [{"line": node.lineno, "call": ast.get_source_segment(text, node.func)}
             for node in ast.walk(tree) if isinstance(node, ast.Call)]
    return {"source": str(path), "sha256": digest(path), "calls": calls,
            "analysis_level": 4, "latency_us": None,
            "hypotheses": [{"grade": "D", "claim": "待验证数据流、布局与 launch 开销",
                            "status": "planned", "refutation": "固定输入和配置，对比资源/规模扫描与正式计时"}],
            "next_experiment": "在授权目标设备先跑原有正确性和 baseline，再选择一个竞争假设的对照实验"}


def main():
    p = argparse.ArgumentParser(description="校验归属并静态分析，或无设备源码分析")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path)
    group.add_argument("--source", type=Path)
    p.add_argument("--compare", type=Path, help="另一个 manifest，生成完整逐 key diff")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = source_analysis(a.source) if a.source else analyze(a.manifest)
    if a.compare:
        if not a.manifest:
            raise ValueError("--compare 要求 --manifest")
        analyze(a.compare)
        left, right = read_json(a.manifest), read_json(a.compare)
        diffs = {}
        for key in sorted(set(left["artifacts"]) | set(right["artifacts"])):
            texts = []
            for root, manifest in ((a.manifest.parent, left), (a.compare.parent, right)):
                info = manifest["artifacts"].get(key)
                texts.append((root / info["path"]).read_text(encoding="utf-8").splitlines(True)
                             if info and not info["binary"] else [])
            diffs[key] = "".join(difflib.unified_diff(*texts, fromfile="baseline/" + key,
                                                   tofile="candidate/" + key))
        result["diff"] = diffs
    write_json(a.output, result)


if __name__ == "__main__":
    raise SystemExit(cli(main))
