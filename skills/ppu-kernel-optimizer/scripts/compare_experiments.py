"""Strict diagnostic comparison, not candidate promotion."""
import argparse
from common import cli, read_json, write_json


def compare(baseline, candidate, max_cv=.05):
    for result in (baseline, candidate):
        if result.get("status") != "measured" or result.get("correctness") != "pass":
            raise ValueError("仅比较已测且正确的结果")
        if result.get("config_changed") or not result["protocol"].get("config_fixed"):
            raise ValueError("autotune 配置未固定或测量期间变化")
    keys = ("shape", "stride", "dtype", "device", "environment_fingerprint", "scope", "cache_protocol", "execution_kind")
    if any(baseline["protocol"][k] != candidate["protocol"][k] for k in keys):
        raise ValueError("运行条件不同，不可比较")
    if any(baseline[k] != candidate[k] for k in ("timer", "batch", "warmup", "repeat")):
        raise ValueError("计时协议不同，不可比较")
    b, c = baseline["statistics"], candidate["statistics"]
    stable = max(b["cv"], c["cv"]) <= max_cv
    separated = c["p90_us"] < b["p10_us"]
    return {"comparison": "diagnostic_only", "speedup": b["median_us"] / c["median_us"],
            "stable": stable, "separated_p10_p90": separated,
            "conclusion": "支持延迟改善，仍需正式全矩阵验证" if stable and separated else "证据不足，可能波动或退化",
            "protocol_baseline": baseline["protocol"], "protocol_candidate": candidate["protocol"],
            "max_cv": max_cv, "statistical_probability": None}


def main():
    p = argparse.ArgumentParser(description="同条件诊断计时比较，不改变正式晋升规则")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-cv", type=float, default=.05)
    a = p.parse_args()
    if not 0 <= a.max_cv < 1:
        raise ValueError("max-cv 必须在 [0,1)")
    write_json(a.output, compare(read_json(a.baseline), read_json(a.candidate), a.max_cv))


if __name__ == "__main__":
    raise SystemExit(cli(main))
