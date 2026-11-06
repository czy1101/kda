"""Small standard-library helpers; no remote execution interface."""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False,
                               default=str) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run(argv, timeout=15):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                           encoding="utf-8", errors="replace")
        return {"command": argv, "exit_code": p.returncode,
                "stdout": p.stdout, "stderr": p.stderr, "tested_at": now()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"command": argv, "exit_code": None, "error": str(e), "tested_at": now()}


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("必须是有限正数")
    return result


def quantile(values, fraction):
    values = sorted(values)
    p = (len(values) - 1) * fraction
    lo = int(p)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (p - lo)


def summarize(samples):
    if not samples or any(not math.isfinite(v) or v <= 0 for v in samples):
        raise ValueError("样本必须为非空、有限、正数；零值可能低于计时分辨率")
    mean = statistics.mean(samples)
    return {"count": len(samples), "median_us": statistics.median(samples),
            "p10_us": quantile(samples, .1), "p90_us": quantile(samples, .9),
            "min_us": min(samples), "max_us": max(samples),
            "cv": statistics.pstdev(samples) / mean}


def kernelpilot():
    import sys
    root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(root / "scripts"))
    import kernelpilot as kp
    return kp


def cli(main):
    try:
        return main() or 0
    except (ValueError, OSError, KeyError, TypeError, ImportError, RuntimeError) as e:
        import sys
        print(f"错误：{e}", file=sys.stderr)
        return 2
