"""Adapter-based diagnostic timing. Never replaces a task's official harness."""
import argparse
import importlib.util
import sys
import time
from pathlib import Path
from common import cli, digest, now, positive, run, summarize, write_json


def load_adapter(path):
    spec = importlib.util.spec_from_file_location("ppu_task_adapter", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(adapter, warmup=10, repeat=50, timer="event", batch=1):
    if not 0 <= warmup <= 10000 or not 1 <= repeat <= 10000 or not 1 <= batch <= 10000:
        raise ValueError("warmup >= 0，repeat/batch 必须在 1..10000")
    for name in ("prepare", "check", "launch", "synchronize", "describe"):
        if not callable(getattr(adapter, name, None)):
            raise ValueError(f"adapter 缺少 {name}()")
    prepare_start = time.perf_counter()
    adapter.prepare()  # allocation, JIT and autotune may happen here
    adapter.check()    # correctness must pass before timing
    adapter.synchronize()
    before = adapter.describe()
    required = ("shape", "stride", "dtype", "device", "environment_fingerprint",
                "config", "config_fixed", "scope", "cache_protocol", "execution_kind")
    if any(k not in before for k in required):
        raise ValueError("describe() 必须包含 " + ", ".join(required))
    for _ in range(warmup):
        adapter.launch()
    adapter.synchronize()
    preparation_s = time.perf_counter() - prepare_start
    samples = []
    for _ in range(repeat):
        adapter.synchronize()
        if timer == "event":
            # Adapter owns backend-specific Event validation and stream semantics.
            start, end = adapter.events()
            start.record()
        else:
            start = time.perf_counter_ns()
        for _ in range(batch):
            adapter.launch()
        if timer == "event":
            end.record()
            end.synchronize()
            elapsed_us = start.elapsed_time(end) * 1000
        else:
            adapter.synchronize()
            elapsed_us = (time.perf_counter_ns() - start) / 1000
        samples.append(elapsed_us / batch)
    after = adapter.describe()
    return {"version": 1, "generated_at": now(), "status": "measured",
            "correctness": "pass", "protocol": before, "protocol_after": after,
            "config_changed": before != after, "timer": timer, "batch": batch,
            "warmup": warmup, "repeat": repeat, "prepare_and_warmup_s": preparation_s,
            "samples_us": samples, "statistics": summarize(samples),
            "includes_host_launch": True if timer == "host" else None,
            "host_submission_gaps_possible": True, "official_benchmark": False,
            "limitations": ["辅助诊断计时；正式晋升以任务 benchmark 为准。",
                            "Event 区间需与 launch stream 一致；冷态须由 adapter 验证。"]}


def main():
    p = argparse.ArgumentParser(description="运行已授权的本地 adapter；远端须置于契约 benchmark 命令中")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=50)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--timer", choices=("event", "host"), default="event")
    p.add_argument("--timeout", type=positive, default=120)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if not a.worker:
        result = run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"], a.timeout)
        write_json(str(a.output) + ".process.json", result)
        if result["exit_code"] != 0:
            raise ValueError("计时子进程失败/超时，见 .process.json；未保证已提交设备任务立即中断")
        return
    result = measure(load_adapter(a.adapter), a.warmup, a.repeat, a.timer, a.batch)
    result["adapter_sha256"] = digest(a.adapter)
    write_json(a.output, result)


if __name__ == "__main__":
    raise SystemExit(cli(main))
