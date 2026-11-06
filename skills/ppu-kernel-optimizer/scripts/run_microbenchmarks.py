"""Plan or run a bounded, serial PPU experiment matrix in worker processes."""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from common import cli, positive, read_json, run, write_json


def allocation_bytes(config):
    n = config["n"]
    # Include device buffers and torch correctness temporaries, conservatively.
    return n * n * 32 if config["mode"] == "gemm" else n * (config["stride"] + 16) * 8


def main():
    p = argparse.ArgumentParser(description="默认仅生成实验计划；--execute 才运行目标设备，不连接远端")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--modes", nargs="+", choices=("read", "write", "copy", "fma", "reuse", "dependency", "gemm"), default=["copy"])
    p.add_argument("--sizes", type=int, nargs="+", default=[4096, 65536])
    p.add_argument("--strides", type=int, nargs="+", default=[1])
    p.add_argument("--blocks", type=int, nargs="+", default=[256])
    p.add_argument("--warps", type=int, nargs="+", default=[4])
    p.add_argument("--stages", type=int, nargs="+", default=[1])
    p.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16", help="矩阵输入 dtype；向量算术为 FP32")
    p.add_argument("--iters", type=int, default=32)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--max-configs", type=int, default=24)
    p.add_argument("--max-bytes", type=int, default=128 * 1024 ** 2)
    p.add_argument("--timeout", type=positive, default=60)
    p.add_argument("--total-seconds", type=positive, default=180)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--worker-config", type=Path, help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.worker_config:
        from microbench_adapter import build
        from benchmark_kernel import measure
        from collect_compiler_artifacts import collect
        config = read_json(a.worker_config)
        adapter = build(config)
        result = measure(adapter, repeat=a.repeat)
        desc = adapter.describe()
        # Resource metadata describes this exact timed specialization, not a cache guess.
        provenance = {"source_sha256": __import__("common").digest(Path(__file__).with_name("microbench_adapter.py")),
                      "signature": desc["dtype"], "shape": desc["shape"], "dtype": desc["dtype"],
                      "stride": desc["stride"], "grid": ([(config["n"] + config["block"] - 1) // config["block"]] if config["mode"] != "gemm" else [(config["n"] + config["block"] - 1) // config["block"]] * 2),
                      "config": config, "target": str(desc["versions"]),
                      "versions": desc["versions"], "candidate": a.worker_config.stem}
        collect(adapter.last, a.output.with_suffix(".compiler"), provenance)
        n, mode = config["n"], config["mode"]
        byte_count = (n * 4 + ((n + config["block"] - 1) // config["block"]) * 4 if mode == "read" else
                      n * 4 if mode == "write" else n * 8 if mode == "copy" else None)
        result["logical_bytes"] = byte_count
        result["effective_GB_s"] = byte_count / result["statistics"]["median_us"] / 1000 if byte_count else None
        result["modeled_flops"] = 2 * n ** 3 if mode == "gemm" else 2 * n * config["iters"] if mode in ("fma", "reuse") else None
        write_json(a.output, result)
        return
    if not (1 <= a.max_configs <= 256 and a.max_bytes > 0 and 1 <= a.iters <= 1024 and 1 <= a.repeat <= 1000):
        raise ValueError("非法预算：configs 1..256，iters 1..1024，repeat 1..1000，bytes > 0")
    axes = [a.modes, a.sizes, a.strides, a.blocks, a.warps, a.stages]
    if any(v <= 0 for axis in axes[1:] for v in axis):
        raise ValueError("配置参数必须为正数")
    count = __import__("math").prod(len(axis) for axis in axes)
    if count > a.max_configs:
        raise ValueError(f"矩阵 {count} 超过 max-configs={a.max_configs}；缩小扫描")
    a.output.mkdir(parents=True, exist_ok=False)
    configs = [dict(zip(("mode", "n", "stride", "block", "warps", "stages"), values),
                    dtype=a.dtype, iters=a.iters) for values in itertools.product(*axes)]
    plan = {"status": "planned", "configs": configs, "limits": {"max_bytes": a.max_bytes, "total_seconds": a.total_seconds},
            "unsupported": {"explicit_shared_bytes": "Triton 自动分配；通过编译资源记录观察，不伪造控制接口",
                            "fp8_int8": "当前未实现；需先核验后端路径"}, "results": []}
    started = time.monotonic()
    for i, config in enumerate(configs):
        config_file = a.output / f"{i:03d}.config.json"
        write_json(config_file, config)
        row = {"config": config, "status": "planned", "estimated_allocation_bytes": allocation_bytes(config)}
        if allocation_bytes(config) > a.max_bytes:
            row.update(status="skipped", reason="分配预算不足")
        elif a.execute:
            remaining = a.total_seconds - (time.monotonic() - started)
            if remaining <= 0:
                row.update(status="skipped", reason="总时长预算耗尽")
            else:
                command = [sys.executable, str(Path(__file__).resolve()), "--output", str(a.output / f"{i:03d}.result.json"),
                           "--worker-config", str(config_file), "--repeat", str(a.repeat)]
                process = run(command, min(a.timeout, remaining))
                row.update(status="measured" if process["exit_code"] == 0 else "failed", process=process)
        plan["results"].append(row)
        write_json(a.output / "plan.json", plan)
    plan["status"] = "executed" if a.execute else "planned"
    write_json(a.output / "plan.json", plan)
    return 2 if any(row["status"] == "failed" for row in plan["results"]) else 0


if __name__ == "__main__":
    raise SystemExit(cli(main))
