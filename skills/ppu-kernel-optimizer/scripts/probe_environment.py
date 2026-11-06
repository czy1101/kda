"""Read-only local/executor probe with isolated optional module imports."""
import argparse
import json
import os
import platform
import shutil
import sys
from common import cli, now, positive, run, write_json

MODULE_PROBE = r'''
import importlib, importlib.metadata, json, sys
m = importlib.import_module(sys.argv[1])
r = {"path": getattr(m, "__file__", None), "version": str(getattr(m,"__version__", "unknown"))}
if sys.argv[1] == "torch":
    import torch
    r["cuda_compat_available"] = torch.cuda.is_available()
    r["devices"] = []
    if r["cuda_compat_available"]:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            r["devices"].append({"index":i, "name":p.name,"properties":str(p)})
    r["event_interface_present"] = hasattr(torch.cuda, "Event")
    r["profiler_interface_present"] = hasattr(torch,"profiler")
if sys.argv[1] == "triton":
    try:
        from triton.runtime import driver
        r["target"] = str(driver.active.get_current_target())
    except Exception as e:
        r["target_error"] = str(e)
    for name in ("flagtree", "triton-for-sail"):
        try: r[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: r[name] = None
print(json.dumps(r))
'''


def probe(timeout=15):
    tools = {}
    for name in ("ppu-smi", "acu", "asys", "hgcc", "hgobjdump", "hgbat",
                 "llvm-dis", "llvm-objdump"):
        path = shutil.which(name)
        record = {"path": path, "state": "not_installed" if not path else "untested"}
        if path:
            flag = "--help" if name in ("hgcc", "hgobjdump", "hgbat") else "--version"
            if name == "ppu-smi":
                flag = "-L"
            record["probe"] = run([path, flag], timeout)
            record["state"] = "runnable_function_untested" if record["probe"]["exit_code"] == 0 else "not_runnable"
            if name == "ppu-smi":
                record["status"] = run([path], timeout)
        tools[name] = record
    modules = {name: run([sys.executable, "-c", MODULE_PROBE, name], timeout)
               for name in ("torch", "triton")}
    torch_info = {}
    try:
        torch_info = json.loads(modules["torch"]["stdout"])
    except (ValueError, KeyError):
        pass
    ppu_names = [d["name"] for d in torch_info.get("devices", [])
                 if "ppu" in d["name"].lower() or "810" in d["name"]]
    device_list = tools["ppu-smi"].get("probe", {}).get("stdout", "")
    detected = bool(ppu_names or "810" in device_list)
    return {"version": 1, "generated_at": now(), "scope": "this_process_host_only",
            "os": platform.platform(), "python": sys.version, "executable": sys.executable,
            "visible_devices": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "PPU_VISIBLE_DEVICES")},
            "tools": tools, "modules": modules, "ppu_detected": detected,
            "analysis_level": 3 if detected else 4, "sdk_release": None,
            "limitations": ["工具存在不证明采集可用；Event 接口未做计时功能验证。",
                            "SDK 发布号未知；兼容 CUDA target 不等于 NVIDIA 硬件。"]}


def main():
    p = argparse.ArgumentParser(description="只探测当前进程所在主机；远端通过已配置的任务命令运行")
    p.add_argument("--output", required=True)
    p.add_argument("--timeout", type=positive, default=15)
    a = p.parse_args()
    write_json(a.output, probe(a.timeout))


if __name__ == "__main__":
    raise SystemExit(cli(main))
