"""Archive only a supplied launched specialization, without assuming asm keys."""
import argparse
import re
from pathlib import Path
from common import cli, digest, now, write_json


def collect(kernel, output, provenance):
    required = ("source_sha256", "signature", "shape", "dtype", "stride", "grid",
                "config", "target", "versions", "candidate")
    if any(key not in provenance for key in required):
        raise ValueError("产物归属缺少：" + ", ".join(k for k in required if k not in provenance))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"version": 1, "generated_at": now(), "provenance": provenance,
                "artifacts": {}, "resources": {}}
    for index, (key, value) in enumerate(getattr(kernel, "asm", {}).items()):
        if not isinstance(value, (str, bytes, bytearray)):
            continue
        name = f"{index:02d}-" + re.sub(r"[^A-Za-z0-9_.-]", "_", str(key))
        path = output / (name + (".bin" if isinstance(value, (bytes, bytearray)) else ".txt"))
        path.write_bytes(bytes(value) if not isinstance(value, str) else value.encode("utf-8"))
        manifest["artifacts"][str(key)] = {"path": path.name, "sha256": digest(path),
                                         "binary": not isinstance(value, str)}
    metadata = getattr(kernel, "metadata", None)
    manifest["metadata_repr"] = str(metadata)
    for label, names in {"registers": ("n_regs", "num_regs"), "spills": ("n_spills",),
                         "shared_bytes": ("shared",)}.items():
        manifest["resources"][label] = None
        for owner in (kernel, metadata):
            for name in names:
                value = owner.get(name) if isinstance(owner, dict) else getattr(owner, name, None)
                if isinstance(value, (int, float)):
                    manifest["resources"][label] = value
                    break
    manifest["missing"] = [k for k in ("ttir", "ttgir", "llir", "hgbin", "asm")
                           if k not in manifest["artifacts"]]
    write_json(output / "manifest.json", manifest)
    return manifest


def main():
    from benchmark_kernel import load_adapter
    p = argparse.ArgumentParser(description="adapter.compiled() 返回 (已 launch kernel, provenance)；不按缓存时间猜测")
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if not a.worker:
        import sys
        from common import run
        result = run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"], a.timeout)
        write_json(str(a.output) + ".process.json", result)
        if result["exit_code"] != 0:
            raise ValueError("编译产物子进程失败/超时，见 process.json")
        return
    adapter = load_adapter(a.adapter)
    kernel, provenance = adapter.compiled()
    collect(kernel, a.output, provenance)


if __name__ == "__main__":
    raise SystemExit(cli(main))
