"""Executor-side MetaX collection. Invoked only by KDA's contracted transport."""

from pathlib import Path
import base64
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(config):
    workspace = Path.cwd().resolve()
    relative = Path(config["directory"])
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or relative.parts[:2] != ("runs", "profile-evidence")
    ):
        raise ValueError("invalid profile output directory")
    job = workspace / relative
    if not job.resolve().is_relative_to(workspace):
        raise ValueError("profile directory escapes workspace")
    job.mkdir(parents=True, exist_ok=False)
    tool = config["tool"]
    variable, fallback = (
        ("KDA_MCPROFILER_BIN", "mcProfiler")
        if tool == "mcprofiler"
        else (
            "KDA_MCTRACER_BIN",
            str(Path(os.environ.get("MACA_PATH", "/opt/maca")) / "bin/mcTracer"),
        )
    )
    binary = shutil.which(os.environ.get(variable, fallback))
    result = {
        "tool": tool,
        "status": "error",
        "directory": str(relative),
        "artifacts": {},
    }
    try:
        if not binary:
            raise RuntimeError("profiler executable not found")
        binary = Path(binary).resolve()
        result.update(binary=str(binary), binary_sha256=digest(binary))
        version_args = [str(binary), "version" if tool == "mcprofiler" else "--help"]
        version = subprocess.run(
            version_args, cwd=job, capture_output=True, text=True, timeout=30
        )
        (job / "version.stdout.log").write_text(version.stdout)
        (job / "version.stderr.log").write_text(version.stderr)
        if version.returncode:
            raise RuntimeError("profiler version probe failed")
        target = job / "target.py"
        # Append one exit record per tool replay; neither a swallowed failure nor
        # a missing workload execution can pass capture validation.
        target.write_text(
            "import json,os,subprocess\nfrom pathlib import Path\n"
            "os.environ.update(" + repr(config.get("runtime_environment", {})) + ")\n"
            "p=Path(" + repr(str(job / "target-exits.jsonl")) + ")\n"
            'r=subprocess.run(["bash","-c",'
            + repr(config["workload_command"])
            + "],cwd="
            + repr(str(workspace))
            + ")\n"
            'with p.open("a") as f:f.write(json.dumps({"returncode":r.returncode})+"\\n")\n'
            "raise SystemExit(r.returncode)\n"
        )
        child = [sys.executable, str(target)]
        existing = set(binary.parent.glob("output*"))
        if tool == "mctracer":
            # mcTracer 3.7.2 prefixes cwd even when --odname is absolute.
            args = [str(binary), "--mctx", "--odname", "trace", "--name", "kda", *child]
        else:
            args = [
                str(binary),
                "perf_exec",
                "--cmdline",
                shlex.join(child),
                "--cwd",
                str(workspace),
                "--casename",
                job.parent.name,
                "--metrics",
                *config["metrics"],
                "--per-kernel",
                "--counts",
                str(config["counts"]),
            ]
            if config.get("start_from_api"):
                args.append("--custom")
            if config.get("kernel_names"):
                args += ["--kernelnames", *config["kernel_names"]]
        result["argv"] = args
        (job / "argv.json").write_text(json.dumps(args, indent=2))
        with (job / "tool.stdout.log").open("w") as stdout, (
            job / "tool.stderr.log"
        ).open("w") as stderr:
            process = subprocess.Popen(
                args, cwd=job, stdout=stdout, stderr=stderr, start_new_session=True
            )
            try:
                result["returncode"] = process.wait(timeout=config["timeout_seconds"])
            except subprocess.TimeoutExpired:
                # Only this invocation's newly-created process group is owned.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise RuntimeError("profiler capture timeout")
        if tool == "mcprofiler":
            log = (job / "tool.stdout.log").read_text(errors="replace")
            matches = re.findall(r"perf done, please check report file ([^\r\n]+)", log)
            if len(matches) != 1:
                raise RuntimeError(
                    "mcProfiler did not report one completed export directory"
                )
            exported = Path(matches[0].strip())
            # This version writes next to its executable, ignoring capture cwd.
            # Read only the exact new directory named by this invocation.
            if (
                exported.is_symlink()
                or exported.resolve().parent != binary.parent
                or not exported.name.startswith("output")
                or exported in existing
            ):
                raise RuntimeError("stale or untrusted mcProfiler export directory")
            result["vendor_export_directory"] = str(exported)
            total = 0
            for path in exported.rglob("*"):
                if path.is_symlink():
                    raise RuntimeError("symlink in profiler export")
                if path.is_file():
                    total += path.stat().st_size
            if total > config["max_artifact_bytes"]:
                raise RuntimeError("profiler export exceeds artifact limit")
            shutil.copytree(exported, job / "vendor-export")
        if result["returncode"]:
            raise RuntimeError("profiler returned nonzero exit")
        exits = job / "target-exits.jsonl"
        if not exits.is_file():
            raise RuntimeError("profiler did not record workload completion")
        exit_rows = [json.loads(line) for line in exits.read_text().splitlines()]
        if not exit_rows or any(row.get("returncode") != 0 for row in exit_rows):
            raise RuntimeError("profiled workload failed")
        result["workload_executions"] = len(exit_rows)
        result["status"] = "captured"
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    (job / "capture.json").write_text(json.dumps(result, indent=2))
    files = []
    total = 0
    for path in sorted(job.rglob("*")):
        if path.is_symlink():
            raise RuntimeError("symlink in capture artifacts")
        if path.is_file():
            total += path.stat().st_size
            if total > config["max_artifact_bytes"]:
                raise RuntimeError("capture artifacts exceed limit; retained remotely")
            files.append(path)
    memory = io.BytesIO()
    with tarfile.open(fileobj=memory, mode="w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(path.relative_to(job)))
    return {
        "capture": result,
        "files": {str(p.relative_to(job)): digest(p) for p in files},
        "archive_base64": base64.b64encode(memory.getvalue()).decode(),
    }


if __name__ == "__main__":
    print(json.dumps(run(json.loads(sys.argv[1])), allow_nan=False))
