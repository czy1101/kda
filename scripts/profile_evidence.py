"""Source-bound profile collection, separate from promotion benchmark evidence."""

from pathlib import Path, PurePosixPath
import base64
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
import uuid
from datetime import datetime, timezone
import candidate_evidence as evidence
import metax_reports

ROOT = Path("runs/profile-evidence")
Error = evidence.EvidenceError


def validate_config(config, tool):
    if tool not in ("mctracer", "mcprofiler"):
        raise Error("unsupported MetaX adapter tool")
    allowed = {
        "adapter",
        "workload_command",
        "guard_command",
        "metrics",
        "kernel_names",
        "counts",
        "start_from_api",
        "runtime_environment",
        "timeout_seconds",
        "max_artifact_bytes",
        "timestamp_unit",
        "capture_range",
        "tool",
    }
    if set(config) - allowed:
        raise Error(
            "unknown MetaX profile option: " + ", ".join(sorted(set(config) - allowed))
        )
    for key in ("workload_command", "guard_command"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise Error("profile requires " + key)
    for key, default in [
        ("timeout_seconds", 300),
        ("max_artifact_bytes", 67108864),
        ("counts", 10),
    ]:
        value = config.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise Error("invalid profile " + key)
    for key, default in [("metrics", ["Total Cycles"]), ("kernel_names", [])]:
        value = config.get(key, default)
        if (
            not isinstance(value, list)
            or (key == "metrics" and not value)
            or any(not isinstance(x, str) or not x or x.startswith("-") for x in value)
        ):
            raise Error("invalid profile " + key)
    env = config.get("runtime_environment", {})
    if not isinstance(env, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in env.items()
    ):
        raise Error("invalid profile runtime_environment")
    if not isinstance(config.get("start_from_api", False), bool):
        raise Error("start_from_api must be boolean")
    if tool == "mctracer" and config.get("timestamp_unit") != "ns":
        raise Error("supported mcTracer schema requires timestamp_unit: ns")
    scope = config.get("capture_range")
    if scope is not None:
        if (
            tool != "mctracer"
            or not isinstance(scope, dict)
            or set(scope) != {"start_api", "stop_api"}
            or any(not isinstance(v, str) or not v for v in scope.values())
            or scope["start_api"] == scope["stop_api"]
        ):
            raise Error("invalid profile capture_range")


def unpack(envelope, destination, limit):
    if not isinstance(envelope.get("files"), dict):
        raise Error("missing artifact manifest")
    payload = base64.b64decode(envelope["archive_base64"], validate=True)
    if len(payload) > limit:
        raise Error("profile archive exceeds limit")
    total = 0
    names = set()
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (
                not member.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or "\\" in member.name
                or not path.parts
                or str(path) == "."
                or str(path) in names
            ):
                raise Error("unsafe or duplicate profiler artifact")
            total += member.size
            if total > limit:
                raise Error("expanded profile artifacts exceed limit")
            names.add(str(path))
            data = archive.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != envelope["files"].get(str(path)):
                raise Error("profile artifact hash mismatch")
            target = destination / str(path)
            if not target.resolve().is_relative_to(destination.resolve()):
                raise Error("profile artifact escape")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    if names != set(envelope["files"]):
        raise Error("profile manifest members differ")


def run(workspace, task, tool, config, execute, discovery, variant="candidate"):
    validate_config(config, tool)
    available = next(
        (
            t
            for t in discovery.get("tools", [])
            if t.get("id") == tool and t.get("available")
        ),
        None,
    )
    if available is None:
        raise Error("profile adapter requires successful discover-tools evidence")
    if variant != "candidate":
        raise Error("MetaX adapter uses a separate frozen task for reference captures")
    entry = workspace / ROOT / uuid.uuid4().hex
    entry.mkdir(parents=True, exist_ok=False)
    record = dict(
        version=1,
        id=entry.name,
        tool=tool,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        discovery=available,
        artifacts={},
        purpose="diagnostic_only_not_promotion",
    )

    def save():
        (entry / "record.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )

    def log(name, value):
        (entry / name).write_text(value or "")
        record["artifacts"][name] = evidence.digest(entry / name)

    def guard(label):
        completed = execute(config["guard_command"])
        log("guard-" + label + ".stdout.log", completed.stdout)
        log("guard-" + label + ".stderr.log", completed.stderr)
        if completed.returncode:
            raise Error("profile guard command failed")
        value = json.loads(completed.stdout)
        if (
            value.get("idle") is not True
            or not isinstance(value.get("environment_id"), str)
            or not value["environment_id"]
        ):
            raise Error("profile guard reports busy/unknown environment")
        record.setdefault("guards", {})[label] = value
        return value["environment_id"]

    log("task.json", json.dumps(task, indent=2))
    log("collector.py", Path(__file__).with_name("metax_capture.py").read_text())
    log("parser.py", Path(metax_reports.__file__).read_text())
    log("profile-runner.py", Path(__file__).read_text())
    save()
    try:
        before = evidence.source_snapshot(workspace, task)
        record["before"] = before
        environment = guard("before")
        capture = {
            **config,
            "tool": tool,
            "directory": str(ROOT / entry.name / "executor"),
        }
        for key, value in [
            ("metrics", ["Total Cycles"]),
            ("counts", 10),
            ("timeout_seconds", 300),
            ("max_artifact_bytes", 67108864),
        ]:
            capture.setdefault(key, value)
        code = (Path(__file__).with_name("metax_capture.py")).read_text()
        record["collector_sha256"] = hashlib.sha256(code.encode()).hexdigest()
        record["parser_sha256"] = evidence.digest(Path(metax_reports.__file__))
        completed = execute(
            "python3 -c " + shlex.quote(code) + " " + shlex.quote(json.dumps(capture))
        )
        log("transport.stderr.log", completed.stderr)
        if completed.returncode:
            log("transport.stdout.log", completed.stdout)
            raise Error("profile collector execution failed")
        try:
            envelope = json.loads(completed.stdout)
        except ValueError:
            log("transport.stdout.log", completed.stdout)
            raise Error("invalid profile artifact envelope")
        raw = entry / "raw"
        raw.mkdir()
        unpack(envelope, raw, capture["max_artifact_bytes"])
        for name, hash_ in envelope["files"].items():
            record["artifacts"]["raw/" + name] = hash_
        record["capture"] = envelope["capture"]
        if json.loads((raw / "capture.json").read_text()) != record["capture"]:
            raise Error("capture envelope differs from hashed record")
        after_environment = guard("after")
        record["after"] = evidence.source_snapshot(workspace, task)
        if environment != after_environment:
            raise Error("profile environment changed")
        if before != record["after"]:
            raise Error("profile source/protected inputs changed")
        if record["capture"]["status"] != "captured":
            raise Error("capture failed: " + record["capture"].get("error", "unknown"))
        report = metax_reports.normalize(
            raw, tool, config.get("timestamp_unit"), config.get("capture_range")
        )
        metax_reports.write_tables(entry / "analysis", report)
        for p in (entry / "analysis").iterdir():
            record["artifacts"][str(p.relative_to(entry))] = evidence.digest(p)
        record["validated_capabilities"] = report["capabilities"]
        if tool == "mcprofiler":
            available_metrics = sorted(
                {r["metric"] for r in report["metrics"] if r["status"] == "available"}
            )
            scopes = sorted({r["source"] for r in report["metrics"]})
            missing_by_scope = {
                scope: [
                    name
                    for name in capture["metrics"]
                    if not any(
                        r["source"] == scope
                        and r["metric"] == name
                        and r["status"] == "available"
                        for r in report["metrics"]
                    )
                ]
                for scope in scopes
            }
            record["metric_coverage"] = {
                "requested": capture["metrics"],
                "available": available_metrics,
                "unavailable": [
                    name for name in capture["metrics"] if name not in available_metrics
                ],
                "missing_by_scope": {
                    scope: names for scope, names in missing_by_scope.items() if names
                },
            }
            if (
                record["metric_coverage"]["unavailable"]
                or record["metric_coverage"]["missing_by_scope"]
            ):
                raise Error(
                    "requested profiler metrics are missing or unsupported; inspect metric_coverage"
                )
        record["status"] = "pass"
        return subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                {
                    "profile_id": entry.name,
                    "status": "pass",
                    "validated_capabilities": report["capabilities"],
                    "evidence_directory": str(entry),
                },
                indent=2,
            )
            + "\n",
            "",
        )
    except BaseException as error:
        record.update(status="error", error=f"{type(error).__name__}: {error}")
        if isinstance(
            error, (metax_reports.ReportError, ValueError)
        ) and not isinstance(error, Error):
            raise Error(str(error)) from error
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        save()


def report(workspace, run_id):
    import re

    if not isinstance(run_id, str) or not re.fullmatch("[0-9a-f]{32}", run_id):
        raise Error("invalid profile run ID")
    entry = workspace / ROOT / run_id
    record = evidence.load_record(entry)
    if record.get("status") != "pass":
        raise Error(
            "profile run is not passing: " + str(record.get("error", "unknown"))
        )
    if record.get("before") != record.get("after"):
        raise Error("profile source identity mismatch")
    data = json.loads((entry / "analysis/normalized.json").read_text())
    return dict(
        profile_id=run_id,
        tool=record["tool"],
        status="verified",
        historical_evidence=True,
        validated_capabilities=record["validated_capabilities"],
        kernel_count=len(data.get("kernels", [])),
        api_count=len(data.get("apis", [])),
        gap_count=len(data.get("gaps", [])),
        metric_count=len(data.get("metrics", [])),
        capture_ranges=data.get("capture_ranges", []),
        devices=data.get("devices", []),
        correlation_coverage={
            "matched": sum(
                k.get("api_link_status") == "matched" for k in data.get("kernels", [])
            ),
            "total": len(data.get("kernels", [])),
        },
        metric_coverage=record.get("metric_coverage", {}),
        limitations=data["limitations"],
        evidence_directory=str(entry),
    )
