"""Source-bound compiler captures and diagnostic comparisons, never promotion."""

from __future__ import annotations

import base64
import difflib
import hashlib
import json
import math
import re
import shlex
import uuid
from datetime import datetime, timezone
from pathlib import Path

import candidate_evidence as E
import compiler_backends

ROOT = Path("runs/compiler-evidence")
COMPARISONS = Path("runs/compiler-comparisons")
Error = E.EvidenceError


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def config(task):
    analysis = task.get("analysis") or {}
    if not isinstance(analysis, dict):
        raise Error("analysis must be a mapping")
    value = analysis.get("compiler")
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {
        "probe",
        "guard_command",
        "adapter",
        "max_artifact_bytes",
        "workloads",
    }:
        raise Error("invalid analysis.compiler configuration")
    for key in ("probe", "guard_command"):
        if not isinstance(value.get(key), str) or not value[key].strip():
            raise Error("analysis.compiler requires " + key)
    probe = E.relative_path(value["probe"])
    _, protected = E.declarations(task)
    if not any(probe == p or probe.startswith(p + "/") for p in protected):
        raise Error("compiler probe must be a declared protected input")
    if value.get("adapter", "generic") not in ("generic", "metax-triton"):
        raise Error("unsupported compiler adapter")
    if (
        value.get("adapter") == "metax-triton"
        and task.get("target", {}).get("backend") != "maca"
    ):
        raise Error("metax-triton adapter requires target.backend: maca")
    limit = value.get("max_artifact_bytes", 64 * 1024 * 1024)
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise Error("max_artifact_bytes must be a positive integer")
    workloads = value.get("workloads")
    if (
        not isinstance(workloads, list)
        or not workloads
        or any(not isinstance(x, str) or not x for x in workloads)
        or len(set(workloads)) != len(workloads)
    ):
        raise Error("compiler.workloads must be unique nonempty workload identities")
    return value


def safe_file(root, relative):
    name = E.relative_path(relative)
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()) or any(
        p.is_symlink()
        for p in [path, *path.parents]
        if p.is_relative_to(root) and p != root
    ):
        raise Error("unsafe evidence path: " + name)
    return path


def load(workspace, run_id):
    E.validate_candidate(run_id)
    entry = safe_file(workspace, (ROOT / run_id).as_posix())
    safe_file(entry, "record.json")
    record = E.load_record(entry)
    for name in record.get("artifacts", {}):
        safe_file(entry, name)
    if (
        record.get("status") != "pass"
        or record.get("run_id") != run_id
        or record.get("before") != record.get("after")
    ):
        raise Error("compiler capture is not a successful source-bound record")
    return entry, record


def validate_metric(value):
    required = {"value", "unit", "scope", "definition", "provenance"}
    if not isinstance(value, dict) or set(value) != required:
        raise Error("metric requires value, unit, scope, definition, provenance")
    number = value["value"]
    if (
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(number)
        or number < 0
    ):
        raise Error("resource metrics must be finite nonnegative numbers")
    if any(
        not isinstance(value[k], str) or not value[k].strip()
        for k in required - {"value"}
    ):
        raise Error("metric semantics must be explicit nonempty strings")


def normalize(envelope, entry, settings):
    compiler = envelope.get("compiler")
    if not isinstance(compiler, dict) or any(
        not isinstance(compiler.get(k), str) or not compiler[k].strip()
        for k in ("name", "version", "backend")
    ):
        raise Error("compiler name/version/backend are required")
    if settings.get("adapter") == "metax-triton" and compiler["backend"] not in (
        "maca",
        "metax",
    ):
        raise Error("metax-triton capture requires MetaX compiler identity")
    cases = envelope.get("cases")
    if not isinstance(cases, list) or not cases:
        raise Error("compiler envelope has no cases")
    total = 0
    rows = []
    seen = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise Error("compiler case must be an object")
        for key in ("workload", "kernel", "input_id"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise Error("compiler case missing " + key)
        if not re.fullmatch("[a-f0-9]{64}", case["input_id"]):
            raise Error("input_id must be a SHA256 of the input specification")
        key = (case["workload"], case["kernel"])
        if key in seen:
            raise Error("duplicate workload/kernel mapping")
        seen.add(key)
        if not isinstance(case.get("config"), dict):
            raise Error("compiled configuration must be an object")
        artifacts = case.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise Error("case has no compiler artifacts")
        raw = {}
        files = {}
        for stage, item in artifacts.items():
            E.validate_candidate(stage)
            if (
                not isinstance(item, dict)
                or set(item) != {"encoding", "data"}
                or not isinstance(item["data"], str)
            ):
                raise Error("invalid artifact payload")
            if item["encoding"] == "utf-8":
                data = item["data"].encode("utf-8")
            elif item["encoding"] == "base64":
                try:
                    data = base64.b64decode(item["data"], validate=True)
                except ValueError as exc:
                    raise Error("invalid artifact base64") from exc
            else:
                raise Error("unknown artifact encoding")
            total += len(data)
            if not data or total > settings.get("max_artifact_bytes", 64 * 1024 * 1024):
                raise Error("empty or oversized compiler artifacts")
            path = Path("cases") / str(index) / stage
            out = entry / path
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(data)
            files[stage] = dict(
                path=path.as_posix(),
                sha256=E.digest(out),
                bytes=len(data),
                encoding=item["encoding"],
            )
            raw[stage] = data
        try:
            metrics, notes = compiler_backends.normalize(
                settings.get("adapter", "generic"), case, raw
            )
        except (ValueError, TypeError, AttributeError) as exc:
            raise Error("invalid backend resource evidence: " + str(exc)) from exc
        for name, metric in metrics.items():
            E.validate_candidate(name)
            validate_metric(metric)
        rows.append(
            {k: case[k] for k in ("workload", "kernel", "input_id", "config")}
            | dict(
                artifacts=files,
                metrics=metrics,
                raw_resources=case.get("raw_resources", {}),
                notes=notes,
            )
        )
    if {x["workload"] for x in rows} != set(settings["workloads"]):
        raise Error("compiler workload coverage differs from contract")
    return dict(compiler=compiler, cases=rows)


def capture(workspace, task, candidate, execute):
    E.validate_candidate(candidate)
    settings = config(task)
    if settings is None:
        raise Error("configure analysis.compiler before capture")
    run_id = uuid.uuid4().hex
    entry = workspace / ROOT / run_id
    entry.mkdir(parents=True, exist_ok=False)
    record = dict(
        version=1,
        run_id=run_id,
        candidate=candidate,
        status="running",
        started_at=datetime.now(timezone.utc).isoformat(),
        artifacts={},
        purpose="diagnostic_only_not_promotion",
    )

    def save():
        (entry / "record.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )

    def write(name, text):
        (entry / name).write_text(text or "", encoding="utf-8")
        record["artifacts"][name] = E.digest(entry / name)

    def guard(label):
        result = execute(settings["guard_command"])
        write("guard-" + label + ".stdout.log", result.stdout)
        write("guard-" + label + ".stderr.log", result.stderr)
        value = json.loads(result.stdout) if result.returncode == 0 else {}
        if (
            value.get("idle") is not True
            or not isinstance(value.get("environment_id"), str)
            or not value["environment_id"]
        ):
            raise Error("compiler guard reports busy/unknown environment")
        return value["environment_id"]

    save()
    try:
        record["before"] = E.source_snapshot(workspace, task)
        if record["before"]["files"].get(settings["probe"]) in (None, "<missing>"):
            raise Error("compiler probe is missing from protected snapshot")
        record["environment_id"] = guard("before")
        helper = (
            Path(__file__)
            .with_name("compiler_export.py")
            .read_text(encoding="utf-8-sig")
        )
        write("exporter.py", helper)
        write("task.json", json.dumps(task, indent=2))
        write("collector.py", Path(__file__).read_text(encoding="utf-8-sig"))
        write(
            "backend-adapter.py",
            Path(compiler_backends.__file__).read_text(encoding="utf-8-sig"),
        )
        code = (
            "import runpy\nnamespace={}\nexec("
            + repr(helper)
            + ",namespace)\n"
            + "runpy.run_path("
            + repr(settings["probe"])
            + ",init_globals={'kda_compiler':namespace},run_name='__main__')\n"
        )
        command = (
            "export KDA_COMPILE_RUN_ID="
            + shlex.quote(run_id)
            + "\n"
            + "export KDA_COMPILE_CANDIDATE="
            + shlex.quote(candidate)
            + "\npython3 -c "
            + shlex.quote(code)
        )
        result = execute(command)
        write("stdout.log", result.stdout)
        write("stderr.log", result.stderr)
        record["returncode"] = result.returncode
        if result.returncode:
            raise Error("compiler probe failed")
        limit = settings.get("max_artifact_bytes", 64 * 1024 * 1024)
        if len(result.stdout.encode()) > limit * 3 + 1024 * 1024:
            raise Error("compiler envelope exceeds limit")
        envelope = json.loads(result.stdout)
        if (
            envelope.get("version") != 1
            or envelope.get("run_id") != run_id
            or envelope.get("candidate") != candidate
        ):
            raise Error("compiler envelope run identity mismatch")
        payload = normalize(envelope, entry, settings)
        write("capture.json", json.dumps(payload, indent=2, allow_nan=False) + "\n")
        for case in payload["cases"]:
            for value in case["artifacts"].values():
                record["artifacts"][value["path"]] = value["sha256"]
        if guard("after") != record["environment_id"]:
            raise Error("compiler environment changed during capture")
        record["after"] = E.source_snapshot(workspace, task)
        if record["before"] != record["after"]:
            raise Error("source or protected inputs changed during compiler capture")
        record["status"] = "pass"
        save()
        return record
    except Exception as exc:
        record.update(status="error", error=str(exc))
        save()
        if isinstance(exc, Error):
            raise
        raise Error(str(exc)) from exc


def protected(identity):
    allowed = identity["allowed_paths"]
    return {
        p: value
        for p, value in identity["files"].items()
        if not any(p == a or p.startswith(a + "/") for a in allowed)
    }


def compare(workspace, left_id, right_id):
    left_dir, left = load(workspace, left_id)
    right_dir, right = load(workspace, right_id)
    if left["before"]["task_sha256"] != right["before"]["task_sha256"] or protected(
        left["before"]
    ) != protected(right["before"]):
        raise Error("captures have different contracts or protected inputs")
    for name in ("exporter.py", "backend-adapter.py", "collector.py"):
        if left["artifacts"].get(name) != right["artifacts"].get(name):
            raise Error("captures use different collector/adapter implementations")
    if left["environment_id"] != right["environment_id"]:
        raise Error("captures have different environments")
    a = json.loads((left_dir / "capture.json").read_text())
    b = json.loads((right_dir / "capture.json").read_text())
    if a["compiler"] != b["compiler"]:
        raise Error("captures have different compiler identities")
    index = lambda data: {(c["workload"], c["kernel"]): c for c in data["cases"]}
    aa, bb = index(a), index(b)
    if aa.keys() != bb.keys():
        raise Error("captures have different workload/kernel coverage")
    # Validate comparability before creating any completed comparison.
    for key in aa:
        if aa[key]["input_id"] != bb[key]["input_id"]:
            raise Error("captures have different input specifications")
    result = dict(
        version=1,
        comparison_id=uuid.uuid4().hex,
        left=left_id,
        right=right_id,
        left_record_sha256=E.digest(left_dir / "record.json"),
        right_record_sha256=E.digest(right_dir / "record.json"),
        purpose="diagnostic_only_not_promotion",
        cases=[],
        artifacts={},
    )
    entry = workspace / COMPARISONS / result["comparison_id"]
    entry.mkdir(parents=True)
    for i, key in enumerate(aa):
        x, y = aa[key], bb[key]
        metrics = {}
        stages = {}
        for name in sorted(x["metrics"].keys() | y["metrics"].keys()):
            xm, ym = x["metrics"].get(name), y["metrics"].get(name)
            if xm is None or ym is None:
                metrics[name] = dict(status="unavailable", left=xm, right=ym)
                continue
            if any(xm[k] != ym[k] for k in ("unit", "scope", "definition")):
                metrics[name] = dict(status="incomparable_semantics", left=xm, right=ym)
                continue
            metrics[name] = dict(
                status="comparable", left=xm, right=ym, delta=ym["value"] - xm["value"]
            )
        for stage in sorted(x["artifacts"].keys() | y["artifacts"].keys()):
            xv, yv = x["artifacts"].get(stage), y["artifacts"].get(stage)
            if xv is None or yv is None:
                stages[stage] = dict(status="unavailable")
                continue
            same = xv["sha256"] == yv["sha256"]
            stages[stage] = dict(
                status="identical_bytes" if same else "different_bytes",
                left=xv["sha256"],
                right=yv["sha256"],
            )
            if not same and xv["encoding"] == yv["encoding"] == "utf-8":
                text = "".join(
                    difflib.unified_diff(
                        (left_dir / xv["path"]).read_text().splitlines(True),
                        (right_dir / yv["path"]).read_text().splitlines(True),
                        fromfile="parent/" + stage,
                        tofile="candidate/" + stage,
                    )
                )
                name = f"{i}-{stage}.diff"
                (entry / name).write_text(text)
                result["artifacts"][name] = E.digest(entry / name)
                stages[stage]["diff"] = name
        result["cases"].append(
            dict(
                workload=key[0],
                kernel=key[1],
                input_id=x["input_id"],
                config_left=x["config"],
                config_right=y["config"],
                metrics=metrics,
                stages=stages,
            )
        )
    result["limitations"] = [
        "Byte differences may include debug/layout metadata; they do not prove semantic or performance changes.",
        "Static resource changes do not measure occupancy, dynamic traffic, spill counts, or latency.",
    ]
    (entry / "record.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result
