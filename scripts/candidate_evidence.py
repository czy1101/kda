"""Content-bound evidence for candidate runs, independent of a GPU backend."""

from __future__ import annotations

import csv
import hashlib
import inspect
import io
import json
import re
import shlex
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from controlled_ssh import ControlledSSH, command_script

ROOT = Path("runs/command-evidence")


class EvidenceError(ValueError):
    pass


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_candidate(candidate):
    if not isinstance(candidate, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", candidate
    ):
        raise EvidenceError(
            "candidate must be a safe non-empty identifier (maximum 128 characters)"
        )


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise EvidenceError(
            "evidence paths must be non-empty repository-relative POSIX paths"
        )
    p = PurePosixPath(value)
    if p.is_absolute() or ".." in p.parts or p.as_posix() == ".":
        raise EvidenceError(f"evidence path must stay inside workspace: {value}")
    return p.as_posix()


def snapshot_files(root, paths, required):
    # This self-contained function is also executed through ControlledSSH.
    import hashlib
    from pathlib import Path

    root = Path(root).resolve()
    result = {}
    for name in paths:
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError("evidence path escapes workspace: " + name)
        if any(
            p.is_symlink()
            for p in [path, *path.parents]
            if p != root and p.is_relative_to(root)
        ):
            raise ValueError("symlinked evidence paths are not supported: " + name)
        if not path.exists():
            if name in required:
                raise ValueError("missing implementation path: " + name)
            result[name] = "<missing>"
            continue
        children = [path] if path.is_file() else sorted(path.rglob("*"))
        result[name] = "<directory>" if path.is_dir() else None
        for child in children:
            if child.is_symlink():
                raise ValueError(
                    "symlinked evidence paths are not supported: " + str(child)
                )
            if child.is_file():
                h = hashlib.sha256()
                with child.open("rb") as f:
                    for block in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(block)
                result[child.relative_to(root).as_posix()] = h.hexdigest()
    return result


def declarations(task):
    constraints = task.get("constraints", {})
    allowed = [relative_path(p) for p in constraints.get("allowed_paths", [])]
    if not allowed:
        raise EvidenceError("candidate evidence requires allowed_paths")
    protected = [relative_path(p) for p in constraints.get("forbidden_paths", [])]
    reference = task.get("reference", {}).get("path")
    if reference:
        protected.append(relative_path(reference))
    return sorted(set(allowed)), sorted(set(protected))


def source_snapshot(workspace, task):
    allowed, protected = declarations(task)
    paths = sorted(set(allowed + protected))
    if task.get("execution", {}).get("transport", "local") == "ssh":
        code = (
            inspect.getsource(snapshot_files)
            + '\nimport json\nprint(json.dumps(snapshot_files(".", '
            + repr(paths)
            + ", "
            + repr(allowed)
            + ")))\n"
        )
        shell, script = command_script(task, "python3 -c " + shlex.quote(code))
        result = ControlledSSH(task).run_script(script, shell=shell, capture=True)
        if result.returncode:
            raise EvidenceError(
                "remote evidence snapshot failed: " + (result.stderr or "")
            )
        try:
            files = json.loads(result.stdout)
        except (TypeError, ValueError) as error:
            raise EvidenceError("remote snapshot did not return JSON") from error
    else:
        try:
            files = snapshot_files(workspace, paths, allowed)
        except ValueError as error:
            raise EvidenceError(str(error)) from error
    return {
        "files": files,
        "allowed_paths": allowed,
        "task_sha256": hashlib.sha256(
            json.dumps(task, sort_keys=True, allow_nan=False).encode()
        ).hexdigest(),
    }


def local_allowed_snapshot(workspace, task):
    allowed, _ = declarations(task)
    return snapshot_files(workspace, allowed, allowed)


def result_path(task):
    p = relative_path(
        task.get("benchmark", {}).get("result_path", "runs/benchmark.csv")
    )
    if PurePosixPath(p).parts[0] not in ("runs", "outputs"):
        raise EvidenceError("benchmark.result_path must be under runs/ or outputs/")
    return p


def read_result(workspace, task, relative=None):
    relative = result_path(task) if relative is None else relative_path(relative)
    if task.get("execution", {}).get("transport", "local") == "ssh":
        # Read only the contracted artifact. Do not extend source-sync allowlists.
        code = (
            "from pathlib import Path; p=Path(" + repr(relative) + "); "
            "assert p.resolve().is_relative_to(Path.cwd().resolve()); "
            'print(p.read_text(), end="")'
        )
        shell, script = command_script(task, "python3 -c " + shlex.quote(code))
        result = ControlledSSH(task).run_script(script, shell=shell, capture=True)
        if result.returncode:
            raise EvidenceError(
                "cannot read contracted benchmark result: " + (result.stderr or "")
            )
        return result.stdout
    p = workspace / relative
    if not p.resolve().is_relative_to(workspace.resolve()):
        raise EvidenceError("benchmark result escapes workspace")
    return p.read_text()


def load_record(entry):
    record = json.loads((entry / "record.json").read_text())
    for name, expected in record.get("artifacts", {}).items():
        p = entry / relative_path(name)
        if not p.resolve().is_relative_to(entry.resolve()) or digest(p) != expected:
            raise EvidenceError("run artifact hash mismatch: " + str(p))
    return record


def latest(workspace, candidate, section):
    records = []
    for entry in (workspace / ROOT).glob("*"):
        if not (entry / "record.json").is_file():
            continue
        record = json.loads((entry / "record.json").read_text())
        if record.get("candidate") == candidate and record.get("section") == section:
            records.append((record["started_at"], entry))
    if not records:
        raise EvidenceError(
            f"no bound {section} evidence for candidate {candidate!r}; rerun with --candidate"
        )
    entry = max(records)[1]
    return entry, load_record(entry)


def require_pass(workspace, candidate, section, identity):
    entry, record = latest(workspace, candidate, section)
    if (
        record.get("status") != "pass"
        or record.get("before") != identity
        or record.get("after") != identity
    ):
        raise EvidenceError(
            f"{section} evidence failed or no longer matches source/task for {candidate!r}"
        )
    return entry, record


def run(workspace, task, section, candidate, execute, evaluate):
    import measurement_evidence as measurement

    validate_candidate(candidate)
    measurement_policy = measurement.policy(task)
    if section not in ("correctness", "benchmark", "generalization"):
        raise EvidenceError(
            "--candidate is supported only for correctness, benchmark and generalization"
        )
    run_id = uuid.uuid4().hex
    entry = workspace / ROOT / run_id
    entry.mkdir(parents=True, exist_ok=False)
    record = {
        "version": 1,
        "run_id": run_id,
        "candidate": candidate,
        "section": section,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "artifacts": {},
    }
    (entry / "record.json").write_text(json.dumps(record, indent=2))
    try:
        before = source_snapshot(workspace, task)
        record["before"] = before
        if section in ("benchmark", "generalization"):
            _, correct = require_pass(workspace, candidate, "correctness", before)
            record["correctness_run_id"] = correct["run_id"]
            if section == "generalization":
                _, benchmark = require_pass(workspace, candidate, "benchmark", before)
                if benchmark.get("correctness_run_id") != correct["run_id"]:
                    raise EvidenceError(
                        "generalization requires benchmark after latest correctness"
                    )
                record["benchmark_run_id"] = benchmark["run_id"]
        command = task[section]["command"]
        prefix = (
            "export KERNELPILOT_RUN_ID="
            + shlex.quote(run_id)
            + "\nexport KERNELPILOT_CANDIDATE="
            + shlex.quote(candidate)
            + "\n"
        )
        if measurement_policy and section == "benchmark":
            prefix += (
                "export KERNELPILOT_MEASUREMENT_CONDITIONS="
                + shlex.quote(
                    json.dumps(measurement_policy["conditions"], sort_keys=True)
                )
                + "\n"
            )
            before_environment = measurement.guard(
                entry, "before", execute, task, record
            )
        completed = execute(prefix + command)
        completed.kernelpilot_run_id = run_id
        for name, value in [
            ("stdout.log", completed.stdout),
            ("stderr.log", completed.stderr),
        ]:
            (entry / name).write_text(value or "")
            record["artifacts"][name] = digest(entry / name)
        record["returncode"] = completed.returncode
        after = source_snapshot(workspace, task)
        record["after"] = after
        if before != after:
            raise EvidenceError(
                "source or declared protected files changed during execution"
            )
        if completed.returncode:
            record["status"] = "fail"
            return completed
        if section == "benchmark":
            raw = read_result(workspace, task)
            (entry / "benchmark-raw.csv").write_text(raw)
            record["artifacts"]["benchmark-raw.csv"] = digest(
                entry / "benchmark-raw.csv"
            )
            reader = csv.DictReader(io.StringIO(raw))
            if not {"run_id", "candidate"}.issubset(reader.fieldnames or []):
                raise EvidenceError(
                    "benchmark result requires run_id and candidate columns"
                )
            rows = [row for row in reader if row.get("candidate") == candidate]
            if not rows or any(row.get("run_id") != run_id for row in rows):
                raise EvidenceError(
                    "benchmark rows must carry this invocation run_id and candidate"
                )
            (entry / "runs").mkdir()
            with (entry / "runs/benchmark.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=reader.fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            record["artifacts"]["runs/benchmark.csv"] = digest(
                entry / "runs/benchmark.csv"
            )
            evaluation, errors = evaluate(entry, task, candidate)
            if errors:
                raise EvidenceError("invalid bound benchmark: " + "; ".join(errors))
            record["evaluation"] = evaluation
            if measurement_policy:
                raw_samples = read_result(
                    workspace, task, measurement_policy["samples_path"]
                )
                (entry / "samples.json").write_text(raw_samples)
                record["artifacts"]["samples.json"] = digest(entry / "samples.json")
                after_environment = measurement.guard(
                    entry, "after", execute, task, record
                )
                if before_environment != after_environment:
                    raise measurement.Inconclusive(
                        "environment changed during benchmark", "environment_changed"
                    )
                record["measurement"] = measurement.accept_samples(
                    raw_samples, task, candidate, run_id, rows
                )
                record["measurement"]["environment_id"] = before_environment
        record["status"] = "pass"
        return completed
    except BaseException as error:
        error.kernelpilot_run_id = run_id
        record["status"] = (
            "inconclusive" if isinstance(error, measurement.Inconclusive) else "error"
        )
        if isinstance(error, measurement.Inconclusive) and error.details is not None:
            record["diagnostics"] = error.details
        record["category"] = getattr(error, "category", "invalid_evidence_or_execution")
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        (entry / "record.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n"
        )
