"""Backend-neutral repeated measurement and confirmation evidence."""

from __future__ import annotations
import csv
import json
import math
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path
import candidate_evidence as evidence

Error = evidence.EvidenceError
ROOT = Path("runs/measurements")
CONFIRM = Path("runs/confirmations")
FROZEN = Path(".kernelpilot/measurement-baseline.json")


class Inconclusive(Error):
    def __init__(self, message, category, details=None):
        super().__init__(message)
        self.category = category
        self.details = details


def policy(task):
    value = task.get("measurement")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise Error("measurement must be an object")
    required = {
        "conditions",
        "process_repeats",
        "min_samples",
        "max_relative_mad",
        "min_improvement",
        "confirmation_pairs",
        "guard_command",
        "samples_path",
    }
    if set(value) != required:
        raise Error("measurement must define exactly: " + ", ".join(sorted(required)))
    for key, minimum in [
        ("process_repeats", 2),
        ("min_samples", 3),
        ("confirmation_pairs", 2),
    ]:
        if (
            isinstance(value[key], bool)
            or not isinstance(value[key], int)
            or value[key] < minimum
        ):
            raise Error(f"measurement.{key} must be an integer >= {minimum}")
    for key in ("max_relative_mad", "min_improvement"):
        number = value[key]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(number)
            or not 0 <= number < 1
        ):
            raise Error(f"measurement.{key} must be finite in [0,1)")
    conditions = value["conditions"]
    if not isinstance(conditions, dict) or set(conditions) != {
        "warmup",
        "iterations",
        "timing_scope",
        "timer",
        "input_id",
    }:
        raise Error(
            "conditions must define warmup, iterations, timing_scope, timer and input_id"
        )
    for key, minimum in [("warmup", 0), ("iterations", 1)]:
        if (
            isinstance(conditions[key], bool)
            or not isinstance(conditions[key], int)
            or conditions[key] < minimum
        ):
            raise Error("invalid measurement condition: " + key)
    for key in ("timing_scope", "timer", "input_id"):
        if not isinstance(conditions[key], str) or not conditions[key].strip():
            raise Error("measurement condition must be non-empty: " + key)
    for key in ("guard_command", "samples_path"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise Error("measurement field must be non-empty: " + key)
    path = evidence.relative_path(value["samples_path"])
    if Path(path).parts[0] not in ("runs", "outputs"):
        raise Error("measurement.samples_path must be under runs/ or outputs/")
    if path == evidence.result_path(task):
        raise Error("sample artifact and benchmark CSV must use different paths")
    shapes = task.get("benchmark", {}).get("shapes")
    if (
        not isinstance(shapes, list)
        or not shapes
        or any(not isinstance(s, str) or not s for s in shapes)
        or len(shapes) != len(set(shapes))
    ):
        raise Error("reliable measurement requires explicit unique benchmark.shapes")
    if task.get("baseline", {}).get("role") != "initial_implementation":
        raise Error(
            "reliable measurement requires baseline.role: initial_implementation; external references stay separate"
        )
    if (
        task.get("benchmark", {}).get("warmup", conditions["warmup"])
        != conditions["warmup"]
    ):
        raise Error("benchmark.warmup conflicts with measurement conditions")
    if task.get("goal", {}).get("max_regression") is None:
        raise Error("reliable measurement requires goal.max_regression")
    return value


def stats(samples):
    if not samples or any(
        isinstance(x, bool)
        or not isinstance(x, (int, float))
        or not math.isfinite(x)
        or x <= 0
        for x in samples
    ):
        raise Error("raw samples must be finite positive numbers")
    median = statistics.median(samples)
    mad = statistics.median(abs(x - median) for x in samples)
    window = max(1, len(samples) // 3)
    return {
        "median": median,
        "relative_mad": mad / median,
        "min": min(samples),
        "max": max(samples),
        "count": len(samples),
        "sequence_window_size": window,
        "sequence_first_median": statistics.median(samples[:window]),
        "sequence_last_median": statistics.median(samples[-window:]),
    }


def write_record(entry, record):
    (entry / "record.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )


def begin(workspace, root, candidate):
    evidence.validate_candidate(candidate)
    entry = workspace / root / uuid.uuid4().hex
    entry.mkdir(parents=True, exist_ok=False)
    record = {
        "id": entry.name,
        "candidate": candidate,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "run_ids": [],
        "artifacts": {},
    }
    write_record(entry, record)
    return entry, record


def latest(workspace, root, candidate):
    found = []
    for entry in (workspace / root).glob("*"):
        if (entry / "record.json").is_file():
            r = json.loads((entry / "record.json").read_text())
            if r.get("candidate") == candidate:
                found.append((r["started_at"], entry))
    if not found:
        raise Error("no measurement/confirmation evidence for " + candidate)
    entry = max(found)[1]
    return entry, evidence.load_record(entry)


def guard(entry, label, execute, task, record):
    completed = execute(policy(task)["guard_command"])
    for suffix, text in [("stdout", completed.stdout), ("stderr", completed.stderr)]:
        name = f"guard-{label}.{suffix}.log"
        (entry / name).write_text(text or "")
        record["artifacts"][name] = evidence.digest(entry / name)
    if completed.returncode:
        raise Error("device guard command failed; measurement not admissible")
    try:
        value = json.loads(completed.stdout)
    except (ValueError, TypeError) as error:
        raise Error("device guard must return JSON") from error
    if (
        not isinstance(value, dict)
        or value.get("idle") is not True
        or not isinstance(value.get("environment_id"), str)
        or not value["environment_id"]
    ):
        raise Inconclusive(
            "device guard reports busy/unknown environment; measurement not admissible",
            "contaminated_or_unknown",
        )
    record.setdefault("guards", {})[label] = value
    return value["environment_id"]


def accept_samples(raw, task, candidate, run_id, rows):
    p = policy(task)
    try:
        data = json.loads(raw)
    except (ValueError, TypeError) as error:
        raise Error("raw sample artifact must be JSON") from error
    if (
        not isinstance(data, dict)
        or data.get("run_id") != run_id
        or data.get("candidate") != candidate
        or data.get("conditions") != p["conditions"]
    ):
        raise Error(
            "raw samples do not match run, candidate or frozen measurement conditions"
        )
    values = data.get("samples")
    if not isinstance(values, dict) or set(values) != set(task["benchmark"]["shapes"]):
        raise Error("raw samples must cover exactly the declared workloads")
    summaries = {}
    noisy = []
    by_shape = {row["shape"]: row for row in rows}
    for shape, samples in values.items():
        if not isinstance(samples, list) or len(samples) < p["min_samples"]:
            raise Error("insufficient raw samples for " + shape)
        summary = stats(samples)
        summaries[shape] = summary
        if summary["relative_mad"] > p["max_relative_mad"]:
            noisy.append(shape)
        row = by_shape.get(shape)
        if row is None or data.get("unit") != row.get("unit"):
            raise Error("raw sample unit/workload does not match CSV")
        if not math.isclose(
            summary["median"], float(row["metric_value"]), rel_tol=1e-9, abs_tol=0
        ):
            raise Error("CSV summary does not match raw-sample median: " + shape)
    result = {
        "conditions": data["conditions"],
        "unit": data["unit"],
        "per_shape": summaries,
    }
    if noisy:
        raise Inconclusive(
            "within-process measurement dispersion exceeds limit: " + ", ".join(noisy),
            "noise",
            dict(exceeded_shapes=noisy, **result),
        )
    return result


def run_record(workspace, run_id):
    return evidence.load_record(workspace / evidence.ROOT / run_id)


def read_frozen(workspace):
    p = workspace / FROZEN
    if not p.is_file():
        raise Error(
            "freeze repeated baseline measurements first: measure --candidate baseline"
        )
    data = json.loads(p.read_text())
    entry = workspace / ROOT / data["measurement_id"]
    if evidence.digest(entry / "record.json") != data["record_sha256"]:
        raise Error("frozen baseline record hash mismatch")
    record = evidence.load_record(entry)
    if record["status"] != "pass":
        raise Error("frozen baseline is not passing")
    for run_id in record["run_ids"]:
        run = run_record(workspace, run_id)
        if (
            run.get("status") != "pass"
            or evidence.digest(workspace / evidence.ROOT / run_id / "record.json")
            != record["run_record_hashes"][run_id]
        ):
            raise Error("frozen baseline execution record changed")
    return record


def make_evaluation(entry, task, candidate, values, baselines, unit, evaluate):
    (entry / "runs").mkdir(exist_ok=True)
    with (entry / "runs/benchmark.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate",
                "shape",
                "metric_value",
                "baseline_value",
                "unit",
                "status",
            ],
        )
        writer.writeheader()
        for shape in task["benchmark"]["shapes"]:
            writer.writerow(
                dict(
                    candidate=candidate,
                    shape=shape,
                    metric_value=values[shape],
                    baseline_value=baselines[shape],
                    unit=unit,
                    status="pass",
                )
            )
    result, errors = evaluate(entry, task, candidate)
    if errors:
        raise Error("; ".join(errors))
    return result


def tracked_benchmark(workspace, record, run_stage, candidate, **labels):
    """Link every attempted process, including errors, without accepting it."""
    outcome = None
    try:
        outcome = run_stage("benchmark", candidate)
        return outcome
    except BaseException as error:
        outcome = error
        raise
    finally:
        attempt = dict(candidate=candidate, **labels)
        run_id = getattr(outcome, "kernelpilot_run_id", None)
        if run_id:
            path = workspace / evidence.ROOT / run_id / "record.json"
            try:
                run = evidence.load_record(path.parent)
                attempt.update(
                    run_id=run_id,
                    status=run["status"],
                    record_sha256=evidence.digest(path),
                )
                for key in ("category", "error"):
                    if key in run:
                        attempt[key] = run[key]
            except (OSError, ValueError, KeyError, Error) as error:
                attempt.update(
                    run_id=run_id, status="unreadable_evidence", error=str(error)
                )
        else:
            attempt.update(status="unbound_error", error=str(outcome))
        record.setdefault("attempts", []).append(attempt)


def measure(workspace, task, candidate, run_stage, evaluate):
    p = policy(task)
    if p is None:
        raise Error("measure requires a measurement policy")
    if candidate == "baseline" and (workspace / FROZEN).exists():
        raise Error("baseline is already frozen; create a new round to replace it")
    entry, record = begin(workspace, ROOT, candidate)
    try:
        identity = evidence.source_snapshot(workspace, task)
        record["identity"] = identity
        frozen = None if candidate == "baseline" else read_frozen(workspace)
        if frozen and frozen["identity"]["task_sha256"] != identity["task_sha256"]:
            raise Error("measurement contract differs from frozen baseline")
        if frozen:
            allowed = identity["allowed_paths"]
            protected = lambda i: {
                k: v
                for k, v in i["files"].items()
                if not any(k == p or k.startswith(p + "/") for p in allowed)
            }
            if protected(frozen["identity"]) != protected(identity):
                raise Error("protected inputs changed from frozen measurement baseline")
        medians = {shape: [] for shape in task["benchmark"]["shapes"]}
        sample_counts = None
        envs = set()
        units = set()
        correct_ids = set()
        for repeat in range(p["process_repeats"]):
            completed = tracked_benchmark(
                workspace, record, run_stage, candidate, repeat=repeat
            )
            if completed.returncode:
                raise Error("benchmark process failed")
            _, run = evidence.require_pass(workspace, candidate, "benchmark", identity)
            if run["run_id"] in record["run_ids"]:
                raise Error("repeated benchmark invocation reused a run ID")
            record["run_ids"].append(run["run_id"])
            record.setdefault("run_record_hashes", {})[run["run_id"]] = evidence.digest(
                workspace / evidence.ROOT / run["run_id"] / "record.json"
            )
            correct_ids.add(run["correctness_run_id"])
            summary = run["measurement"]
            counts = {
                shape: row["count"] for shape, row in summary["per_shape"].items()
            }
            if sample_counts is None:
                sample_counts = counts
            if counts != sample_counts or (
                frozen and counts != frozen["sample_counts"]
            ):
                raise Error(
                    "sample counts changed between processes or from frozen baseline"
                )
            envs.add(summary["environment_id"])
            units.add(summary["unit"])
            for shape in medians:
                medians[shape].append(summary["per_shape"][shape]["median"])
            write_record(entry, record)
        if len(envs) != 1 or len(units) != 1 or len(correct_ids) != 1:
            raise Error(
                "environment, unit or correctness identity changed across repeats"
            )
        summaries = {shape: stats(values) for shape, values in medians.items()}
        if any(s["relative_mad"] > p["max_relative_mad"] for s in summaries.values()):
            raise Inconclusive(
                "between-process measurement dispersion exceeds limit",
                "noise",
                dict(
                    per_shape=summaries,
                    exceeded_shapes=[
                        shape
                        for shape, value in summaries.items()
                        if value["relative_mad"] > p["max_relative_mad"]
                    ],
                ),
            )
        unit = next(iter(units))
        env = next(iter(envs))
        if frozen and (
            frozen["environment_id"] != env or frozen["evaluation"]["unit"] != unit
        ):
            raise Inconclusive(
                "runtime environment or unit differs from frozen baseline",
                "environment_changed",
            )
        values = {shape: s["median"] for shape, s in summaries.items()}
        baseline_values = (
            values
            if frozen is None
            else {
                row["shape"]: row["metric_value"]
                for row in frozen["evaluation"]["per_shape"]
            }
        )
        record.update(
            environment_id=env,
            correctness_run_id=next(iter(correct_ids)),
            per_shape=summaries,
            sample_counts=sample_counts,
        )
        record["evaluation"] = make_evaluation(
            entry, task, candidate, values, baseline_values, unit, evaluate
        )
        record["artifacts"]["runs/benchmark.csv"] = evidence.digest(
            entry / "runs/benchmark.csv"
        )
        record["status"] = "pass"
    except BaseException as error:
        if isinstance(error, Inconclusive) and error.details is not None:
            record["diagnostics"] = error.details
        record.update(
            status="inconclusive" if isinstance(error, Inconclusive) else "error",
            category=getattr(error, "category", "invalid_evidence_or_execution"),
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_record(entry, record)
    if candidate == "baseline":
        destination = workspace / FROZEN
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("x") as f:
            json.dump(
                {
                    "measurement_id": entry.name,
                    "record_sha256": evidence.digest(entry / "record.json"),
                },
                f,
                indent=2,
            )
    return record


def require_measurement(workspace, task, candidate, identity, evaluate):
    entry, record = latest(workspace, ROOT, candidate)
    if record.get("status") != "pass" or record.get("identity") != identity:
        raise Error("latest repeated measurement is failed/stale")
    _, correct = evidence.require_pass(workspace, candidate, "correctness", identity)
    if record["correctness_run_id"] != correct["run_id"]:
        raise Error("repeated measurements precede latest correctness")
    for run_id in record["run_ids"]:
        run = run_record(workspace, run_id)
        if (
            evidence.digest(workspace / evidence.ROOT / run_id / "record.json")
            != record["run_record_hashes"][run_id]
        ):
            raise Error("bound run record changed after aggregation")
        if (
            run["status"] != "pass"
            or run["before"] != identity
            or run["after"] != identity
        ):
            raise Error("repeated measurement contains invalid execution")
    frozen = read_frozen(workspace)
    if candidate == "baseline" and frozen["id"] != record["id"]:
        raise Error("baseline measurement does not match frozen record")
    result, errors = evaluate(entry, task, candidate)
    if errors:
        raise Error("; ".join(errors))
    return record, result


def require_confirmation(workspace, task, candidate, measured, previous):
    entry, record = latest(workspace, CONFIRM, candidate)
    if (
        record.get("status") != "pass"
        or record.get("identity") != measured["identity"]
        or record.get("measurement_id") != measured["id"]
    ):
        raise Error("latest interleaved confirmation is missing, failed or stale")
    best_path = workspace / ".kernelpilot/checkpoints/best/checkpoint.json"
    if evidence.digest(best_path) != record.get("best_checkpoint_sha256"):
        raise Error("current best differs from interleaved reference")
    if record.get("best_candidate") != previous.get("candidate"):
        raise Error("interleaved comparison selected a different best")
    if len(record.get("pairs", [])) != policy(task)["confirmation_pairs"]:
        raise Error("confirmation lacks required pairs")
    for run_id in record["run_ids"]:
        run = run_record(workspace, run_id)
        if (
            run.get("status") != "pass"
            or evidence.digest(workspace / evidence.ROOT / run_id / "record.json")
            != record["run_record_hashes"][run_id]
        ):
            raise Error("confirmation run evidence changed")
    _, current = evidence.latest(workspace, candidate, "benchmark")
    if current["run_id"] != record["run_ids"][-1]:
        raise Error("confirmation does not cover the latest candidate benchmark")
    return record
