#!/usr/bin/env python3
"""Minimal, evidence-gated Codex runner for KDA task workspaces."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


KDA_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from controlled_ssh import (  # noqa: E402
    ControlledSSH,
    SSHAdapterError,
    command_script as remote_command_script,
    inspect_remote,
    mirror_root,
    pull_allowed,
    push_allowed,
)

DEFAULT_TASK = Path(".kernelpilot/task.yaml")
EVIDENCE_DIRS = ("docs", "runs", "outputs", "profile")
REQUIRED_TOP_LEVEL = (
    "version",
    "task",
    "target",
    "implementation",
    "correctness",
    "benchmark",
    "goal",
)


class ContractError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return "<" in value and ">" in value
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return False


def load_task(workspace: Path, task_path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    path = task_path or DEFAULT_TASK
    if not path.is_absolute():
        path = workspace / path
    if not path.is_file():
        raise ContractError(f"task contract not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        task = yaml.safe_load(handle)
    if not isinstance(task, dict):
        raise ContractError("task contract must be a YAML mapping")
    return path.resolve(), task


def validate_backend_profile(task: dict[str, Any]) -> list[str]:
    profile_id = task.get("target", {}).get("profile")
    if not profile_id:
        return []
    path = KDA_ROOT / "backends" / f"{profile_id}.yaml"
    if not path.is_file():
        return [f"unknown target.profile: {profile_id}"]
    try:
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        return [f"invalid backend profile {profile_id}: {error}"]
    if not isinstance(profile, dict):
        return [f"backend profile {profile_id} must be a mapping"]
    errors: list[str] = []
    if profile.get("version") != 1 or profile.get("id") != profile_id:
        errors.append(f"backend profile {profile_id} has an invalid version or id")
    target = task.get("target", {})
    match = profile.get("match") or {}
    if match.get("backend") != target.get("backend"):
        errors.append(f"backend profile {profile_id} does not match target.backend")
    for field, target_field in (("arches", "arch"), ("devices", "device"), ("languages", "language")):
        accepted = match.get(field) or []
        actual = target.get(target_field)
        if accepted and actual and actual not in accepted:
            errors.append(f"backend profile {profile_id} does not match target.{target_field}")
    skill_groups = profile.get("skills") or {}
    for group in ("always", "when_profiling", "when_researching"):
        for relative in skill_groups.get(group, []):
            skill_file = KDA_ROOT / relative / "SKILL.md"
            if not skill_file.is_file():
                errors.append(f"backend profile {profile_id} references missing skill: {relative}")
    return errors


def validate_task(workspace: Path, task: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in REQUIRED_TOP_LEVEL:
        if key not in task:
            errors.append(f"missing required top-level field: {key}")
    if _contains_placeholder(task):
        errors.append("task contract still contains <placeholder> values")
    if task.get("version") != 1:
        errors.append("version must be 1")

    execution = task.get("execution") or {"transport": "local"}
    transport = execution.get("transport", "local") if isinstance(execution, dict) else None
    if transport not in ("local", "ssh"):
        errors.append("execution.transport must be local or ssh")
    if transport == "ssh":
        if not execution.get("host"):
            errors.append("execution.host is required for ssh transport")
        remote_workspace = execution.get("workspace")
        if not isinstance(remote_workspace, str) or not remote_workspace.startswith("/"):
            errors.append("execution.workspace must be an absolute POSIX path for ssh transport")

    implementation = task.get("implementation", {})
    implementation_path = implementation.get("path")
    if not isinstance(implementation_path, str) or not implementation_path:
        errors.append("implementation.path must be a non-empty string")
    elif transport != "ssh" and not (workspace / implementation_path).is_file():
        errors.append(f"implementation.path does not exist: {implementation_path}")

    for section in ("correctness", "benchmark"):
        command = task.get(section, {}).get("command")
        if not isinstance(command, str) or not command.strip():
            errors.append(f"{section}.command must be a non-empty string")

    goal = task.get("goal", {})
    if "relative_to_baseline" not in goal and "target_value" not in goal:
        errors.append("goal needs relative_to_baseline or target_value")
    direction = goal.get("direction", "minimize")
    if direction not in ("minimize", "maximize"):
        errors.append("goal.direction must be minimize or maximize")
    aggregation = goal.get("aggregation", "sum_ratio")
    if aggregation not in ("sum_ratio", "mean_ratio", "geomean_ratio", "all"):
        errors.append(
            "goal.aggregation must be sum_ratio, mean_ratio, geomean_ratio, or all"
        )

    errors.extend(validate_backend_profile(task))

    constraints = task.get("constraints", {})
    allowed = constraints.get("allowed_paths", [])
    forbidden = constraints.get("forbidden_paths", [])
    if implementation_path and allowed and implementation_path not in allowed:
        errors.append("implementation.path must appear in constraints.allowed_paths")
    overlap = sorted(set(allowed) & set(forbidden))
    if overlap:
        errors.append(f"allowed_paths and forbidden_paths overlap: {overlap}")

    environment = task.get("environment")
    if environment is not None:
        if not isinstance(environment, dict):
            errors.append("environment must be a mapping")
        else:
            setup = environment.get("setup")
            if setup is not None and not isinstance(setup, str):
                errors.append("environment.setup must be a shell snippet string")
            shell = environment.get("shell", "bash")
            if not isinstance(shell, str) or not shell:
                errors.append("environment.shell must be a non-empty string")
            preflight = environment.get("preflight")
            if preflight is not None and not isinstance(preflight, dict):
                errors.append("environment.preflight must be a mapping")

    return errors


def _command_script(task: dict[str, Any], command: str) -> tuple[str, str]:
    environment = task.get("environment") or {}
    shell = environment.get("shell", "bash")
    setup = (environment.get("setup") or "").strip()
    parts = ["set -euo pipefail"]
    if setup:
        parts.append(setup)
    parts.append(command)
    return shell, "\n".join(parts)


def run_contract_command(
    workspace: Path,
    task: dict[str, Any],
    section: str,
    *,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    config = task.get(section)
    if not isinstance(config, dict) or not config.get("command"):
        raise ContractError(f"task has no {section}.command")
    execution = task.get("execution") or {"transport": "local"}
    if execution.get("transport", "local") == "ssh":
        return ControlledSSH(task).run_contract(section, capture=capture)
    shell, script = _command_script(task, config["command"])
    return subprocess.run(
        [shell, "-lc", script],
        cwd=workspace,
        text=True,
        capture_output=capture,
        check=False,
    )


def run_preflight(workspace: Path, task: dict[str, Any]) -> None:
    environment = task.get("environment") or {}
    preflight = environment.get("preflight")
    if not preflight:
        return
    command = preflight.get("command")
    if not command:
        raise ContractError("environment.preflight.command is required")
    execution = task.get("execution") or {"transport": "local"}
    if execution.get("transport", "local") == "ssh":
        remote = ControlledSSH(task)
        shell, script = remote_command_script(task, command)
        result = remote.run_script(script, shell=shell, capture=True)
    else:
        shell, script = _command_script(task, command)
        result = subprocess.run(
            [shell, "-lc", script],
            cwd=workspace,
            text=True,
            capture_output=True,
            check=False,
        )
    if result.returncode:
        raise ContractError(
            f"environment preflight failed ({result.returncode}):\n{result.stderr}"
        )
    expected = preflight.get("expect_stdout_contains")
    if expected and expected not in result.stdout:
        raise ContractError(
            f"environment preflight output did not contain {expected!r}:\n{result.stdout}"
        )


def _iter_contract_paths(workspace: Path, paths: Iterable[str]) -> Iterable[Path]:
    for relative in paths:
        path = workspace / relative
        if path.is_file() or path.is_symlink():
            yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() or child.is_symlink():
                    yield child
        else:
            # A missing path is part of the snapshot too; creating it is a change.
            yield path


def _digest(path: Path) -> str:
    if not path.exists() and not path.is_symlink():
        return "<missing>"
    if path.is_symlink():
        return f"symlink:{os.readlink(path)}"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_forbidden(workspace: Path, task: dict[str, Any]) -> dict[str, str]:
    forbidden = task.get("constraints", {}).get("forbidden_paths", [])
    snapshot: dict[str, str] = {}
    for path in _iter_contract_paths(workspace, forbidden):
        relative = str(path.relative_to(workspace))
        snapshot[relative] = _digest(path)
    return snapshot


def compare_forbidden(
    workspace: Path, task: dict[str, Any], before: dict[str, str]
) -> list[str]:
    after = snapshot_forbidden(workspace, task)
    changed = sorted(
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    )
    return changed


def _load_candidate_ids(workspace: Path) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    errors: list[str] = []
    candidates_path = workspace / "runs/candidates.jsonl"
    if not candidates_path.is_file():
        return ids, errors
    seen: set[str] = set()
    with candidates_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"candidates.jsonl:{line_number}: {error}")
                continue
            for field in ("id", "hypothesis", "correctness", "decision"):
                if field not in record:
                    errors.append(
                        f"candidates.jsonl:{line_number}: missing field {field}"
                    )
            candidate_id = record.get("id")
            if candidate_id in seen:
                errors.append(f"candidates.jsonl:{line_number}: duplicate id {candidate_id}")
            if candidate_id:
                seen.add(candidate_id)
                ids.append(candidate_id)
    return ids, errors


def evaluate_goal(
    workspace: Path, task: dict[str, Any], candidate: str
) -> tuple[dict[str, Any] | None, list[str]]:
    """Recompute target attainment from the standard per-shape benchmark ledger."""
    path = workspace / "runs/benchmark.csv"
    errors: list[str] = []
    if not path.is_file():
        return None, errors
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "candidate",
            "shape",
            "metric_value",
            "baseline_value",
            "unit",
            "status",
        }
        missing = sorted(required_columns - set(reader.fieldnames or []))
        if missing:
            errors.append(
                "runs/benchmark.csv is not machine-verifiable; missing columns: "
                + ", ".join(missing)
            )
            return None, errors
        rows = [row for row in reader if row["candidate"] == candidate]

    configured_shapes = task.get("benchmark", {}).get("shapes") or []
    if configured_shapes:
        by_shape = {row["shape"]: row for row in rows}
        missing_shapes = [shape for shape in configured_shapes if shape not in by_shape]
        if missing_shapes:
            errors.append(f"best candidate lacks required shapes: {missing_shapes}")
            return None, errors
        rows = [by_shape[shape] for shape in configured_shapes]
    if not rows:
        errors.append(f"no benchmark rows found for best candidate {candidate!r}")
        return None, errors
    if any(row["status"] != "pass" for row in rows):
        errors.append(f"best candidate {candidate!r} has non-passing benchmark rows")
        return None, errors

    try:
        values = [float(row["metric_value"]) for row in rows]
        baselines = [float(row["baseline_value"]) for row in rows]
    except ValueError:
        errors.append("benchmark metric_value and baseline_value must be numeric")
        return None, errors
    if any(value <= 0 for value in values + baselines):
        errors.append("benchmark metric values must be positive")
        return None, errors

    ratios = [value / baseline for value, baseline in zip(values, baselines)]
    goal = task["goal"]
    aggregation = goal.get("aggregation", "sum_ratio")
    direction = goal.get("direction", "minimize")
    if aggregation == "sum_ratio":
        aggregate_ratio = sum(values) / sum(baselines)
    elif aggregation == "mean_ratio":
        aggregate_ratio = sum(ratios) / len(ratios)
    elif aggregation == "geomean_ratio":
        aggregate_ratio = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
    else:
        aggregate_ratio = max(ratios) if direction == "minimize" else min(ratios)

    relative_target = goal.get("relative_to_baseline")
    reached = None
    if relative_target is not None:
        reached = (
            aggregate_ratio <= relative_target
            if direction == "minimize"
            else aggregate_ratio >= relative_target
        )
    max_regression = goal.get("max_regression")
    regression_ok = True
    if max_regression is not None:
        regression_ok = (
            max(ratios) <= 1 + max_regression
            if direction == "minimize"
            else min(ratios) >= 1 - max_regression
        )
        if not regression_ok:
            reached = False
    return {
        "candidate": candidate,
        "aggregation": aggregation,
        "aggregate_ratio": aggregate_ratio,
        "per_shape_ratios": ratios,
        "regression_ok": regression_ok,
        "target_reached": reached,
    }, errors


def verify_evidence(
    workspace: Path,
    task: dict[str, Any] | None = None,
    final_result: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    required = (
        "docs/draft.md",
        "docs/plan.md",
        "runs/candidates.jsonl",
        "runs/benchmark.csv",
    )
    for relative in required:
        path = workspace / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty evidence file: {relative}")

    candidate_ids, candidate_errors = _load_candidate_ids(workspace)
    errors.extend(candidate_errors)
    if task is not None:
        maximum = task.get("stop", {}).get("max_candidates")
        attempted = [candidate for candidate in candidate_ids if candidate != "baseline"]
        if maximum is not None and len(attempted) > maximum:
            errors.append(
                f"candidate budget exceeded: {len(attempted)} attempted, maximum {maximum}"
            )

    benchmark_path = workspace / "runs/benchmark.csv"
    if benchmark_path.is_file():
        with benchmark_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                errors.append("runs/benchmark.csv has no header")
            elif "candidate" not in reader.fieldnames or "shape" not in reader.fieldnames:
                errors.append("runs/benchmark.csv must include candidate and shape columns")
            elif not any(True for _ in reader):
                errors.append("runs/benchmark.csv has no measurements")
    if final_result is not None and task is not None:
        best = final_result.get("best_candidate")
        if best and best not in candidate_ids:
            errors.append(f"final result names unknown best candidate: {best}")
        if best:
            evaluation, goal_errors = evaluate_goal(workspace, task, best)
            errors.extend(goal_errors)
            if evaluation is not None:
                reported = final_result.get("target_reached")
                measured = evaluation["target_reached"]
                if measured is not None and reported != measured:
                    errors.append(
                        "final target_reached disagrees with benchmark ledger: "
                        f"reported={reported}, measured={measured}"
                    )
    return errors


def load_final_result(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.is_file():
        return None, [f"Codex did not write structured final result: {path}"]
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return None, [f"invalid structured final result: {error}"]
    required = ("status", "best_candidate", "target_reached", "correctness", "summary")
    missing = [field for field in required if field not in result]
    if missing:
        return result, [f"structured final result missing fields: {missing}"]
    return result, []


def build_prompt(workspace: Path, task: dict[str, Any]) -> str:
    driver = KDA_ROOT / "scripts/kernelpilot.py"
    execution = task.get("execution") or {"transport": "local"}
    profile_id = task.get("target", {}).get("profile")
    profile_instruction = ""
    if profile_id:
        profile_instruction = f"""
Read backend profile {KDA_ROOT / 'backends' / f'{profile_id}.yaml'} and load
only the skills it routes for the current stage.
"""
    remote_instructions = ""
    if execution.get("transport", "local") == "ssh":
        remote_instructions = f"""
This is a split controller/executor task. Codex runs here; the target runs on
{execution['host']}:{execution['workspace']}. Never invoke ssh, scp, rsync, or
another remote shell directly. Inspect and edit only the local mirror at
{mirror_root(workspace)}. Synchronize declared implementation paths with:
- pull: python3 {driver} remote pull --workspace {workspace}
- push: python3 {driver} remote push --workspace {workspace}
The push is rejected if a path is not allow-listed or the remote file changed
since the last pull. Contracted commands below execute on the remote host.
"""
    return f"""Read the task workspace's AGENTS.md and .kernelpilot/task.yaml, then follow
the KDA workflow at {KDA_ROOT / 'workflows/kernel-optimization.yaml'} and the
optimization prompt at {KDA_ROOT / 'prompts/optimize-kernel.md'}.
{profile_instruction}
{remote_instructions}

Treat the workflow as the execution contract for this complete run. Start from
its initial stage, select each following stage from the declared transition,
and maintain runs/workflow-state.json and runs/stage-events.jsonl as required
by the optimization prompt.

Run every contracted command through the environment-aware wrapper:
- correctness: python3 {driver} run correctness --workspace {workspace}
- benchmark: python3 {driver} run benchmark --workspace {workspace}
- profile, only when configured and justified: python3 {driver} run profile --workspace {workspace}

Before editing, create docs/draft.md and docs/plan.md and preserve a rollback
copy of the implementation. Modify only constraints.allowed_paths. Never modify
constraints.forbidden_paths. Record every candidate and benchmark result,
including failures and regressions. Stop only when the target or a configured
stop condition is reached. Finish by writing docs/final-report.md and returning
the structured run result requested by the output schema.
"""


def run_codex(
    workspace: Path,
    task: dict[str, Any],
    codex_bin: str,
    sandbox: str,
    dry_run: bool,
) -> int:
    errors = validate_task(workspace, task)
    if errors:
        raise ContractError("invalid task contract:\n- " + "\n- ".join(errors))
    if not dry_run:
        run_preflight(workspace, task)
    for directory in EVIDENCE_DIRS:
        (workspace / directory).mkdir(parents=True, exist_ok=True)

    execution = task.get("execution") or {"transport": "local"}
    remote = execution.get("transport", "local") == "ssh"
    if remote and not dry_run:
        pull_allowed(workspace, task)
        before = {}
    elif remote:
        before = {}
    else:
        before = snapshot_forbidden(workspace, task)
    prompt = build_prompt(workspace, task)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    events_path = workspace / "runs" / f"codex-events-{timestamp}.jsonl"
    final_path = workspace / "runs" / f"codex-final-{timestamp}.json"
    state_path = workspace / "runs/kernelpilot-state.json"
    command = [
        codex_bin,
        "exec",
        "--sandbox",
        sandbox,
        "--json",
        "--output-schema",
        str(KDA_ROOT / "schemas/run-result.schema.json"),
        "--output-last-message",
        str(final_path),
        prompt,
    ]
    state = {
        "stage": "ready" if dry_run else "running",
        "started_at": _utc_now(),
        "task": task.get("task", {}).get("name"),
        "workspace": str(workspace),
        "events": str(events_path),
        "final": str(final_path),
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    if dry_run:
        print(json.dumps({"command": command, "prompt": prompt}, indent=2))
        return 0

    thread_id: str | None = None
    with events_path.open("w", encoding="utf-8") as events:
        process = subprocess.Popen(
            command,
            cwd=workspace,
            text=True,
            stdout=subprocess.PIPE,
            stderr=None,
        )
        assert process.stdout is not None
        for line in process.stdout:
            events.write(line)
            events.flush()
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "thread.started":
                thread_id = event.get("thread_id")
        returncode = process.wait()

    final_result, final_errors = load_final_result(final_path)
    evidence_errors = final_errors + verify_evidence(workspace, task, final_result)
    forbidden_changes = [] if remote else compare_forbidden(workspace, task, before)
    state.update(
        {
            "stage": "complete"
            if returncode == 0 and not evidence_errors and not forbidden_changes
            else "failed",
            "finished_at": _utc_now(),
            "codex_exit_code": returncode,
            "thread_id": thread_id,
            "evidence_errors": evidence_errors,
            "forbidden_changes": forbidden_changes,
            "result": final_result,
        }
    )
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    if forbidden_changes:
        print(
            "KernelPilot rejected the run because forbidden paths changed:\n- "
            + "\n- ".join(forbidden_changes),
            file=sys.stderr,
        )
    if evidence_errors:
        print(
            "KernelPilot evidence validation failed:\n- "
            + "\n- ".join(evidence_errors),
            file=sys.stderr,
        )
    return 0 if state["stage"] == "complete" else 1


def init_workspace(workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    destination = workspace / ".kernelpilot"
    destination.mkdir(parents=True, exist_ok=True)
    copies = (
        (KDA_ROOT / "templates/task.yaml", destination / "task.yaml"),
        (KDA_ROOT / "templates/operator-workspace/AGENTS.md", workspace / "AGENTS.md"),
    )
    for source, target in copies:
        if target.exists():
            raise ContractError(f"refusing to overwrite existing file: {target}")
        shutil.copyfile(source, target)
    print(f"initialized KernelPilot workspace: {workspace}")


def _workspace(value: str) -> Path:
    return Path(value).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kernelpilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("workspace", type=_workspace)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--workspace", type=_workspace, default=Path.cwd())
    validate_parser.add_argument("--task", type=Path, default=DEFAULT_TASK)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("section", choices=("correctness", "benchmark", "profile"))
    run_parser.add_argument("--workspace", type=_workspace, default=Path.cwd())
    run_parser.add_argument("--task", type=Path, default=DEFAULT_TASK)

    remote_parser = subparsers.add_parser("remote")
    remote_parser.add_argument("action", choices=("inspect", "pull", "push"))
    remote_parser.add_argument("--workspace", type=_workspace, default=Path.cwd())
    remote_parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    remote_parser.add_argument("--ssh-bin", default="ssh")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--workspace", type=_workspace, default=Path.cwd())

    optimize_parser = subparsers.add_parser("optimize")
    optimize_parser.add_argument("--workspace", type=_workspace, default=Path.cwd())
    optimize_parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    optimize_parser.add_argument(
        "--codex-bin", default=os.environ.get("KERNELPILOT_CODEX", "codex")
    )
    optimize_parser.add_argument(
        "--sandbox",
        choices=("workspace-write", "read-only"),
        default="workspace-write",
    )
    optimize_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            init_workspace(args.workspace)
            return 0
        if args.command == "verify":
            _, task = load_task(args.workspace, DEFAULT_TASK)
            errors = verify_evidence(args.workspace, task)
            if errors:
                raise ContractError("invalid evidence:\n- " + "\n- ".join(errors))
            print("evidence: valid")
            return 0

        _, task = load_task(args.workspace, args.task)
        if args.command == "remote":
            errors = validate_task(args.workspace, task)
            if errors:
                raise ContractError("invalid task contract:\n- " + "\n- ".join(errors))
            if args.action == "inspect":
                print(json.dumps(inspect_remote(task, args.ssh_bin), indent=2))
            elif args.action == "pull":
                hashes = pull_allowed(args.workspace, task, args.ssh_bin)
                print(json.dumps({"pulled": sorted(hashes)}, indent=2))
            else:
                changed = push_allowed(args.workspace, task, args.ssh_bin)
                print(json.dumps({"pushed": changed}, indent=2))
            return 0
        if args.command == "validate":
            errors = validate_task(args.workspace, task)
            if errors:
                raise ContractError("invalid task contract:\n- " + "\n- ".join(errors))
            run_preflight(args.workspace, task)
            print("task contract and environment: valid")
            return 0
        if args.command == "run":
            result = run_contract_command(args.workspace, task, args.section)
            return result.returncode
        if args.command == "optimize":
            return run_codex(
                args.workspace,
                task,
                args.codex_bin,
                args.sandbox,
                args.dry_run,
            )
    except (ContractError, SSHAdapterError, OSError, yaml.YAMLError) as error:
        print(f"kernelpilot: {error}", file=sys.stderr)
        return 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
