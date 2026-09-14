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
ANALYSIS_CAPABILITIES = Path("runs/analysis-capabilities.json")
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
    for group in ("always", "when_discovering", "when_profiling", "when_researching"):
        for relative in skill_groups.get(group, []):
            skill_file = KDA_ROOT / relative / "SKILL.md"
            if not skill_file.is_file():
                errors.append(f"backend profile {profile_id} references missing skill: {relative}")
    analysis_tools = profile.get("analysis_tools") or []
    if not isinstance(analysis_tools, list):
        errors.append(f"backend profile {profile_id} analysis_tools must be a list")
    else:
        seen_tools: set[str] = set()
        for index, tool in enumerate(analysis_tools):
            if not isinstance(tool, dict):
                errors.append(f"backend profile {profile_id} analysis_tools[{index}] must be a mapping")
                continue
            tool_id = tool.get("id")
            if not isinstance(tool_id, str) or not tool_id:
                errors.append(f"backend profile {profile_id} analysis_tools[{index}].id is required")
            elif tool_id in seen_tools:
                errors.append(f"backend profile {profile_id} has duplicate analysis tool: {tool_id}")
            else:
                seen_tools.add(tool_id)
            if not isinstance(tool.get("probe"), str) or not tool.get("probe", "").strip():
                errors.append(f"backend profile {profile_id} analysis tool {tool_id!r} needs a probe")
            for relative in tool.get("skills") or []:
                skill_file = KDA_ROOT / relative / "SKILL.md"
                if not skill_file.is_file():
                    errors.append(
                        f"backend profile {profile_id} analysis tool {tool_id!r} "
                        f"references missing skill: {relative}"
                    )
    return errors


def _backend_profile(task: dict[str, Any]) -> dict[str, Any]:
    profile_id = task.get("target", {}).get("profile")
    if not profile_id:
        return {}
    path = KDA_ROOT / "backends" / f"{profile_id}.yaml"
    if not path.is_file():
        raise ContractError(f"unknown target.profile: {profile_id}")
    profile = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(profile, dict):
        raise ContractError(f"backend profile {profile_id} must be a mapping")
    return profile


def _analysis_tool_specs(profile: dict[str, Any]) -> list[dict[str, Any]]:
    configured = profile.get("analysis_tools") or []
    if configured:
        return [dict(tool) for tool in configured if isinstance(tool, dict)]
    capabilities = profile.get("capabilities") or {}
    return [
        {"id": tool, "kind": "profiler", "priority": 0, "capabilities": []}
        for tool in capabilities.get("profilers") or []
    ]


def _profile_tool_config(task: dict[str, Any], tool_id: str | None) -> dict[str, Any]:
    profile_task = task.get("profile") or {}
    config = {
        key: value
        for key, value in profile_task.items()
        if key not in ("tools", "discovery", "preferred_tools")
    }
    per_tool = profile_task.get("tools") or {}
    if tool_id and isinstance(per_tool, dict) and isinstance(per_tool.get(tool_id), dict):
        config.update(per_tool[tool_id])
    return config


def _rank_analysis_tools(
    specs: list[dict[str, Any]], preferred: list[str]
) -> list[dict[str, Any]]:
    preferred_rank = {name: index for index, name in enumerate(preferred)}
    return sorted(
        specs,
        key=lambda tool: (
            preferred_rank.get(tool.get("id"), len(preferred)),
            -int(tool.get("priority", 0)),
            str(tool.get("id", "")),
        ),
    )


def _run_environment_probe(
    workspace: Path,
    task: dict[str, Any],
    command: str,
    ssh_bin: str,
) -> subprocess.CompletedProcess[str]:
    execution = task.get("execution") or {"transport": "local"}
    if execution.get("transport", "local") == "ssh":
        shell, script = remote_command_script(task, command)
        return ControlledSSH(task, ssh_bin=ssh_bin).run_script(
            script, shell=shell, capture=True
        )
    shell, script = _command_script(task, command)
    return subprocess.run(
        [shell, "-lc", script],
        cwd=workspace,
        text=True,
        capture_output=True,
        check=False,
    )


def _analysis_environment_fingerprint(workspace: Path, task: dict[str, Any]) -> str:
    execution = task.get("execution") or {"transport": "local"}
    environment = task.get("environment") or {}
    target = task.get("target") or {}
    descriptor = {
        "transport": execution.get("transport", "local"),
        "host": execution.get("host"),
        "workspace": execution.get("workspace") or str(workspace.resolve()),
        "shell": environment.get("shell", "bash"),
        "setup": environment.get("setup", ""),
        "backend_profile": target.get("profile"),
        "backend": target.get("backend"),
        "device": target.get("device"),
        "arch": target.get("arch"),
    }
    encoded = json.dumps(descriptor, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def discover_analysis_tools(
    workspace: Path,
    task: dict[str, Any],
    ssh_bin: str = "ssh",
) -> dict[str, Any]:
    """Probe only backend-declared tools in the contracted target environment."""
    profile = _backend_profile(task)
    policies = task.get("analysis") or {}
    discovery_policy = policies.get("tool_discovery", "auto")
    profiling_policy = policies.get("profiling", "auto")
    specs = _analysis_tool_specs(profile)
    preferred = policies.get("preferred_tools") or []
    explicit = (task.get("profile") or {}).get("tool")
    if explicit and explicit != "auto":
        preferred = [explicit, *[name for name in preferred if name != explicit]]

    result: dict[str, Any] = {
        "version": 1,
        "generated_at": _utc_now(),
        "backend_profile": profile.get("id"),
        "transport": (task.get("execution") or {}).get("transport", "local"),
        "environment_fingerprint": _analysis_environment_fingerprint(workspace, task),
        "policy": discovery_policy,
        "selected": None,
        "tools": [],
    }
    if discovery_policy == "disabled" or profiling_policy == "disabled":
        result["status"] = "disabled"
        result["disabled_reason"] = (
            "analysis.profiling is disabled"
            if profiling_policy == "disabled"
            else "analysis.tool_discovery is disabled"
        )
    else:
        for spec in _rank_analysis_tools(specs, preferred):
            probe = spec.get("probe")
            if not isinstance(probe, str) or not probe.strip():
                continue
            completed = _run_environment_probe(workspace, task, probe, ssh_bin)
            available = completed.returncode == 0
            record: dict[str, Any] = {
                "id": spec.get("id"),
                "kind": spec.get("kind", "profiler"),
                "priority": int(spec.get("priority", 0)),
                "available": available,
                "probe_exit_code": completed.returncode,
                "capabilities": spec.get("capabilities") or [],
            }
            stderr = (completed.stderr or "").strip()
            if stderr:
                record["probe_error"] = stderr[:2000]
            version_command = spec.get("version_command")
            if available and isinstance(version_command, str) and version_command.strip():
                version = _run_environment_probe(workspace, task, version_command, ssh_bin)
                version_text = ((version.stdout or "") + (version.stderr or "")).strip()
                record["version_exit_code"] = version.returncode
                if version_text:
                    record["version"] = version_text[:2000]
            result["tools"].append(record)
        available_ids = [tool["id"] for tool in result["tools"] if tool["available"]]
        result["selected"] = available_ids[0] if available_ids else None
        result["status"] = "complete" if available_ids else "no_tool_available"

    output = workspace / ANALYSIS_CAPABILITIES
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def load_analysis_capabilities(
    workspace: Path, task: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    path = workspace / ANALYSIS_CAPABILITIES
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ContractError(f"invalid analysis capability evidence: {error}") from error
    if not isinstance(value, dict):
        raise ContractError("analysis capability evidence must be a JSON object")
    if task is not None:
        expected = _analysis_environment_fingerprint(workspace, task)
        if value.get("environment_fingerprint") != expected:
            raise ContractError(
                "analysis capability evidence belongs to a different or changed task environment; "
                "rerun discover-tools"
            )
    return value


def select_analysis_strategy(
    task: dict[str, Any], discovery: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Select analysis routes from policy, backend declarations, and live probes."""
    profile = _backend_profile(task)
    capabilities = profile.get("capabilities") or {}
    policies = task.get("analysis") or {}
    profile_task = task.get("profile") or {}
    specs = _analysis_tool_specs(profile)
    specs_by_id = {tool.get("id"): tool for tool in specs}
    discovery_policy = policies.get("tool_discovery", "auto")
    profiling_policy = policies.get("profiling", "auto")
    preferred = policies.get("preferred_tools") or profile_task.get("preferred_tools") or []
    explicit = profile_task.get("tool")

    discovery_pending = bool(
        specs
        and discovery_policy != "disabled"
        and profiling_policy != "disabled"
        and discovery is None
    )
    if discovery is not None:
        available_ids = [
            tool.get("id")
            for tool in discovery.get("tools") or []
            if isinstance(tool, dict) and tool.get("available")
        ]
    elif discovery_policy == "disabled":
        available_ids = [explicit] if explicit and explicit != "auto" else []
    elif not profile.get("analysis_tools"):
        available_ids = list(capabilities.get("profilers") or [])
    else:
        available_ids = []

    available_specs = [specs_by_id[name] for name in available_ids if name in specs_by_id]
    ranked = _rank_analysis_tools(available_specs, preferred)
    if explicit and explicit != "auto":
        ranked = [tool for tool in ranked if tool.get("id") == explicit]

    selected_spec = next(
        (
            tool
            for tool in ranked
            if _profile_tool_config(task, tool.get("id")).get("candidate_command")
            or _profile_tool_config(task, tool.get("id")).get("command")
        ),
        ranked[0] if ranked else None,
    )
    selected_profiler = selected_spec.get("id") if selected_spec else None
    selected_capabilities = set((selected_spec or {}).get("capabilities") or [])
    selected_config = _profile_tool_config(task, selected_profiler)

    candidate_command = selected_config.get("candidate_command") or selected_config.get("command")
    profiling_available = bool(selected_profiler and candidate_command)

    comparison_policy = policies.get("reference_profile_comparison", "auto")
    reference_available = bool(
        (
            "reference_comparison" in selected_capabilities
            or capabilities.get("reference_profile_comparison")
        )
        and selected_config.get("reference_command")
    )

    if profiling_policy == "required" and not profiling_available and not discovery_pending:
        raise ContractError("analysis.profiling is required but no backend profiler and candidate command are available")
    if comparison_policy == "required" and not reference_available and not discovery_pending:
        raise ContractError("reference profile comparison is required but unavailable or not configured")

    if discovery_pending:
        route = "discover_tools"
    elif profiling_policy == "disabled" or not selected_profiler:
        route = "benchmark_only"
    elif not candidate_command:
        route = "profile_tool_needs_configuration"
    elif comparison_policy != "disabled" and reference_available:
        route = "profile_compare"
    else:
        route = "profile_candidate"

    low_level_policy = policies.get("low_level_artifacts", "auto")
    low_level = capabilities.get("low_level_artifacts") or []
    if low_level_policy == "disabled":
        low_level = []
    if low_level_policy == "required" and not low_level:
        raise ContractError("low-level artifacts are required but the backend declares none")

    requested_extensions = policies.get("extensions") or {}
    backend_extensions = profile.get("extensions") or {}
    active_extensions: list[str] = []
    skipped_extensions: dict[str, str] = {}
    for name, policy in requested_extensions.items():
        if policy == "disabled":
            continue
        extension = backend_extensions.get(name) or {}
        status = extension.get("status", "unavailable")
        if status in ("available", "experimental"):
            active_extensions.append(name)
        elif policy == "required":
            raise ContractError(f"analysis extension {name!r} is required but unavailable")
        else:
            skipped_extensions[name] = extension.get("reason", "not declared by backend")

    return {
        "backend_profile": profile.get("id"),
        "route": route,
        "tool_discovery_policy": discovery_policy,
        "tool_discovery_status": (discovery or {}).get("status", "not_run"),
        "discovery_required": discovery_pending,
        "selected_profiler": selected_profiler,
        "profilers": [tool.get("id") for tool in ranked],
        "unavailable_profilers": [
            tool.get("id")
            for tool in (discovery or {}).get("tools") or []
            if isinstance(tool, dict) and not tool.get("available")
        ],
        "structured_profile_export": bool(
            "structured_export" in selected_capabilities
            or capabilities.get("structured_profile_export")
        ),
        "profile_export_configured": bool(selected_config.get("export_command")),
        "ai_profile_fallback": bool(
            policies.get(
                "ai_profile_fallback",
                capabilities.get("ai_profile_fallback", True),
            )
        ),
        "low_level_artifacts": low_level,
        "active_extensions": active_extensions,
        "skipped_extensions": skipped_extensions,
        "discovery_skills": (profile.get("skills") or {}).get("when_discovering", []),
        "profiling_skills": [
            *(profile.get("skills") or {}).get("when_profiling", []),
            *((selected_spec or {}).get("skills") or []),
        ],
    }


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
    if aggregation not in ("sum_ratio", "mean_ratio", "geomean_ratio", "all", "any"):
        errors.append(
            "goal.aggregation must be sum_ratio, mean_ratio, geomean_ratio, all, or any"
        )

    errors.extend(validate_backend_profile(task))

    analysis = task.get("analysis")
    if analysis is not None:
        if not isinstance(analysis, dict):
            errors.append("analysis must be a mapping")
        else:
            allowed_policies = {"auto", "disabled", "required"}
            for field in (
                "tool_discovery",
                "profiling",
                "reference_profile_comparison",
                "low_level_artifacts",
            ):
                if analysis.get(field, "auto") not in allowed_policies:
                    errors.append(f"analysis.{field} must be auto, disabled, or required")
            preferred_tools = analysis.get("preferred_tools") or []
            if not isinstance(preferred_tools, list) or not all(
                isinstance(tool, str) and tool for tool in preferred_tools
            ):
                errors.append("analysis.preferred_tools must contain non-empty strings")
            extensions = analysis.get("extensions") or {}
            if not isinstance(extensions, dict):
                errors.append("analysis.extensions must be a mapping")
            elif any(value not in allowed_policies for value in extensions.values()):
                errors.append("analysis extension policies must be auto, disabled, or required")
            if not errors:
                try:
                    select_analysis_strategy(task)
                except ContractError as error:
                    errors.append(str(error))

    constraints = task.get("constraints", {})
    allowed = constraints.get("allowed_paths", [])
    forbidden = constraints.get("forbidden_paths", [])
    if implementation_path and allowed and implementation_path not in allowed:
        errors.append("implementation.path must appear in constraints.allowed_paths")
    overlap = sorted(set(allowed) & set(forbidden))
    if overlap:
        errors.append(f"allowed_paths and forbidden_paths overlap: {overlap}")

    generalization = task.get("generalization")
    if generalization is not None:
        if not isinstance(generalization, dict):
            errors.append("generalization must be a mapping")
        else:
            allow_specialization = generalization.get("allow_shape_specialization", False)
            if not isinstance(allow_specialization, bool):
                errors.append("generalization.allow_shape_specialization must be boolean")
            guard_shapes = generalization.get("guard_shapes") or []
            if not isinstance(guard_shapes, list) or not all(
                isinstance(shape, str) and shape for shape in guard_shapes
            ):
                errors.append("generalization.guard_shapes must contain non-empty strings")

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
    tool: str | None = None,
    variant: str = "candidate",
) -> subprocess.CompletedProcess[str]:
    if section == "profile":
        if not tool:
            discovery = load_analysis_capabilities(workspace, task) or {}
            tool = discovery.get("selected")
        config = _profile_tool_config(task, tool)
        command_field = {
            "candidate": "candidate_command",
            "reference": "reference_command",
            "export": "export_command",
        }.get(variant)
        if not command_field:
            raise ContractError(f"unsupported profile variant: {variant}")
        command = config.get(command_field)
        if variant == "candidate" and not command:
            command = config.get("command")
        if not command:
            raise ContractError(
                f"task has no profile command for tool {tool!r}, variant {variant!r}"
            )
        config = {"command": command}
    else:
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
    elif aggregation == "all":
        aggregate_ratio = max(ratios) if direction == "minimize" else min(ratios)
    else:  # any
        aggregate_ratio = min(ratios) if direction == "minimize" else max(ratios)

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
        "comparator": task.get("baseline", {}).get("name", "baseline"),
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
    required = (
        "status",
        "best_candidate",
        "target_reached",
        "correctness",
        "summary",
        "stop_reason",
        "workflow_issues",
    )
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
- discover backend analysis tools: python3 {driver} discover-tools --workspace {workspace}
- correctness: python3 {driver} run correctness --workspace {workspace}
- benchmark: python3 {driver} run benchmark --workspace {workspace}
- inspect the selected route: python3 {driver} strategy --workspace {workspace}
- profile the selected tool, only when configured and justified: python3 {driver} run profile --workspace {workspace} --variant candidate
- profile a comparable reference when configured: python3 {driver} run profile --workspace {workspace} --variant reference

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

    strategy_parser = subparsers.add_parser("strategy")
    strategy_parser.add_argument("--workspace", type=_workspace, default=Path.cwd())
    strategy_parser.add_argument("--task", type=Path, default=DEFAULT_TASK)

    discovery_parser = subparsers.add_parser("discover-tools")
    discovery_parser.add_argument("--workspace", type=_workspace, default=Path.cwd())
    discovery_parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    discovery_parser.add_argument("--ssh-bin", default="ssh")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("section", choices=("correctness", "benchmark", "profile"))
    run_parser.add_argument("--workspace", type=_workspace, default=Path.cwd())
    run_parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    run_parser.add_argument("--tool")
    run_parser.add_argument(
        "--variant", choices=("candidate", "reference", "export"), default="candidate"
    )

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
        if args.command == "strategy":
            errors = validate_task(args.workspace, task)
            if errors:
                raise ContractError("invalid task contract:\n- " + "\n- ".join(errors))
            discovery = load_analysis_capabilities(args.workspace, task)
            print(json.dumps(select_analysis_strategy(task, discovery), indent=2))
            return 0
        if args.command == "discover-tools":
            errors = validate_task(args.workspace, task)
            if errors:
                raise ContractError("invalid task contract:\n- " + "\n- ".join(errors))
            print(
                json.dumps(
                    discover_analysis_tools(args.workspace, task, args.ssh_bin), indent=2
                )
            )
            return 0
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
            result = run_contract_command(
                args.workspace,
                task,
                args.section,
                tool=args.tool,
                variant=args.variant,
            )
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
