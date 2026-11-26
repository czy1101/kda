#!/usr/bin/env python3
"""Minimal, evidence-gated agent runner for KDA task workspaces.

Codex remains the default for existing setups. OpenCode is selected explicitly
with ``--agent opencode`` (or through ``KERNELPILOT_AGENT``). The two drivers
share the task contract, environment setup, evidence gates, path allowlists,
checkpoint/rollback and benchmark verification; only invocation and event
parsing differ.
"""

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
from pathlib import Path, PurePosixPath
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
    list_tle_wiki,
    mirror_root,
    pull_allowed,
    push_allowed,
    read_tle_wiki,
)

DEFAULT_TASK = Path(".kernelpilot/task.yaml")
AGENT_CHOICES = ("auto", "codex", "opencode")
OPENCODE_DEFAULT_AGENT = "build"
EVIDENCE_DIRS = ("docs", "runs", "outputs", "profile")
ANALYSIS_CAPABILITIES = Path("runs/analysis-capabilities.json")
ANALYSIS_STRATEGY = Path("runs/analysis-strategy.json")
TLE_ASSESSMENT = Path("runs/tle-assessment.json")
BOTTLENECK_SCREEN = Path("runs/bottleneck-screen.json")
CHECKPOINT_ROOT = Path(".kernelpilot/checkpoints")
SCREEN_LAYER_IDS = (
    "algorithm_numerics",
    "operator_dataflow",
    "kernel_implementation",
    "launch_config_autotune",
    "compiler_codegen",
    "runtime_system",
)
SCREEN_DISPOSITIONS = ("yes", "no", "not_applicable")
DEEP_DIVE_DIRECTIONS = (
    "algorithm",
    "dataflow",
    "kernel",
    "config",
    "compiler",
    "runtime",
    "none",
)
DEEP_DIVE_FIELDS = (
    "direction",
    "hypothesis",
    "control",
    "expected_change",
    "supports_if",
    "refutes_if",
    "next_action",
    "stop",
)
CHANGE_CLASSES = ("kernel", "config", "compiler", "runtime", "measurement", "none")
REQUIRED_TOP_LEVEL = (
    "version",
    "task",
    "target",
    "implementation",
    "correctness",
    "benchmark",
    "baseline",
    "goal",
)


class ContractError(ValueError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _agent_binary(
    agent: str,
    *,
    codex_bin: str | None = None,
    opencode_bin: str | None = None,
) -> str:
    if agent == "codex":
        return codex_bin or os.environ.get("KERNELPILOT_CODEX") or "codex"
    if agent == "opencode":
        return opencode_bin or os.environ.get("KERNELPILOT_OPENCODE") or "opencode"
    raise ContractError(f"unsupported agent: {agent!r}")


def _binary_available(binary: str) -> bool:
    if shutil.which(binary):
        return True
    return Path(binary).expanduser().is_file()


def select_agent(
    requested: str = "auto",
    *,
    codex_bin: str | None = None,
    opencode_bin: str | None = None,
    agent_bin: str | None = None,
    require_available: bool = False,
) -> dict[str, Any]:
    """Resolve the agent driver without silently changing the Codex default.

    ``auto`` honours ``KERNELPILOT_AGENT`` first, then prefers an installed
    Codex executable and finally an installed OpenCode executable. An explicit
    selection always wins, which keeps both paths testable and reproducible.
    """
    selected = (requested or os.environ.get("KERNELPILOT_AGENT") or "auto").strip().lower()
    if selected not in AGENT_CHOICES:
        raise ContractError(
            f"agent must be one of {', '.join(AGENT_CHOICES)}; got {requested!r}"
        )
    if selected == "auto":
        configured = (os.environ.get("KERNELPILOT_AGENT") or "").strip().lower()
        if configured in ("codex", "opencode"):
            selected = configured
        else:
            for candidate in ("codex", "opencode"):
                if _binary_available(
                    _agent_binary(
                        candidate,
                        codex_bin=codex_bin,
                        opencode_bin=opencode_bin,
                    )
                ):
                    selected = candidate
                    break
            else:
                raise ContractError(
                    "no supported agent executable found; install codex or opencode, "
                    "or pass --agent and --agent-bin"
                )
    binary = agent_bin or _agent_binary(
        selected, codex_bin=codex_bin, opencode_bin=opencode_bin
    )
    if require_available and not _binary_available(binary):
        raise ContractError(
            f"{selected} executable not found: {binary}; install it or pass --agent-bin"
        )
    return {"id": selected, "binary": binary}


def build_agent_command(
    agent: str,
    binary: str,
    prompt: str,
    *,
    final_path: Path,
    task_name: str,
    sandbox: str = "workspace-write",
    model: str | None = None,
    opencode_agent: str = OPENCODE_DEFAULT_AGENT,
    auto_approve: bool = False,
) -> list[str]:
    """Build the verified non-interactive command for each supported driver."""
    if agent == "codex":
        return [
            binary,
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
    if agent == "opencode":
        command = [
            binary,
            "run",
            "--standalone",
            "--format",
            "json",
            "--agent",
            opencode_agent,
            "--title",
            f"KDA {task_name}",
        ]
        if model:
            command.extend(["--model", model])
        if auto_approve:
            command.append("--auto")
        command.append(prompt)
        return command
    raise ContractError(f"unsupported agent: {agent!r}")


def _event_message(event: dict[str, Any]) -> str:
    for key in ("message", "error", "detail", "reason"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            for nested in ("message", "detail", "reason"):
                text = value.get(nested)
                if isinstance(text, str) and text.strip():
                    return text.strip()
    part = event.get("part")
    if isinstance(part, dict):
        for key in ("message", "error", "reason"):
            value = part.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return str(event.get("type", "agent error"))


def observe_agent_event(
    agent: str, event: dict[str, Any]
) -> tuple[str | None, str | None, str | None]:
    """Return ``(session_id, error_message, text_delta)`` for one JSONL event.

    Codex emits ``thread.started`` and ``turn.failed`` records. OpenCode emits
    the session id on every record and text in ``part.text``. Unknown records
    are ignored rather than guessed at.
    """
    session_id: str | None = None
    error: str | None = None
    text: str | None = None
    event_type = event.get("type")
    if agent == "codex":
        if event_type == "thread.started":
            thread_id = event.get("thread_id")
            session_id = thread_id if isinstance(thread_id, str) else None
        if event_type in ("error", "turn.failed"):
            error = _event_message(event)
    elif agent == "opencode":
        raw_session = event.get("sessionID")
        session_id = raw_session if isinstance(raw_session, str) else None
        if event_type == "text":
            part = event.get("part")
            if isinstance(part, dict):
                raw_text = part.get("text")
                text = raw_text if isinstance(raw_text, str) else None
        if event_type in ("error", "session.error"):
            error = _event_message(event)
        part = event.get("part")
        if isinstance(part, dict) and part.get("reason") in (
            "error",
            "cancelled",
            "aborted",
        ):
            error = _event_message(event)
    return session_id, error, text


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Find the first complete JSON object that looks like a KDA final result."""
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "status" in value and "best_candidate" in value:
            return value
    return None


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


def _profile_matches_target(profile: dict[str, Any], task: dict[str, Any]) -> bool:
    target = task.get("target") or {}
    match = profile.get("match") or {}
    if match.get("backend") != target.get("backend"):
        return False
    for field, target_field in (
        ("arches", "arch"),
        ("devices", "device"),
        ("languages", "language"),
    ):
        accepted = match.get(field) or []
        actual = target.get(target_field)
        if accepted and actual and actual not in accepted:
            return False
    return True


def _matching_backend_profiles(task: dict[str, Any]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for path in sorted((KDA_ROOT / "backends").glob("*.yaml")):
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(profile, dict) and _profile_matches_target(profile, task):
            matches.append(profile)
    return matches


def _backend_profile(task: dict[str, Any]) -> dict[str, Any]:
    profile_id = task.get("target", {}).get("profile")
    if profile_id:
        path = KDA_ROOT / "backends" / f"{profile_id}.yaml"
        if not path.is_file():
            raise ContractError(f"unknown target.profile: {profile_id}")
        profile = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(profile, dict):
            raise ContractError(f"backend profile {profile_id} must be a mapping")
        return profile
    matches = _matching_backend_profiles(task)
    if len(matches) > 1:
        ids = [profile.get("id") for profile in matches]
        raise ContractError(f"multiple backend profiles match target; set target.profile: {ids}")
    return matches[0] if matches else {}


def validate_backend_profile(task: dict[str, Any]) -> list[str]:
    requested_id = task.get("target", {}).get("profile")
    try:
        profile = _backend_profile(task)
    except (ContractError, yaml.YAMLError) as error:
        return [str(error)]
    if not profile:
        return []
    profile_id = profile.get("id")
    errors: list[str] = []
    if profile.get("version") != 1 or not profile_id:
        errors.append(f"backend profile {profile_id} has an invalid version or id")
    if requested_id and profile_id != requested_id:
        errors.append(f"backend profile id {profile_id!r} does not match target.profile")
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
    for name, extension in (profile.get("extensions") or {}).items():
        if not isinstance(extension, dict):
            errors.append(f"backend profile {profile_id} extension {name!r} must be a mapping")
            continue
        relative = extension.get("skill")
        if relative and not (KDA_ROOT / relative / "SKILL.md").is_file():
            errors.append(
                f"backend profile {profile_id} extension {name!r} references missing skill: {relative}"
            )
    return errors


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
        "backend_profile": (_backend_profile(task) or {}).get("id"),
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


def assess_tle(workspace: Path, task: dict[str, Any], ssh_bin: str = "ssh") -> dict[str, Any]:
    """Check the configured TLE Wiki and task-declared API probe without installing anything."""
    policy = (task.get("analysis") or {}).get("extensions", {}).get("tle", "disabled")
    config = (task.get("extension_config") or {}).get("tle") or {}
    wiki_path = config.get("wiki_path", "tle-wiki")
    result: dict[str, Any] = {
        "version": 1,
        "generated_at": _utc_now(),
        "environment_fingerprint": _analysis_environment_fingerprint(workspace, task),
        "policy": policy,
        "wiki_path": wiki_path,
        "wiki_available": False,
        "api_probe_configured": bool(config.get("api_probe_command")),
        "api_available": False,
        "status": "disabled" if policy == "disabled" else "unavailable",
    }
    if policy != "disabled":
        execution = task.get("execution") or {"transport": "local"}
        try:
            if execution.get("transport", "local") == "ssh":
                result["wiki_files"] = list_tle_wiki(task, ssh_bin=ssh_bin)
                result["wiki_available"] = True
            else:
                root = (workspace / wiki_path).resolve()
                try:
                    root.relative_to(workspace.resolve())
                except ValueError as error:
                    raise ContractError("extension_config.tle.wiki_path escapes workspace") from error
                if root.is_dir():
                    result["wiki_available"] = True
                    result["wiki_files"] = [
                        str(path.relative_to(root)) for path in sorted(root.rglob("*")) if path.is_file()
                    ]
                else:
                    result["reason"] = "configured tle-wiki directory is missing"
        except (SSHAdapterError, OSError) as error:
            result["reason"] = str(error)

        probe = config.get("api_probe_command")
        if result["wiki_available"] and probe:
            completed = _run_environment_probe(workspace, task, probe, ssh_bin)
            result["api_probe_exit_code"] = completed.returncode
            output = ((completed.stdout or "") + (completed.stderr or "")).strip()
            if output:
                result["api_probe_output"] = output[:4000]
            result["api_available"] = completed.returncode == 0
            if completed.returncode != 0:
                result["reason"] = "TLE API probe failed"
        elif result["wiki_available"] and not probe:
            result["reason"] = "extension_config.tle.api_probe_command is not configured"

        if result["wiki_available"] and result["api_available"]:
            result["status"] = "available"

    output_path = workspace / TLE_ASSESSMENT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def load_tle_assessment(
    workspace: Path, task: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    path = workspace / TLE_ASSESSMENT
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ContractError(f"invalid TLE assessment evidence: {error}") from error
    if not isinstance(value, dict):
        raise ContractError("TLE assessment evidence must be a JSON object")
    if task is not None:
        expected = _analysis_environment_fingerprint(workspace, task)
        if value.get("environment_fingerprint") != expected:
            raise ContractError("TLE assessment is stale; rerun assess-tle")
    return value


def select_analysis_strategy(
    task: dict[str, Any],
    discovery: dict[str, Any] | None = None,
    tle_assessment: dict[str, Any] | None = None,
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
    pending_extensions: list[str] = []
    skipped_extensions: dict[str, str] = {}
    for name, policy in requested_extensions.items():
        if policy == "disabled":
            continue
        extension = backend_extensions.get(name) or {}
        status = extension.get("status", "unavailable")
        if name == "tle" and status in ("available", "experimental"):
            assessment_status = (tle_assessment or {}).get("status")
            if assessment_status == "available":
                active_extensions.append(name)
            elif tle_assessment is None:
                pending_extensions.append(name)
            elif policy == "required":
                raise ContractError("TLE is required but its Wiki/API assessment did not pass")
            else:
                skipped_extensions[name] = (tle_assessment or {}).get(
                    "reason", "TLE Wiki/API assessment did not pass"
                )
        elif status in ("available", "experimental"):
            active_extensions.append(name)
        elif policy == "required":
            raise ContractError(f"analysis extension {name!r} is required but unavailable")
        else:
            skipped_extensions[name] = extension.get("reason", "not declared by backend")

    return {
        "backend_profile": profile.get("id"),
        "environment_fingerprint": (discovery or {}).get("environment_fingerprint"),
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
        "pending_extensions": pending_extensions,
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

    baseline = task.get("baseline") or {}
    if not isinstance(baseline.get("name"), str) or not baseline.get("name", "").strip():
        errors.append("baseline.name must be a non-empty comparator name")
    baseline_command = baseline.get("command")
    if baseline_command is not None and (
        not isinstance(baseline_command, str) or not baseline_command.strip()
    ):
        errors.append("baseline.command must be a non-empty string when configured")

    goal = task.get("goal", {})
    configured_targets = [
        field for field in ("relative_to_baseline", "target_value") if field in goal
    ]
    if len(configured_targets) != 1:
        errors.append("goal needs exactly one of relative_to_baseline or target_value")
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
                "bottleneck_screen",
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
    try:
        _allowed_checkpoint_paths(task)
    except ContractError as error:
        errors.append(str(error))
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

    extension_config = task.get("extension_config")
    if extension_config is not None:
        if not isinstance(extension_config, dict):
            errors.append("extension_config must be a mapping")
        else:
            tle = extension_config.get("tle") or {}
            if not isinstance(tle, dict):
                errors.append("extension_config.tle must be a mapping")
            else:
                wiki_path = tle.get("wiki_path", "tle-wiki")
                if not isinstance(wiki_path, str) or not wiki_path:
                    errors.append("extension_config.tle.wiki_path must be a non-empty string")
                else:
                    parsed = PurePosixPath(wiki_path)
                    if parsed.is_absolute() or ".." in parsed.parts:
                        errors.append("extension_config.tle.wiki_path must stay inside the task workspace")
                probe = tle.get("api_probe_command")
                if probe is not None and (not isinstance(probe, str) or not probe.strip()):
                    errors.append("extension_config.tle.api_probe_command must be a non-empty string")

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
            strategy_path = workspace / ANALYSIS_STRATEGY
            if strategy_path.is_file():
                strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
                expected = _analysis_environment_fingerprint(workspace, task)
                if strategy.get("environment_fingerprint") != expected:
                    raise ContractError("analysis strategy is stale; rerun strategy")
                tool = strategy.get("selected_profiler")
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
        shell, script = remote_command_script(task, config["command"])
        return ControlledSSH(task).run_script(script, shell=shell, capture=capture)
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


def _allowed_checkpoint_paths(task: dict[str, Any]) -> list[str]:
    values = task.get("constraints", {}).get("allowed_paths") or []
    paths: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ContractError("constraints.allowed_paths must contain non-empty strings")
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ContractError(f"allowed path must stay inside the workspace: {value}")
        paths.append(path.as_posix())
    if not paths:
        raise ContractError("constraints.allowed_paths cannot be empty")
    return paths


def _checkpoint_source_root(workspace: Path, task: dict[str, Any]) -> Path:
    execution = task.get("execution") or {"transport": "local"}
    return mirror_root(workspace) if execution.get("transport", "local") == "ssh" else workspace


def _copy_checkpoint_entry(source: Path, destination: Path) -> None:
    if source.is_symlink():
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(os.readlink(source), target_is_directory=source.is_dir())
    elif source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    elif source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    else:
        raise ContractError(f"cannot checkpoint missing allowed path: {source}")


def save_checkpoint(
    workspace: Path,
    task: dict[str, Any],
    name: str,
    candidate: str,
) -> dict[str, Any]:
    if name not in ("baseline", "best"):
        raise ContractError("checkpoint name must be baseline or best")
    source_root = _checkpoint_source_root(workspace, task)
    checkpoint_parent = workspace / CHECKPOINT_ROOT
    checkpoint_parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_parent / f".{name}.next"
    destination = checkpoint_parent / name
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    for relative in _allowed_checkpoint_paths(task):
        _copy_checkpoint_entry(source_root / relative, temporary / relative)
    metadata = {
        "version": 1,
        "name": name,
        "candidate": candidate,
        "saved_at": _utc_now(),
        "paths": _allowed_checkpoint_paths(task),
    }
    (temporary / "checkpoint.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    if destination.exists():
        shutil.rmtree(destination)
    temporary.rename(destination)
    return metadata


def restore_checkpoint(
    workspace: Path,
    task: dict[str, Any],
    name: str = "best",
) -> dict[str, Any]:
    checkpoint = workspace / CHECKPOINT_ROOT / name
    metadata_path = checkpoint / "checkpoint.json"
    if not metadata_path.is_file():
        raise ContractError(f"checkpoint does not exist: {name}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    destination_root = _checkpoint_source_root(workspace, task)
    for relative in _allowed_checkpoint_paths(task):
        source = checkpoint / relative
        destination = destination_root / relative
        if destination.is_symlink() or destination.is_file():
            destination.unlink()
        elif destination.is_dir():
            shutil.rmtree(destination)
        _copy_checkpoint_entry(source, destination)
    execution = task.get("execution") or {"transport": "local"}
    if execution.get("transport", "local") == "ssh":
        metadata["pushed"] = push_allowed(workspace, task)
    metadata["restored_at"] = _utc_now()
    return metadata
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


def _load_candidate_records(workspace: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    candidates_path = workspace / "runs/candidates.jsonl"
    if not candidates_path.is_file():
        return records, errors
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
            if not isinstance(record, dict):
                errors.append(f"candidates.jsonl:{line_number}: record must be an object")
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
            if record.get("correctness") not in ("pass", "fail", "not_run"):
                errors.append(f"candidates.jsonl:{line_number}: invalid correctness")
            decision = record.get("decision")
            if decision not in ("keep", "revise", "reject"):
                errors.append(f"candidates.jsonl:{line_number}: invalid decision")
            if decision in ("revise", "reject"):
                if not record.get("reason"):
                    errors.append(f"candidates.jsonl:{line_number}: rejected/revised candidate needs reason")
                if record.get("rollback") not in ("restored_best", "not_needed", "failed"):
                    errors.append(f"candidates.jsonl:{line_number}: rejected/revised candidate needs rollback")
            if decision == "keep" and record.get("correctness") != "pass":
                errors.append(f"candidates.jsonl:{line_number}: kept candidate must pass correctness")
            records.append(record)
    return records, errors


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
        aggregate_value = sum(values)
    elif aggregation == "mean_ratio":
        aggregate_ratio = sum(ratios) / len(ratios)
        aggregate_value = sum(values) / len(values)
    elif aggregation == "geomean_ratio":
        aggregate_ratio = math.exp(sum(math.log(ratio) for ratio in ratios) / len(ratios))
        aggregate_value = math.exp(sum(math.log(value) for value in values) / len(values))
    elif aggregation == "all":
        aggregate_ratio = max(ratios) if direction == "minimize" else min(ratios)
        aggregate_value = max(values) if direction == "minimize" else min(values)
    else:  # any
        aggregate_ratio = min(ratios) if direction == "minimize" else max(ratios)
        aggregate_value = min(values) if direction == "minimize" else max(values)

    relative_target = goal.get("relative_to_baseline")
    reached = None
    if relative_target is not None:
        reached = (
            aggregate_ratio <= relative_target
            if direction == "minimize"
            else aggregate_ratio >= relative_target
        )
    absolute_target = goal.get("target_value")
    if absolute_target is not None:
        if aggregation == "all":
            reached = all(
                value <= absolute_target if direction == "minimize" else value >= absolute_target
                for value in values
            )
        elif aggregation == "any":
            reached = any(
                value <= absolute_target if direction == "minimize" else value >= absolute_target
                for value in values
            )
        else:
            reached = (
                aggregate_value <= absolute_target
                if direction == "minimize"
                else aggregate_value >= absolute_target
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
        "aggregate_value": aggregate_value,
        "per_shape_ratios": ratios,
        "regression_ok": regression_ok,
        "target_reached": reached,
    }, errors


def load_bottleneck_screen(workspace: Path) -> dict[str, Any] | None:
    """Read runs/bottleneck-screen.json when the agent wrote one."""
    path = workspace / BOTTLENECK_SCREEN
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"unreadable bottleneck screen {BOTTLENECK_SCREEN}: {error}") from error


def validate_bottleneck_screen(record: Any) -> list[str]:
    """Check the structure of a two-step bottleneck screen.

    These checks are structural. They can show that all six layers were
    addressed, that each layer states a disposition, and that a selected
    deep-dive names a falsifiable experiment. They cannot judge whether the
    evidence is sufficient, whether the stated control is real, or whether a
    natural-language attribution is overreaching. Passing this validator is not
    evidence that a diagnosis is correct.
    """
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["bottleneck screen must be a JSON object"]
    if record.get("version") != 1:
        errors.append("bottleneck screen version must be 1")

    layers = record.get("layers")
    if not isinstance(layers, list) or not layers:
        return errors + ["bottleneck screen needs a non-empty layers list"]

    seen: list[str] = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            errors.append(f"layers[{index}] must be an object")
            continue
        layer_id = layer.get("id")
        if layer_id not in SCREEN_LAYER_IDS:
            errors.append(
                f"layers[{index}].id {layer_id!r} is not one of {', '.join(SCREEN_LAYER_IDS)}"
            )
            continue
        seen.append(layer_id)
        for field in ("evidence", "key_unknown", "decision_supported"):
            value = layer.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"layers[{index}].{field} must be a non-empty string")
        disposition = layer.get("continue")
        if disposition not in SCREEN_DISPOSITIONS:
            errors.append(
                f"layers[{index}].continue must be one of {', '.join(SCREEN_DISPOSITIONS)}"
            )
    duplicates = sorted({item for item in seen if seen.count(item) > 1})
    if duplicates:
        errors.append(f"duplicate layer ids: {', '.join(duplicates)}")
    missing = [item for item in SCREEN_LAYER_IDS if item not in seen]
    if missing:
        errors.append(f"missing layers: {', '.join(missing)}")

    deep_dives = record.get("deep_dives")
    if not isinstance(deep_dives, list):
        errors.append("bottleneck screen needs a deep_dives list")
        deep_dives = []
    if any(item.get("continue") == "yes" for item in layers if isinstance(item, dict)):
        if not deep_dives:
            errors.append(
                "at least one layer continues, so deep_dives must name the selected diagnosis"
            )
    for index, dive in enumerate(deep_dives):
        if not isinstance(dive, dict):
            errors.append(f"deep_dives[{index}] must be an object")
            continue
        for field in DEEP_DIVE_FIELDS:
            value = dive.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"deep_dives[{index}].{field} must be a non-empty string")
        if dive.get("direction") not in DEEP_DIVE_DIRECTIONS:
            errors.append(
                f"deep_dives[{index}].direction must be one of "
                f"{', '.join(DEEP_DIVE_DIRECTIONS)}"
            )

    change_class = record.get("change_class")
    if change_class is not None:
        if not isinstance(change_class, dict):
            errors.append("change_class must be an object")
        elif change_class.get("kind") not in CHANGE_CLASSES:
            errors.append(
                f"change_class.kind must be one of {', '.join(CHANGE_CLASSES)}"
            )
        else:
            evidence = change_class.get("evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                errors.append("change_class.evidence must be a non-empty string")
    return errors


def verify_evidence(
    workspace: Path,
    task: dict[str, Any] | None = None,
    final_result: dict[str, Any] | None = None,
) -> list[str]:
    errors: list[str] = []
    required = (
        "docs/draft.md",
        "docs/plan.md",
        "docs/final-report.md",
        "runs/workflow-state.json",
        "runs/stage-events.jsonl",
        "runs/candidates.jsonl",
        "runs/benchmark.csv",
    )
    for relative in required:
        path = workspace / relative
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty evidence file: {relative}")

    if task is not None:
        tle_policy = (
            task.get("extensions", {}).get("tle", {}).get("policy", "disabled")
        )
        if tle_policy != "disabled":
            try:
                load_tle_assessment(workspace, task)
            except ContractError as error:
                errors.append(f"invalid or missing TLE assessment evidence: {error}")

        analysis = task.get("analysis", {})
        profile = _backend_profile(task) or {}
        if (
            analysis.get("tool_discovery", "auto") != "disabled"
            and analysis.get("profiling", "auto") != "disabled"
            and profile.get("analysis_tools")
        ):
            capabilities_path = workspace / ANALYSIS_CAPABILITIES
            if not capabilities_path.is_file() or capabilities_path.stat().st_size == 0:
                errors.append(
                    "missing or empty analysis-tool discovery evidence: "
                    f"{ANALYSIS_CAPABILITIES}"
                )

        # The two-step diagnosis screen is opt-in so existing task workspaces
        # keep validating: absent file plus the default `auto` policy is not an
        # error. A present but malformed screen always is, because a malformed
        # record would otherwise pass as evidence.
        screen_policy = analysis.get("bottleneck_screen", "auto")
        if screen_policy != "disabled":
            try:
                screen = load_bottleneck_screen(workspace)
            except ContractError as error:
                errors.append(str(error))
            else:
                if screen is None:
                    if screen_policy == "required":
                        errors.append(
                            "analysis.bottleneck_screen is required but evidence is "
                            f"missing: {BOTTLENECK_SCREEN}"
                        )
                else:
                    errors.extend(
                        f"invalid bottleneck screen: {error}"
                        for error in validate_bottleneck_screen(screen)
                    )

    candidate_records, candidate_errors = _load_candidate_records(workspace)
    candidate_ids = [record.get("id") for record in candidate_records if record.get("id")]
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
            best_records = [record for record in candidate_records if record.get("id") == best]
            if best_records and best_records[-1].get("correctness") != "pass":
                errors.append(f"final best candidate {best!r} did not pass recorded correctness")
            if final_result.get("correctness") != "pass":
                errors.append("final result correctness must be pass for a selected best candidate")
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


def load_final_result(
    path: Path,
    *,
    fallback_text: str | None = None,
    agent_label: str = "agent",
) -> tuple[dict[str, Any] | None, list[str]]:
    if not path.is_file():
        if fallback_text:
            embedded = _extract_json_object(fallback_text)
            if embedded is not None:
                return embedded, []
        return None, [f"{agent_label} did not write structured final result: {path}"]
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        if fallback_text:
            embedded = _extract_json_object(fallback_text)
            if embedded is not None:
                return embedded, []
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


def build_prompt(
    workspace: Path,
    task: dict[str, Any],
    *,
    agent: str = "codex",
    final_result_path: Path | None = None,
) -> str:
    driver = KDA_ROOT / "scripts/kernelpilot.py"
    execution = task.get("execution") or {"transport": "local"}
    agent_label = "OpenCode" if agent == "opencode" else "Codex"
    knowledge_instruction = ""
    if task.get("target", {}).get("backend") == "gcu":
        knowledge_instruction = f"""
Read the Enflame knowledge index at {KDA_ROOT / 'knowledge/enflame/README.md'}
and only the cards relevant to this target, language and measured bottleneck.
Also read the S60 runbook at {KDA_ROOT / 'docs/enflame-s60-runbook.md'} for
device allocation, staged measurement, A/B, TLE, and knowledge-update rules.
Every diagnose stage must follow {KDA_ROOT / 'docs/bottleneck-localization.md'}:
screen the six layers, then select one deep dive instead of collecting every
artifact. Write the screen and the selected experiment to
{BOTTLENECK_SCREEN.as_posix()} before you spend a device run on it. When the
question is about generated code, lowering or a stage artifact, also read
{KDA_ROOT / 'docs/enflame-compiler-diagnostics.md'}.
Record applicable card IDs and environment differences in docs/plan.md.
Historical observations suggest hypotheses, not current compatibility or
performance guarantees. Revalidate in the contracted environment; knowledge
does not override tool discovery, skill routing, constraints or evidence gates.
Record new reusable Enflame mechanisms or failures in the knowledge cards,
catalog and source pointers with evidence and limits; mention updates in the
final report. If repository writes are unavailable, leave a proposed update
in task docs without changing frozen task files or bypassing permissions.
"""
    profile_id = (_backend_profile(task) or {}).get("id")
    profile_instruction = ""
    if profile_id:
        profile_instruction = f"""
Read backend profile {KDA_ROOT / 'backends' / f'{profile_id}.yaml'} and load
only the skills it routes for the current stage.
"""
    remote_instructions = ""
    if execution.get("transport", "local") == "ssh":
        remote_instructions = f"""
This is a split controller/executor task. {agent_label} runs here; the target runs on
{execution['host']}:{execution['workspace']}. Never invoke ssh, scp, rsync, or
another remote shell directly. Inspect and edit only the local mirror at
{mirror_root(workspace)}. Synchronize declared implementation paths with:
- pull: python3 {driver} remote pull --workspace {workspace}
- push: python3 {driver} remote push --workspace {workspace}
The push is rejected if a path is not allow-listed or the remote file changed
since the last pull. Contracted commands below execute on the remote host.
When TLE is enabled, inspect its remote Wiki only through:
- list: python3 {driver} remote list-wiki --workspace {workspace}
- read: python3 {driver} remote read-wiki --workspace {workspace} --path <file-inside-wiki>
"""
    runtime_instructions = ""
    if agent == "opencode":
        runtime_instructions = """
OpenCode runtime note: this is a non-interactive OpenCode session. Use the
available tools directly; do not launch an interactive editor and do not start a
nested `opencode` or `codex` session. OpenCode does not provide Codex's
filesystem sandbox. The task contract's `constraints.allowed_paths` and
`constraints.forbidden_paths` remain binding, and KernelPilot hashes forbidden
paths and restores the best checkpoint after the run.
"""
    final_instructions = ""
    if agent == "opencode":
        if final_result_path is None:
            raise ContractError("OpenCode runs require a structured final-result path")
        final_instructions = f"""
Before finishing, write one JSON object matching
{KDA_ROOT / 'schemas/run-result.schema.json'} to {final_result_path}.
This file is the structured final result; do not only print it in chat.
"""
    else:
        final_instructions = """
Finish by writing docs/final-report.md and returning the structured run result
requested by the output schema.
"""
    return f"""Read the task workspace's AGENTS.md and .kernelpilot/task.yaml, then follow
the KDA workflow at {KDA_ROOT / 'workflows/kernel-optimization.yaml'} and the
optimization prompt at {KDA_ROOT / 'prompts/optimize-kernel.md'}.
{profile_instruction}
{knowledge_instruction}
{remote_instructions}
{runtime_instructions}

Treat the workflow as the execution contract for this complete run. Start from
its initial stage, select each following stage from the declared transition,
and maintain runs/workflow-state.json and runs/stage-events.jsonl as required
by the optimization prompt.

Run every contracted command through the environment-aware wrapper:
- discover backend analysis tools: python3 {driver} discover-tools --workspace {workspace}
- assess TLE only when enabled: python3 {driver} assess-tle --workspace {workspace}
- correctness: python3 {driver} run correctness --workspace {workspace}
- baseline, when configured: python3 {driver} run baseline --workspace {workspace}
- benchmark: python3 {driver} run benchmark --workspace {workspace}
- inspect the selected route: python3 {driver} strategy --workspace {workspace}
- profile the selected tool, only when configured and justified: python3 {driver} run profile --workspace {workspace} --variant candidate
- profile a comparable reference when configured: python3 {driver} run profile --workspace {workspace} --variant reference
- index a local compiler-artifact directory, offline, no device: python3 {KDA_ROOT / 'scripts/compile_artifacts.py'} index --root <artifact-dir> --out runs/compile-artifacts.json
- compare two artifact captures by identity: python3 {KDA_ROOT / 'scripts/compile_artifacts.py'} compare --left <dir-or-manifest> --right <dir-or-manifest> --out runs/compile-artifact-diff.json
- save a correct improved candidate as the recoverable best: python3 {driver} checkpoint save-best --workspace {workspace} --candidate <candidate-id>
- restore the current best when needed: python3 {driver} checkpoint restore-best --workspace {workspace}

Before editing, create docs/draft.md and docs/plan.md and preserve a rollback
copy of the implementation. Modify only constraints.allowed_paths. Never modify
constraints.forbidden_paths. Record every candidate and benchmark result,
including failures and regressions. Stop only when the target or a configured
stop condition is reached.
{final_instructions}
"""


def run_agent(
    workspace: Path,
    task: dict[str, Any],
    *,
    agent: str,
    agent_bin: str,
    sandbox: str,
    dry_run: bool,
    model: str | None = None,
    opencode_agent: str = OPENCODE_DEFAULT_AGENT,
    auto_approve: bool = False,
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
    if not dry_run:
        save_checkpoint(workspace, task, "baseline", "baseline")
        save_checkpoint(workspace, task, "best", "baseline")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    events_path = workspace / "runs" / f"{agent}-events-{timestamp}.jsonl"
    final_path = workspace / "runs" / f"{agent}-final-{timestamp}.json"
    state_path = workspace / "runs/kernelpilot-state.json"
    prompt = build_prompt(
        workspace,
        task,
        agent=agent,
        final_result_path=final_path,
    )
    command = build_agent_command(
        agent,
        agent_bin,
        prompt,
        final_path=final_path,
        task_name=task.get("task", {}).get("name", "task"),
        sandbox=sandbox,
        model=model,
        opencode_agent=opencode_agent,
        auto_approve=auto_approve,
    )
    state: dict[str, Any] = {
        "stage": "ready" if dry_run else "running",
        "started_at": _utc_now(),
        "agent": agent,
        "agent_bin": agent_bin,
        "task": task.get("task", {}).get("name"),
        "workspace": str(workspace),
        "events": str(events_path),
        "final": str(final_path),
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    if dry_run:
        print(
            json.dumps(
                {
                    "agent": agent,
                    "command": command,
                    "prompt": prompt,
                    "final_result": str(final_path),
                },
                indent=2,
            )
        )
        return 0

    session_id: str | None = None
    event_errors: list[str] = []
    text_parts: list[str] = []
    event_count = 0
    returncode = 1
    restoration: dict[str, Any] | None = None
    restoration_errors: list[str] = []
    try:
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
                if not isinstance(event, dict):
                    continue
                event_count += 1
                observed_session, event_error, text_delta = observe_agent_event(
                    agent, event
                )
                if observed_session:
                    session_id = observed_session
                if event_error:
                    event_errors.append(event_error)
                if text_delta:
                    text_parts.append(text_delta)
            process.stdout.close()
            returncode = process.wait()
    finally:
        try:
            restoration = restore_checkpoint(workspace, task, "best")
        except (ContractError, SSHAdapterError, OSError, json.JSONDecodeError) as error:
            restoration_errors.append(f"automatic best-checkpoint restore failed: {error}")

    fallback_text = "\n".join(text_parts) if agent == "opencode" else None
    final_result, final_errors = load_final_result(
        final_path,
        fallback_text=fallback_text,
        agent_label=agent,
    )
    result_source: str | None = None
    if final_result is not None:
        if final_path.is_file():
            result_source = "final_file"
        elif fallback_text:
            final_path.write_text(
                json.dumps(final_result, indent=2) + "\n", encoding="utf-8"
            )
            result_source = "event_text"
    if final_result and restoration:
        reported_best = final_result.get("best_candidate")
        restored_best = restoration.get("candidate")
        if reported_best and reported_best != restored_best:
            restoration_errors.append(
                f"final best candidate {reported_best!r} does not match restored checkpoint {restored_best!r}"
            )
    event_error_messages = [
        f"{agent} event reported failure: {message}" for message in event_errors
    ]
    evidence_errors = (
        final_errors
        + event_error_messages
        + restoration_errors
        + verify_evidence(workspace, task, final_result)
    )
    forbidden_changes = [] if remote else compare_forbidden(workspace, task, before)
    state.update(
        {
            "stage": "complete"
            if returncode == 0 and not evidence_errors and not forbidden_changes
            else "failed",
            "finished_at": _utc_now(),
            "agent_exit_code": returncode,
            "session_id": session_id,
            "event_count": event_count,
            "event_errors": event_errors,
            "result_source": result_source,
            "evidence_errors": evidence_errors,
            "forbidden_changes": forbidden_changes,
            "restoration": restoration,
            "result": final_result,
        }
    )
    if agent == "codex":
        state["codex_exit_code"] = returncode
        state["thread_id"] = session_id
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


def run_codex(
    workspace: Path,
    task: dict[str, Any],
    codex_bin: str,
    sandbox: str,
    dry_run: bool,
) -> int:
    """Backward-compatible Codex-only entry point."""
    return run_agent(
        workspace,
        task,
        agent="codex",
        agent_bin=codex_bin,
        sandbox=sandbox,
        dry_run=dry_run,
    )


def init_workspace(workspace: Path, agent: str = "auto") -> None:
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
    if agent == "opencode":
        config_dir = workspace / ".opencode"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "opencode.jsonc"
        if config_path.exists():
            raise ContractError(f"refusing to overwrite existing file: {config_path}")
        config_path.write_text(
            json.dumps(
                {
                    "$schema": "https://opencode.ai/config.json",
                    "skills": [str(KDA_ROOT / "skills")],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"initialized KernelPilot workspace: {workspace}")


def _workspace(value: str) -> Path:
    return Path(value).expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kernelpilot")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("workspace", type=_workspace)
    init_parser.add_argument(
        "--agent",
        choices=AGENT_CHOICES,
        default="auto",
        help="Also write OpenCode project config when set to opencode.",
    )

    prompt_parser = subparsers.add_parser("prompt")
    prompt_parser.add_argument("--workspace", type=_workspace, default=Path.cwd())
    prompt_parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    prompt_parser.add_argument(
        "--agent", choices=("codex", "opencode"), default="opencode"
    )
    prompt_parser.add_argument("--final", type=Path)

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

    tle_parser = subparsers.add_parser("assess-tle")
    tle_parser.add_argument("--workspace", type=_workspace, default=Path.cwd())
    tle_parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    tle_parser.add_argument("--ssh-bin", default="ssh")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument(
        "section", choices=("correctness", "baseline", "benchmark", "profile")
    )
    run_parser.add_argument("--workspace", type=_workspace, default=Path.cwd())
    run_parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    run_parser.add_argument("--tool")
    run_parser.add_argument(
        "--variant", choices=("candidate", "reference", "export"), default="candidate"
    )

    remote_parser = subparsers.add_parser("remote")
    remote_parser.add_argument(
        "action", choices=("inspect", "pull", "push", "list-wiki", "read-wiki")
    )
    remote_parser.add_argument("--workspace", type=_workspace, default=Path.cwd())
    remote_parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    remote_parser.add_argument("--ssh-bin", default="ssh")
    remote_parser.add_argument("--path")

    checkpoint_parser = subparsers.add_parser("checkpoint")
    checkpoint_parser.add_argument("action", choices=("save-best", "restore-best"))
    checkpoint_parser.add_argument("--workspace", type=_workspace, default=Path.cwd())
    checkpoint_parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    checkpoint_parser.add_argument("--candidate")

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--workspace", type=_workspace, default=Path.cwd())

    optimize_parser = subparsers.add_parser("optimize")
    optimize_parser.add_argument("--workspace", type=_workspace, default=Path.cwd())
    optimize_parser.add_argument("--task", type=Path, default=DEFAULT_TASK)
    optimize_parser.add_argument(
        "--agent",
        choices=AGENT_CHOICES,
        default=os.environ.get("KERNELPILOT_AGENT", "auto"),
        help="Agent driver. auto prefers Codex, then OpenCode.",
    )
    optimize_parser.add_argument(
        "--agent-bin",
        help="Explicit executable for the selected agent.",
    )
    optimize_parser.add_argument(
        "--codex-bin",
        default=os.environ.get("KERNELPILOT_CODEX", "codex"),
        help="Codex executable (kept for existing scripts).",
    )
    optimize_parser.add_argument(
        "--opencode-bin",
        default=os.environ.get("KERNELPILOT_OPENCODE", "opencode"),
        help="OpenCode executable.",
    )
    optimize_parser.add_argument(
        "--model",
        default=os.environ.get("KERNELPILOT_OPENCODE_MODEL"),
        help="OpenCode model in provider/model form.",
    )
    optimize_parser.add_argument(
        "--opencode-agent",
        default=os.environ.get("KERNELPILOT_OPENCODE_AGENT", OPENCODE_DEFAULT_AGENT),
        help="OpenCode primary agent name (default: build).",
    )
    optimize_parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Pass --auto to OpenCode. Prefer narrow permission rules instead.",
    )
    optimize_parser.add_argument(
        "--sandbox",
        choices=("workspace-write", "read-only"),
        default="workspace-write",
        help="Codex sandbox mode; ignored by OpenCode.",
    )
    optimize_parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            init_workspace(args.workspace, args.agent)
            return 0
        if args.command == "verify":
            _, task = load_task(args.workspace, DEFAULT_TASK)
            errors = verify_evidence(args.workspace, task)
            if errors:
                raise ContractError("invalid evidence:\n- " + "\n- ".join(errors))
            print("evidence: valid")
            return 0

        _, task = load_task(args.workspace, args.task)
        if args.command == "prompt":
            final_path = args.final
            if final_path is None:
                final_path = (
                    args.workspace
                    / "runs"
                    / f"{args.agent}-final-manual.json"
                )
            elif not final_path.is_absolute():
                final_path = args.workspace / final_path
            prompt = build_prompt(
                args.workspace,
                task,
                agent=args.agent,
                final_result_path=final_path,
            )
            print(prompt, end="" if prompt.endswith("\n") else "\n")
            return 0
        if args.command == "strategy":
            errors = validate_task(args.workspace, task)
            if errors:
                raise ContractError("invalid task contract:\n- " + "\n- ".join(errors))
            discovery = load_analysis_capabilities(args.workspace, task)
            tle = load_tle_assessment(args.workspace, task)
            strategy = select_analysis_strategy(task, discovery, tle)
            output = args.workspace / ANALYSIS_STRATEGY
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(strategy, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(strategy, indent=2))
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
        if args.command == "assess-tle":
            errors = validate_task(args.workspace, task)
            if errors:
                raise ContractError("invalid task contract:\n- " + "\n- ".join(errors))
            assessment = assess_tle(args.workspace, task, args.ssh_bin)
            print(json.dumps(assessment, indent=2))
            required = (task.get("analysis") or {}).get("extensions", {}).get("tle") == "required"
            return 2 if required and assessment.get("status") != "available" else 0
        if args.command == "remote":
            errors = validate_task(args.workspace, task)
            if errors:
                raise ContractError("invalid task contract:\n- " + "\n- ".join(errors))
            if args.action == "inspect":
                print(json.dumps(inspect_remote(task, args.ssh_bin), indent=2))
            elif args.action == "pull":
                hashes = pull_allowed(args.workspace, task, args.ssh_bin)
                print(json.dumps({"pulled": sorted(hashes)}, indent=2))
            elif args.action == "push":
                changed = push_allowed(args.workspace, task, args.ssh_bin)
                print(json.dumps({"pushed": changed}, indent=2))
            elif args.action == "list-wiki":
                print(json.dumps({"files": list_tle_wiki(task, args.ssh_bin)}, indent=2))
            else:
                if not args.path:
                    raise ContractError("remote read-wiki requires --path")
                print(read_tle_wiki(task, args.path, args.ssh_bin), end="")
            return 0
        if args.command == "checkpoint":
            errors = validate_task(args.workspace, task)
            if errors:
                raise ContractError("invalid task contract:\n- " + "\n- ".join(errors))
            if args.action == "save-best":
                if not args.candidate:
                    raise ContractError("checkpoint save-best requires --candidate")
                result = save_checkpoint(args.workspace, task, "best", args.candidate)
            else:
                result = restore_checkpoint(args.workspace, task, "best")
            print(json.dumps(result, indent=2))
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
                capture=True,
                tool=args.tool,
                variant=args.variant,
            )
            if result.stdout:
                print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
            if result.stderr:
                print(
                    result.stderr,
                    end="" if result.stderr.endswith("\n") else "\n",
                    file=sys.stderr,
                )
            return result.returncode
        if args.command == "optimize":
            selection = select_agent(
                args.agent,
                codex_bin=args.codex_bin,
                opencode_bin=args.opencode_bin,
                agent_bin=args.agent_bin,
                require_available=not args.dry_run,
            )
            if args.model and selection["id"] != "opencode":
                raise ContractError("--model is only supported for OpenCode")
            return run_agent(
                args.workspace,
                task,
                agent=selection["id"],
                agent_bin=selection["binary"],
                sandbox=args.sandbox,
                dry_run=args.dry_run,
                model=args.model,
                opencode_agent=args.opencode_agent,
                auto_approve=args.auto_approve,
            )
    except (ContractError, SSHAdapterError, OSError, yaml.YAMLError) as error:
        print(f"kernelpilot: {error}", file=sys.stderr)
        return 2
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
