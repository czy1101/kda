#!/usr/bin/env python3
"""Controlled SSH transport for a local KDA controller.

The optimization agent stays local.  This module exposes only contract-declared
commands and allow-listed file synchronization to an execution-only host.
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


class SSHAdapterError(RuntimeError):
    pass


_SAFE_HOST = re.compile(r"^[A-Za-z0-9_.@-]+$")
_SAFE_CONTAINER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9_.+/@-]+$")


def _relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or not _SAFE_RELATIVE.fullmatch(value):
        raise SSHAdapterError(f"unsafe repository-relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise SSHAdapterError(f"path must stay inside the remote workspace: {value}")
    return path.as_posix()


def _execution(task: dict[str, Any]) -> dict[str, Any]:
    execution = task.get("execution") or {"transport": "local"}
    if execution.get("transport", "local") != "ssh":
        raise SSHAdapterError("controlled SSH requires execution.transport: ssh")
    host = execution.get("host")
    workspace = execution.get("workspace")
    if not isinstance(host, str) or not _SAFE_HOST.fullmatch(host):
        raise SSHAdapterError("execution.host contains unsupported characters")
    if not isinstance(workspace, str) or not PurePosixPath(workspace).is_absolute():
        raise SSHAdapterError("execution.workspace must be an absolute POSIX path")
    if "\n" in workspace or "\r" in workspace:
        raise SSHAdapterError("execution.workspace must be a single line")
    container = execution.get("container")
    if container is not None and (
        not isinstance(container, str) or not _SAFE_CONTAINER.fullmatch(container)
    ):
        raise SSHAdapterError("execution.container contains unsupported characters")
    return execution


def _allowed_paths(task: dict[str, Any]) -> list[str]:
    values = task.get("constraints", {}).get("allowed_paths", [])
    if not values:
        raise SSHAdapterError("constraints.allowed_paths cannot be empty for SSH sync")
    return [_relative_path(value) for value in values]


def _tle_wiki_root(task: dict[str, Any]) -> str:
    policy = (task.get("analysis") or {}).get("extensions", {}).get("tle", "disabled")
    if policy == "disabled":
        raise SSHAdapterError("TLE is disabled for this task")
    configured = (task.get("extension_config") or {}).get("tle", {}).get(
        "wiki_path", "tle-wiki"
    )
    return _relative_path(configured)


def command_script(task: dict[str, Any], command: str) -> tuple[str, str]:
    execution = _execution(task)
    environment = task.get("environment") or {}
    shell = environment.get("shell", "bash")
    setup = (environment.get("setup") or "").strip()
    parts = ["set -euo pipefail"]
    if setup:
        parts.append(setup)
    parts.append(f"cd -- {shlex.quote(execution['workspace'])}")
    parts.append(command)
    return shell, "\n".join(parts)


@dataclass
class ControlledSSH:
    task: dict[str, Any]
    ssh_bin: str = "ssh"

    @property
    def execution(self) -> dict[str, Any]:
        return _execution(self.task)

    def _argv(self, script: str, shell: str | None = None) -> list[str]:
        selected_shell = shell or (self.task.get("environment") or {}).get("shell", "bash")
        remote_script = script
        container = self.execution.get("container")
        if container:
            # Keep setup, workspace changes, commands, and stdin-based file sync
            # inside the same container. `-i` is required by the allow-listed
            # push path, which streams the candidate file through SSH stdin.
            remote_script = shlex.join(
                ["docker", "exec", "-i", container, selected_shell, "-lc", script]
            )
        return [
            self.ssh_bin,
            self.execution["host"],
            selected_shell,
            "-lc",
            shlex.quote(remote_script),
        ]

    def run_script(
        self,
        script: str,
        *,
        shell: str | None = None,
        capture: bool = False,
        input_bytes: bytes | None = None,
        text: bool | None = None,
    ) -> subprocess.CompletedProcess:
        if text is None:
            text = input_bytes is None
        return subprocess.run(
            self._argv(script, shell),
            input=input_bytes,
            capture_output=capture,
            text=text,
            encoding="utf-8" if text else None,
            errors="replace" if text else None,
            check=False,
        )

    def run_contract(self, section: str, *, capture: bool = False) -> subprocess.CompletedProcess:
        config = self.task.get(section)
        if not isinstance(config, dict) or not config.get("command"):
            raise SSHAdapterError(f"task has no {section}.command")
        shell, script = command_script(self.task, config["command"])
        return self.run_script(script, shell=shell, capture=capture)

    def sha256(self, relative: str) -> str:
        relative = _relative_path(relative)
        workspace = shlex.quote(self.execution["workspace"])
        target = shlex.quote(relative)
        result = self.run_script(
            f"set -euo pipefail\ncd -- {workspace}\nsha256sum -- {target}",
            capture=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() if isinstance(result.stderr, str) else ""
            raise SSHAdapterError(f"cannot hash remote path {relative}: {stderr}")
        return result.stdout.split()[0]

    def fetch(self, relative: str, destination: Path) -> str:
        relative = _relative_path(relative)
        if relative not in _allowed_paths(self.task):
            raise SSHAdapterError(f"refusing to fetch undeclared path: {relative}")
        workspace = shlex.quote(self.execution["workspace"])
        target = shlex.quote(relative)
        result = self.run_script(
            f"set -euo pipefail\ncd -- {workspace}\ncat -- {target}",
            capture=True,
            text=False,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip() if isinstance(result.stderr, bytes) else result.stderr
            raise SSHAdapterError(f"cannot fetch remote path {relative}: {stderr}")
        data = result.stdout.encode() if isinstance(result.stdout, str) else result.stdout
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def push(self, relative: str, source: Path, expected_sha256: str) -> str:
        relative = _relative_path(relative)
        if relative not in _allowed_paths(self.task):
            raise SSHAdapterError(f"refusing to write undeclared path: {relative}")
        if not source.is_file():
            raise SSHAdapterError(f"local candidate does not exist: {source}")
        current = self.sha256(relative)
        if current != expected_sha256:
            raise SSHAdapterError(
                f"remote file changed since pull: {relative} "
                f"(expected {expected_sha256}, found {current})"
            )
        data = source.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        workspace = shlex.quote(self.execution["workspace"])
        target = shlex.quote(relative)
        parent = shlex.quote(str(PurePosixPath(relative).parent))
        script = "\n".join(
            (
                "set -euo pipefail",
                f"cd -- {workspace}",
                # POSIX `test` does not accept the `--` option separator;
                # the path is already shell-quoted and cannot begin with '-'
                # because it is a validated repository-relative path.
                f"test -d {parent}",
                "tmp=$(mktemp .kernelpilot-upload.XXXXXX)",
                "trap 'rm -f -- \"$tmp\"' EXIT",
                "cat > \"$tmp\"",
                f"test \"$(sha256sum -- \"$tmp\" | cut -d' ' -f1)\" = {shlex.quote(digest)}",
                f"chmod --reference={target} \"$tmp\"",
                f"mv -- \"$tmp\" {target}",
                "trap - EXIT",
            )
        )
        result = self.run_script(script, input_bytes=data, capture=True)
        if result.returncode != 0:
            stderr = result.stderr.decode(errors="replace") if isinstance(result.stderr, bytes) else result.stderr
            raise SSHAdapterError(f"remote write failed for {relative}: {stderr.strip()}")
        return digest


def mirror_root(control_workspace: Path) -> Path:
    return control_workspace / "remote-worktree"


def state_path(control_workspace: Path) -> Path:
    return control_workspace / ".kernelpilot/remote-state.json"


def pull_allowed(control_workspace: Path, task: dict[str, Any], ssh_bin: str = "ssh") -> dict[str, str]:
    adapter = ControlledSSH(task, ssh_bin=ssh_bin)
    hashes: dict[str, str] = {}
    for relative in _allowed_paths(task):
        destination = mirror_root(control_workspace) / relative
        hashes[relative] = adapter.fetch(relative, destination)
    state = {
        "host": adapter.execution["host"],
        "workspace": adapter.execution["workspace"],
        "container": adapter.execution.get("container"),
        "hashes": hashes,
    }
    output = state_path(control_workspace)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return hashes


def _load_state(control_workspace: Path, task: dict[str, Any]) -> dict[str, Any]:
    path = state_path(control_workspace)
    if not path.is_file():
        raise SSHAdapterError("run `kernelpilot remote pull` before pushing")
    state = json.loads(path.read_text(encoding="utf-8"))
    execution = _execution(task)
    if (
        state.get("host") != execution["host"]
        or state.get("workspace") != execution["workspace"]
        or state.get("container") != execution.get("container")
    ):
        raise SSHAdapterError("remote-state target does not match the task contract")
    return state


def push_allowed(control_workspace: Path, task: dict[str, Any], ssh_bin: str = "ssh") -> list[str]:
    adapter = ControlledSSH(task, ssh_bin=ssh_bin)
    state = _load_state(control_workspace, task)
    hashes = state.get("hashes", {})
    changed: list[str] = []
    for relative in _allowed_paths(task):
        source = mirror_root(control_workspace) / relative
        local_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        expected = hashes.get(relative)
        if not expected:
            raise SSHAdapterError(f"remote-state has no hash for {relative}")
        if local_hash == expected:
            continue
        hashes[relative] = adapter.push(relative, source, expected)
        changed.append(relative)
        state_path(control_workspace).write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
    return changed


def _forbidden_paths(task: dict[str, Any]) -> list[str]:
    values = (task.get("constraints") or {}).get("forbidden_paths") or []
    return [_relative_path(value) for value in values if isinstance(value, str)]


def hash_forbidden(task: dict[str, Any], ssh_bin: str = "ssh") -> dict[str, str]:
    """Hash contract-declared forbidden paths on the executor without copying them.

    A read-only counterpart to the local snapshot: files contribute their
    sha256, directories contribute a marker plus every file below them, and a
    missing path keeps the same ``<missing>`` sentinel the local snapshot uses
    so that creating it later is detected as a change.
    """
    paths = _forbidden_paths(task)
    if not paths:
        return {}
    adapter = ControlledSSH(task, ssh_bin=ssh_bin)
    workspace = shlex.quote(adapter.execution["workspace"])
    lines = ["set -euo pipefail", f"cd -- {workspace}"]
    for relative in paths:
        quoted = shlex.quote(relative)
        lines.append(
            f"if [ -f {quoted} ]; then "
            f"printf 'file\\t%s\\t%s\\n' {quoted} "
            f"\"$(sha256sum -- {quoted} | cut -d' ' -f1)\"; "
            f"elif [ -d {quoted} ]; then "
            f"printf 'dir\\t%s\\n' {quoted}; "
            f"find -P -- {quoted} -type f -print0 | sort -z | "
            f"while IFS= read -r -d '' entry; do "
            f"printf 'file\\t%s\\t%s\\n' \"$entry\" "
            f"\"$(sha256sum -- \"$entry\" | cut -d' ' -f1)\"; done; "
            f"else printf 'missing\\t%s\\n' {quoted}; fi"
        )
    result = adapter.run_script("\n".join(lines), capture=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() if isinstance(result.stderr, str) else ""
        raise SSHAdapterError(f"cannot hash remote forbidden paths: {stderr}")
    snapshot: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if parts[0] == "file" and len(parts) == 3:
            snapshot[parts[1]] = parts[2]
        elif parts[0] == "dir" and len(parts) == 2:
            snapshot[parts[1].rstrip("/") + "/"] = "dir"
        elif parts[0] == "missing" and len(parts) == 2:
            snapshot[parts[1]] = "<missing>"
    return snapshot


def inspect_remote(task: dict[str, Any], ssh_bin: str = "ssh") -> dict[str, Any]:
    adapter = ControlledSSH(task, ssh_bin=ssh_bin)
    environment = task.get("environment") or {}
    preflight = environment.get("preflight")
    result: dict[str, Any] = {
        "transport": "ssh",
        "host": adapter.execution["host"],
        "workspace": adapter.execution["workspace"],
        "container": adapter.execution.get("container"),
        "allowed_paths": _allowed_paths(task),
    }
    if preflight:
        shell, script = command_script(task, preflight["command"])
        completed = adapter.run_script(script, shell=shell, capture=True)
        result["preflight_exit_code"] = completed.returncode
        result["preflight_stdout"] = completed.stdout
        if completed.returncode != 0:
            raise SSHAdapterError(f"remote preflight failed: {completed.stderr}")
        expected = preflight.get("expect_stdout_contains")
        if expected and expected not in completed.stdout:
            raise SSHAdapterError(f"remote preflight output does not contain {expected!r}")
    return result


def list_tle_wiki(task: dict[str, Any], ssh_bin: str = "ssh") -> list[str]:
    """List regular files under the configured remote TLE Wiki without copying it."""
    adapter = ControlledSSH(task, ssh_bin=ssh_bin)
    root = _tle_wiki_root(task)
    workspace = shlex.quote(adapter.execution["workspace"])
    quoted_root = shlex.quote(root)
    script = (
        "set -euo pipefail\n"
        f"cd -- {workspace}\n"
        f"test -d {quoted_root}\n"
        f"find -P -- {quoted_root} -type f -print"
    )
    result = adapter.run_script(script, capture=True)
    if result.returncode != 0:
        raise SSHAdapterError(
            f"cannot list remote TLE Wiki {root}: {(result.stderr or '').strip()}"
        )
    prefix = f"{root}/"
    files: list[str] = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if value.startswith(prefix):
            files.append(value[len(prefix) :])
    return sorted(files)


def read_tle_wiki(task: dict[str, Any], relative: str, ssh_bin: str = "ssh") -> str:
    """Read one regular file whose resolved path remains inside the remote TLE Wiki."""
    adapter = ControlledSSH(task, ssh_bin=ssh_bin)
    root = _tle_wiki_root(task)
    child = _relative_path(relative)
    target = (PurePosixPath(root) / child).as_posix()
    workspace = shlex.quote(adapter.execution["workspace"])
    quoted_root = shlex.quote(root)
    quoted_target = shlex.quote(target)
    script = "\n".join(
        (
            "set -euo pipefail",
            f"cd -- {workspace}",
            f"root_real=$(realpath -- {quoted_root})",
            f"target_real=$(realpath -- {quoted_target})",
            'case "$target_real" in "$root_real"/*) ;; *) exit 13 ;; esac',
            'test -f "$target_real"',
            'cat -- "$target_real"',
        )
    )
    result = adapter.run_script(script, capture=True)
    if result.returncode != 0:
        raise SSHAdapterError(
            f"cannot read remote TLE Wiki file {relative}: {(result.stderr or '').strip()}"
        )
    return result.stdout
