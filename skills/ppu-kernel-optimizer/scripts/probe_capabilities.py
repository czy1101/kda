"""Validate contracted capture/export; persist failure to KernelPilot strategy."""
import argparse
import csv
import io
import json
import math
import re
from pathlib import Path
from common import cli, kernelpilot, now, positive, write_json


def validate_export(text, fmt, kernel_pattern, name_column, value_column):
    if "no kernel in report" in text.lower():
        raise ValueError("空报告")
    if fmt == "json":
        data = json.loads(text)
        rows = data["kernels"]
    else:
        # asys emits progress/log preambles before its CSV header. Locate a
        # header by the explicitly requested column names, never by row values.
        lines = text.splitlines()
        header = next((i for i, line in enumerate(lines)
                       if name_column in next(csv.reader([line]), [])
                       and value_column in next(csv.reader([line]), [])), None)
        if header is None:
            raise ValueError("未找到指定 CSV 列；检查当前工具实际导出格式")
        rows = list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))
    matched = []
    for row in rows:
        if not re.search(kernel_pattern, str(row.get(name_column, ""))):
            continue
        value = float(row[value_column])
        if math.isfinite(value) and value > 0:
            matched.append(row)
    if not matched:
        raise ValueError("未找到匹配 Kernel 的有效正数记录；不能把空报告当成功")
    return matched


def verify(workspace, task, tool, fmt, pattern, name_column, value_column,
           timeout=60, retry=False):
    kp = kernelpilot()
    errors = kp.validate_task(workspace, task)
    if errors:
        raise ValueError("; ".join(errors))
    if (task.get("analysis") or {}).get("profiling") == "disabled":
        raise ValueError("任务禁用 profiling")
    discovery = kp.load_analysis_capabilities(workspace, task)
    if not discovery:
        raise ValueError("先通过 KernelPilot discover-tools 获取当前执行环境证据")
    record = next((t for t in discovery["tools"] if t["id"] == tool), None)
    if not record or not record["available"]:
        raise ValueError("工具未发现或不可运行")
    if record.get("functional_status") == "failed":
        if not retry:
            return {"status": "failed", "reason": "同环境已失败，未重试", "tool": tool}
        kp.record_analysis_function(workspace, task, tool, "untested",
                                    {"reason": "explicit user-requested retry"})
    config = kp._profile_tool_config(task, tool)
    if not config.get("export_command"):
        raise ValueError("缺少契约 export_command；禁止运行无可验证导出的采集")
    maximum = (task.get("stop") or {}).get("max_profile_runs")
    attempts = list((workspace / "profile/tool-validation").glob("*/*-candidate.stdout.txt"))
    if maximum is not None and len(attempts) >= maximum:
        raise ValueError("采集预算已耗尽；验证也计入 max_profile_runs")
    result = {"tool": tool, "tested_at": now(), "status": "failed",
              "environment_fingerprint": discovery["environment_fingerprint"],
              "kernel_pattern": pattern, "export_format": fmt,
              "name_column": name_column, "value_column": value_column,
              "commands": {k: config.get(k) for k in ("command", "candidate_command", "export_command")}}
    try:
        for variant in ("candidate", "export"):
            completed = kp.run_contract_command(workspace, task, "profile", capture=True,
                                               tool=tool, variant=variant, timeout=timeout)
            result[variant + "_exit_code"] = completed.returncode
            if completed.returncode:
                raise ValueError(f"{variant} 失败，退出码 {completed.returncode}")
            refreshed = kp.load_analysis_capabilities(workspace, task)
            if any(t["id"] == tool and t.get("functional_status") == "failed" for t in refreshed["tools"]):
                raise ValueError("采集/导出返回无效信息，见原始日志")
        rows = validate_export(completed.stdout, fmt, pattern, name_column, value_column)
        result.update(status="verified", matched_records=len(rows))
    except Exception as error:
        result["reason"] = str(error)
    kp.record_analysis_function(workspace, task, tool, result["status"], result)
    return result


def main():
    p = argparse.ArgumentParser(description="只执行任务契约中的 profile 命令；首次验证也消耗采集预算")
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--tool", choices=("acu", "asys"), required=True)
    p.add_argument("--format", choices=("json", "csv"), default="json")
    p.add_argument("--kernel", required=True)
    p.add_argument("--name-column", default="name")
    p.add_argument("--value-column", default="duration_us")
    p.add_argument("--timeout", type=positive, default=60)
    p.add_argument("--retry", action="store_true", help="仅在用户明确要求重试或环境已确认变化时使用")
    a = p.parse_args()
    kp = kernelpilot()
    _, task = kp.load_task(a.workspace, kp.DEFAULT_TASK)
    result = verify(a.workspace, task, a.tool, a.format, a.kernel, a.name_column,
                    a.value_column, a.timeout, a.retry)
    write_json(a.workspace / "profile/tool-validation" / (a.tool + "-validation.json"), result)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "verified" else 2


if __name__ == "__main__":
    raise SystemExit(cli(main))
