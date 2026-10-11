"""Strict MetaX export normalization; profiling data never becomes benchmark data."""

import csv
import json
import math
import re
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


class ReportError(ValueError):
    pass


def timestamp(value, unit):
    if unit not in ("ns", "us", "ms", "s"):
        raise ReportError("explicit supported timestamp unit required")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReportError("timestamp/duration must be numeric")
    if isinstance(value, float) and (not math.isfinite(value) or abs(value) >= 2**53):
        raise ReportError("nonfinite or precision-losing float timestamp")
    number = (
        Decimal(str(value))
        * {"ns": 1, "us": 1000, "ms": 1000000, "s": 1000000000}[unit]
    )
    if number < 0 or number != number.to_integral_value():
        raise ReportError("negative or sub-nanosecond timestamp/duration")
    return int(number)


def identifier(value, field):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or str(value) == "":
        raise ReportError("missing/invalid " + field)
    return str(value)


def interval_union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return merged


def parse_trace(data, source, unit):
    if not isinstance(data, dict) or not isinstance(data.get("traceEvents"), list):
        raise ReportError("unknown mcTracer trace schema: " + source)
    kernels = []
    apis = []
    other = 0
    for event in data["traceEvents"]:
        if not isinstance(event, dict):
            raise ReportError("invalid trace event")
        if event.get("ph") != "X":
            continue
        args = event.get("args", {})
        if not isinstance(args, dict):
            raise ReportError("invalid event arguments")
        kernel_fields = ("device_id", "queue_id", "grid", "block")
        if ("grid" in args or "block" in args) and not all(
            k in args for k in kernel_fields
        ):
            raise ReportError("incomplete GPU kernel identity")
        is_kernel = all(k in args for k in kernel_fields)
        is_api = all(k in args for k in ("process_id", "thread_id", "co_id"))
        if not (is_kernel or is_api):
            other += 1
            continue
        start = timestamp(event.get("ts"), unit)
        duration = timestamp(event.get("dur"), unit)
        if not isinstance(event.get("name"), str) or not event["name"]:
            raise ReportError("missing event name")
        row = dict(
            source=source,
            name=event["name"],
            start_ns=start,
            end_ns=start + duration,
            duration_ns=duration,
            correlation_id=identifier(args.get("co_id"), "co_id"),
        )
        if is_kernel:
            row.update(
                device_id=identifier(args["device_id"], "device_id"),
                queue_id=identifier(args["queue_id"], "queue_id"),
            )
            if "submit_ts" in args:
                row["submit_ns"] = timestamp(args["submit_ts"], unit)
            kernels.append(row)
        else:
            row.update(
                process_id=identifier(args["process_id"], "process_id"),
                thread_id=identifier(args["thread_id"], "thread_id"),
            )
            apis.append(row)
    if not kernels:
        raise ReportError("trace contains no recognized GPU kernels: " + source)
    processes = {a["process_id"] for a in apis}
    matches = defaultdict(list)
    for a in apis:
        matches[a["correlation_id"]].append(a)
    for k in kernels:
        found = matches[k["correlation_id"]]
        k["api_link_status"] = "unavailable_or_ambiguous"
        # mcTracer kernel records omit OS process ID. Only a single-process
        # trace with one matching API can supply an unambiguous correlation.
        if len(processes) == 1 and len(found) == 1:
            a = found[0]
            k.update(
                process_id=a["process_id"],
                api_link_status="matched",
                api_name=a["name"],
                api_start_ns=a["start_ns"],
                api_end_ns=a["end_ns"],
                api_end_to_kernel_start_ns=k["start_ns"] - a["end_ns"],
            )
        if "submit_ns" in k:
            k["submit_to_start_ns"] = k["start_ns"] - k["submit_ns"]
    return kernels, apis, other


def summarize_timeline(kernels, apis, other):
    queues = defaultdict(list)
    devices = defaultdict(list)
    for k in kernels:
        queues[(k["source"], k["device_id"], k["queue_id"])].append(k)
        devices[(k["source"], k["device_id"])].append(k)
    gaps = []
    for (source, device, queue), values in sorted(queues.items()):
        intervals = interval_union([(k["start_ns"], k["end_ns"]) for k in values])
        for previous, next_ in zip(intervals, intervals[1:]):
            gaps.append(
                dict(
                    source=source,
                    device_id=device,
                    queue_id=queue,
                    start_ns=previous[1],
                    end_ns=next_[0],
                    duration_ns=next_[0] - previous[1],
                )
            )
    summaries = []
    for (source, device), values in sorted(devices.items()):
        intervals = interval_union([(k["start_ns"], k["end_ns"]) for k in values])
        summaries.append(
            dict(
                source=source,
                device_id=device,
                kernel_count=len(values),
                kernel_duration_sum_ns=sum(k["duration_ns"] for k in values),
                kernel_interval_union_ns=sum(b - a for a, b in intervals),
                kernel_envelope_ns=intervals[-1][1] - intervals[0][0],
            )
        )
    return dict(
        format="mcTracer-traceEvents",
        unit="ns",
        kernels=kernels,
        apis=apis,
        gaps=gaps,
        devices=summaries,
        ignored_complete_events=other,
        capabilities=["kernel_timeline", "same_queue_kernel_gaps"]
        + (
            ["api_correlation"]
            if any(k["api_link_status"] == "matched" for k in kernels)
            else []
        ),
        limitations=[
            "Kernel gaps are not necessarily device idle time; other engines/queues may run.",
            "API-end-to-kernel-start is signed and can include dependencies or queueing; it is not proven host delay.",
            "Do not sum overlapping kernel durations as device busy time.",
            "Capture clocks and instrumentation differ from the formal benchmark.",
        ],
    )


def parse_counters(paths, root):
    rows = []
    for path in paths:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ReportError("unknown counter schema")
        for section, entries in data.items():
            if not isinstance(entries, list):
                raise ReportError("unknown counter section")
            for entry in entries:
                if (
                    not isinstance(entry, dict)
                    or not isinstance(entry.get("name"), str)
                    or "value" not in entry
                ):
                    raise ReportError("unknown counter metric schema")
                raw = entry["value"]
                number = None
                unit = "unspecified"
                status = "unsupported_value"
                if (
                    isinstance(raw, (int, float))
                    and not isinstance(raw, bool)
                    and math.isfinite(raw)
                ):
                    number = raw
                    status = "available"
                elif isinstance(raw, str):
                    match = re.fullmatch(
                        r"\s*([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:\(([^()]+)\)|(%|byte))?\s*",
                        raw,
                    )
                    if match:
                        number = float(match[1].replace(",", ""))
                        unit = match[2] or match[3] or "unspecified"
                        if not math.isfinite(number):
                            raise ReportError("nonfinite counter")
                        status = "available"
                rows.append(
                    dict(
                        source=str(path.relative_to(root)),
                        scope=path.name.removesuffix(".txt.json"),
                        section=section,
                        metric=entry["name"],
                        value=number,
                        unit=unit,
                        raw_value=raw,
                        status=status,
                        description=entry.get("description", ""),
                    )
                )
    if not any(r["status"] == "available" for r in rows):
        raise ReportError("no valid exported counter metrics")
    return dict(
        format="mcProfiler-txt-json",
        metrics=rows,
        capabilities=["counter_metrics"],
        limitations=[
            "Counter scopes can aggregate launches/replay passes; not an execution timeline.",
            "Preserve exported units; cycles must not be converted to latency without clock and counter semantics.",
            "Unspecified units are not inferred from the metric name.",
        ],
    )


def select_range(kernels, apis, spec):
    """Select exactly one explicit API-delimited region per trace file."""
    ranges = []
    selected = []
    selected_apis = []
    for source in sorted({k["source"] for k in kernels}):
        events = sorted(
            [a for a in apis if a["source"] == source], key=lambda a: a["start_ns"]
        )
        starts = [a for a in events if a["name"] == spec["start_api"]]
        if len(starts) != 1:
            raise ReportError(
                "capture_range requires exactly one start marker per trace"
            )
        start = starts[0]
        stops = [
            a
            for a in events
            if a["name"] == spec["stop_api"]
            and a["process_id"] == start["process_id"]
            and a["start_ns"] >= start["end_ns"]
        ]
        if len(stops) != 1:
            raise ReportError("capture_range requires exactly one later stop marker")
        begin, end = start["end_ns"], stops[0]["start_ns"]
        if end <= begin:
            raise ReportError("empty capture_range")
        chosen = []
        for k in [k for k in kernels if k["source"] == source]:
            overlaps = k["start_ns"] < end and k["end_ns"] > begin
            submitted = (
                k.get("api_link_status") == "matched"
                and begin <= k["api_start_ns"]
                and k["api_end_ns"] <= end
            )
            if overlaps or submitted:
                if (
                    not submitted
                    or k.get("process_id") != start["process_id"]
                    or k["start_ns"] < begin
                    or k["end_ns"] > end
                ):
                    raise ReportError(
                        "capture_range has uncorrelated or boundary-crossing GPU work"
                    )
                chosen.append(k)
        if not chosen:
            raise ReportError("capture_range has no correlated kernels")
        selected.extend(chosen)
        selected_apis.extend(
            a
            for a in events
            if a["process_id"] == start["process_id"]
            and begin <= a["start_ns"]
            and a["end_ns"] <= end
        )
        ranges.append(
            dict(
                source=source,
                start_ns=begin,
                end_ns=end,
                kernel_count=len(chosen),
                start_api=spec["start_api"],
                stop_api=spec["stop_api"],
            )
        )
    return selected, selected_apis, ranges


def normalize(root, tool, unit=None, capture_range=None):
    root = Path(root)
    if tool == "mcprofiler":
        paths = sorted((root / "vendor-export").rglob("*.txt.json"))
        if not paths:
            raise ReportError("mcProfiler export contains no recognized metric reports")
        return parse_counters(paths, root)
    if tool != "mctracer":
        raise ReportError("unsupported profiler")
    kernels = []
    apis = []
    other = 0
    paths = sorted((root / "trace").rglob("*.json"))
    if not paths:
        raise ReportError("mcTracer produced no trace JSON (exit code is insufficient)")
    for path in paths:
        k, a, n = parse_trace(
            json.loads(path.read_text()), str(path.relative_to(root)), unit
        )
        kernels.extend(k)
        apis.extend(a)
        other += n
    ranges = []
    if capture_range:
        kernels, apis, ranges = select_range(kernels, apis, capture_range)
    result = summarize_timeline(kernels, apis, other)
    result["capture_ranges"] = ranges
    if ranges:
        result["capabilities"].append("scoped_kernel_timeline")
    return result


def write_tables(output, report):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "normalized.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for key in ("kernels", "apis", "gaps", "devices", "metrics"):
        rows = report.get(key, [])
        if not rows:
            continue
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (output / (key + ".csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
