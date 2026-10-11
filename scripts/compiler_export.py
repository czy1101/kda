"""Executor-side helpers injected by KernelPilot; no runtime installation required.

A task probe calls kda_compiler['emit'] with explicit input identities. A compiled
object must be the exact instance used for that input, not an unordered cache.
"""

import base64
import json
import os


def artifact(value):
    if isinstance(value, str):
        return {"encoding": "utf-8", "data": value}
    if isinstance(value, bytes):
        return {"encoding": "base64", "data": base64.b64encode(value).decode("ascii")}
    raise ValueError("artifact must be text or bytes")


def triton_case(workload, input_id, compiled, driver_source=None):
    """Export a compiled instance; local memory semantics are resolved separately."""
    meta = compiled.metadata
    metadata = meta._asdict() if hasattr(meta, "_asdict") else dict(meta)
    stages = {
        name: artifact(value)
        for name, value in compiled.asm.items()
        if isinstance(value, (str, bytes))
    }
    if driver_source is not None:
        stages["driver_source"] = artifact(driver_source)
    # Only serialize launch configuration, not opaque compiler/runtime objects.
    config = {
        key: metadata[key]
        for key in (
            "num_warps",
            "num_stages",
            "num_ctas",
            "arch",
            "pipeline",
            "maxnreg",
        )
        if key in metadata
    }
    return {
        "workload": workload,
        "input_id": input_id,
        "kernel": metadata.get("name", getattr(compiled, "name", "")),
        "config": config,
        "artifacts": stages,
        "raw_resources": {
            "n_regs": compiled.n_regs,
            "n_spills": compiled.n_spills,
            "shared": metadata.get("shared"),
        },
        "metrics": {},
    }


def emit(cases, compiler):
    """Print exactly one JSON envelope. Send probe diagnostics to stderr."""
    print(
        json.dumps(
            {
                "version": 1,
                "run_id": os.environ["KDA_COMPILE_RUN_ID"],
                "candidate": os.environ["KDA_COMPILE_CANDIDATE"],
                "compiler": compiler,
                "cases": cases,
            },
            allow_nan=False,
        )
    )
