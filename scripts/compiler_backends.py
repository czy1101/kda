"""Backend resource semantics. Unknown fields remain unknown, never guessed."""

import hashlib
import re


def normalize(adapter, case, artifacts):
    metrics = dict(case.get("metrics", {}))
    raw = case.get("raw_resources", {})
    if adapter not in ("generic", "metax-triton"):
        raise ValueError("unsupported compiler adapter")
    if adapter == "generic":
        return metrics, []
    if metrics:
        raise ValueError(
            "metax-triton metrics must come from the adapter, not overrides"
        )

    def put(name, value, unit, scope, definition, source):
        if value is not None:
            metrics[name] = dict(
                value=value,
                unit=unit,
                scope=scope,
                definition=definition,
                provenance=source,
            )

    put(
        "registers",
        raw.get("n_regs"),
        "registers",
        "driver_function_attribute",
        "Runtime n_regs field; not occupancy or dynamic instruction count",
        "raw_resources.n_regs",
    )
    put(
        "shared_memory_bytes",
        raw.get("shared"),
        "B",
        "launch",
        "Triton launch shared-memory metadata",
        "raw_resources.shared",
    )
    source = artifacts.get("driver_source", b"").decode("utf-8", errors="replace")
    # Strip comments so prose containing example code does not qualify as proof.
    code = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.S)
    pattern = (
        r"mcFuncGetAttribute\s*\(\s*&n_spills\s*,\s*"
        r"MC_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES\s*,[^;]+;\s*"
        r"n_spills\s*/=\s*4\s*;"
    )
    notes = []
    if re.search(pattern, code) and raw.get("n_spills") is not None:
        value = raw["n_spills"]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("invalid raw n_spills field")
        proof = (
            "driver_source:sha256:"
            + hashlib.sha256(artifacts["driver_source"]).hexdigest()
        )
        put(
            "local_memory_bytes",
            value * 4,
            "B",
            "driver_function_attribute",
            "MC_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES reconstructed from n_spills * 4; not spill counts",
            proof,
        )
    else:
        notes.append(
            "local_memory_bytes unavailable: driver field semantics were not verified"
        )
    return metrics, notes
