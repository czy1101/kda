"""Cross-check MUSA compiler metadata against executable LLVM intrinsic calls.

Read-only: this does not prove final machine-code execution or performance.
"""
import argparse
import json
from pathlib import Path
import re


def inspect_artifacts(metadata, llvm_ir):
    # Do not count declarations: an unused intrinsic does not establish usage.
    calls = sorted(set(re.findall(
        r"\bcall\b[^\n@]*@(llvm\.musa\.sqmma\.[\w.]+)\(", llvm_ir
    )))
    flag = metadata.get("uses_sqmma")
    return {
        "kernel": metadata.get("name"),
        "metadata_uses_sqmma": flag,
        "observed_sqmma_intrinsic_calls": calls,
        "metadata_ir_disagree": flag is False and bool(calls),
        "note": "Cross-check final machine code and effective compiler options before making a hardware-use claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    parser.add_argument("llvm_ir", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect_artifacts(
        json.loads(args.metadata.read_text()), args.llvm_ir.read_text()
    ), indent=2))


if __name__ == "__main__":
    main()
