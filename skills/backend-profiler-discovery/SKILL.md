---
name: backend-profiler-discovery
description: Discover and select performance-analysis tools for a kernel task from a backend allowlist and live environment probes. Use before profiling when the available chip vendor, profiler installation, permissions, or report capabilities may vary. Do not install tools or replace correctness and benchmark gates.
---

# Backend profiler discovery

Read the task contract and selected backend profile. Run only the profile's
declared read-only probes through KernelPilot so `environment.setup`, SSH
transport, and the target workspace are identical to benchmark execution.
Never infer remote availability from the controller machine.

Preserve `runs/analysis-capabilities.json`. Distinguish these states:

- discovered: the executable or module probe succeeded;
- usable: the required capture and export operation also succeeds;
- useful: its evidence can distinguish the current competing explanations.

Select a tool for the question, not merely the highest-priority installed
binary. Kernel-counter profilers suit execution-pipeline and memory diagnoses;
timeline profilers suit launches, synchronization, overlap, and host gaps;
framework profilers suit operator attribution; compiler/IR tools suit generated
code and lowering questions. Use task `preferred_tools` only as an explicit
override among compatible discovered tools.

Do not install packages, change drivers, enable privileged counters, alter
clocks, or reconfigure the host unless separately authorized. When a probe,
capture, export, permission check, or parser fails, record the exact failure and
try another compatible discovered capability. If none remains, continue with
benchmark, source, compiler, and available runtime evidence. Missing tooling is
an acceptable capability limit, not evidence of an optimization win or loss.

Run expensive profiling only when it can choose between concrete candidate
hypotheses. Preserve opaque reports, but base AI analysis on attributable,
structured exports whenever possible.
