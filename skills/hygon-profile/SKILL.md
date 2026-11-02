---
name: hygon-profile
description: Route profiling work for Hygon gfx936/BW tasks to the centralized backend notes and the live-discovered profiler tools.
---

# Hygon profiling skill

For Hygon gfx936/BW profiling, first read:

1. `backends/hygon/note/INDEX.md` for the report and guide catalog.
2. `backends/hygon/note/METHODS.md` to route the observed bottleneck to the
   experiment that tests it.
3. `backends/hygon/note/profiling-workflow.md` for the profiling sequence,
   task-contract examples, trace/counter procedures, and evidence requirements.
4. `backends/hygon/note/gfx936-validation.md` for validated hardware/toolchain
   facts and limitations.
5. `backends/hygon/note/rejected/` before retrying an idea that may already have
   been measured and rejected.

During performance analysis, collect a usable profiler capture before making a
bottleneck claim or promoting a candidate. Default to a `rocprofv2` kernel
trace; use `rocprof` only after its task-local capture smoke test succeeds.
Run `rocminfo` and either `rocm-smi` or `hy-smi` to record architecture and
device state. Use `hipprof` for timeline questions and `rocsys` to control
selective captures of long-running workloads. Preserve failed capture errors.

Use only profiler tools whose probes succeed in the current contracted task
environment. The backend notes are reusable references; the task contract,
correctness results, and current benchmark evidence determine each decision.
