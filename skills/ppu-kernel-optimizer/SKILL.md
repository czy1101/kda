---
name: ppu-kernel-optimizer
description: Diagnose and optimize Triton operators on T-Head Zhenwu 810E PPU by splitting end-to-end time into kernels, inspecting the compiled execution path, testing bottleneck hypotheses, and validating targeted changes. Use for PPU performance work even without PCM counters; not for unrelated CUDA tuning or functional debugging.
---

# T-Head PPU operator optimization

Read the task workspace's `.kernelpilot/task.yaml` first. It defines the implementation, reference, correctness test, formal benchmark, transport, allowed edits, and promotion criteria. KDA is the controller, not the operator repository. Preserve the original source and a recoverable best version before experiments.

## Bottleneck-first workflow

1. Confirm the controller, container, device mapping, active source hash, and authorized command route. Remote commands and source sync must use KDA controlled transport; do not bypass it with direct SSH or file copy. Use [environment and capabilities](references/environment-and-capabilities.md) for stack discovery. If the target device is unavailable, report the limitation instead of claiming a PPU result.
2. Read the real implementation, test, and formal benchmark. Lock its shape and dtype matrix, inputs, comparison provider, warmup, timing scope, and aggregation. Keep diagnostic timers separate from the formal score. See [benchmark methodology](references/benchmark-methodology.md).
3. Establish correctness and a complete unmodified baseline. Use `kernelpilot.py discover-tools` and `strategy`; a discovered profiler is not verified until capture and export produce matching kernel data. On PCM unavailable or not-ready failure, record it once and follow [the no-PCM route](references/no-pcm-workflow.md).
4. Build an execution map from an actual formal invocation: kernel launch order, names, input/output tensors, dependencies, launch gaps, and allocations inside the timing scope. Measure each stage on representative shapes from every distinct workload regime and retain the operator total. Rank stages by measured contribution and estimate the maximum end-to-end gain before choosing one. See the procedure in [bottleneck diagnosis](references/bottleneck-diagnosis.md).
5. For the selected kernel and exact launched specialization, inspect source, TTIR/TTGIR, LLVM IR, final ISA when available, and resource metadata. Trace the expensive loop or matrix operation through lowering; check actual Tensor Cell/AIU instructions, transfers and waits, layout conversions, spills, and grid size. Static code or fewer instructions alone does not establish a runtime bottleneck. Follow [compiler analysis](references/compiler-analysis.md).
6. Form competing explanations for the measured cost and run the smallest controlled experiment that distinguishes them. Hold the formal workload and numerical contract fixed; vary one causal factor at a time. When explaining a measured speedup, distinguish the effective software change from its proposed hardware mechanism. For grid/order changes, unavailable PCM counters, or a claim about scheduling, tail, or cache behavior, follow [causal mechanism verification](references/causal-mechanism-verification.md): isolate compiler differences, observe execution directly where feasible, and remove the hypothesized cause in a counterfactual. A microbenchmark or alternate CUDA implementation may test an algorithmic idea, but it does not establish PPU performance. If neither profiling nor a controlled experiment supports the cause, report the uncertainty and gather evidence before changing production code.
7. Only then implement a targeted candidate: name the exact kernel and computation changed, predict which kernel time, compiler feature, and end-to-end time should move, and state a falsifying outcome. Inspect the new compiled path, run correctness, compare that kernel and the full operator with its parent, then cover the required shape/dtype matrix. If the predicted effect does not occur, revise the diagnosis instead of stacking unrelated tweaks. Use [optimization strategies](references/optimization-strategies.md).
8. Preserve raw per-run samples and use interleaved parent/candidate runs when a small gain or regression may be noise. Record measured facts separately from hardware inferences and report every shape, including regressions. Archive the candidate and decision; promote to KDA `best` only under the task's current acceptance policy, and restore the approved best after rejected or superseded experiments. See [benchmark methodology](references/benchmark-methodology.md) and [evidence and reporting](references/evidence-and-reporting.md).

Use the maintained [810E architecture notes](references/ppu-architecture.md) first, then the [detailed device knowledge base](../THead/真武810E_知识库.md) for a relevant issue. Verify any uncertain capability on the actual runtime. Do not transfer NVIDIA or other PPU warp, Tensor Cell, shared-memory, cache, or register assumptions to this device.

For chained GEMMs, recurrent state, triangular solves, AIU descriptors, or explicit Shared staging, read [PPU matrix and transfer lessons](references/matrix-and-transfer-lessons.md). It covers observed C-to-A fragment reuse constraints, value-major/K-split dataflow, native-tile algebra, physical-pitch descriptors, and version-specific TLE lowering limitations. Use these as evidence-backed candidate patterns, not mandatory tile sizes or guaranteed speedups.

## Helpers

`S` below is this skill's absolute `scripts` directory and `W` is the task workspace. Helper measurements are diagnostic; they do not replace formal benchmark records or KDA promotion.

```text
python S/probe_environment.py --output W/profile/<kernel>/<run>/environment.json
python <kda>/scripts/kernelpilot.py discover-tools --workspace W
python S/probe_capabilities.py --workspace W --tool acu --kernel <actual-name-regex>
python <kda>/scripts/kernelpilot.py strategy --workspace W
python S/benchmark_kernel.py --adapter <authorized-adapter.py> --output W/profile/<kernel>/<run>/timing.json
python S/collect_compiler_artifacts.py --adapter <authorized-adapter.py> --output W/profile/<kernel>/<run>/compiler
python S/analyze_compiler_artifacts.py --manifest <compiler/manifest.json> --output <analysis.json>
python S/run_microbenchmarks.py --output <new-experiment-dir> --modes copy --sizes 4096 65536
python S/compare_experiments.py --baseline <timing.json> --candidate <timing.json> --output <comparison.json>
python S/generate_report.py --workspace W --facts <facts.json> --output <report.md>
```

Read each helper's `--help` before use. Microbenchmarks generate a plan by default and run only with `--execute`. Remote helper commands must be declared in the task contract. Do not install tools, change drivers or clocks, reset a device, or interfere with other workloads as part of routine optimization.
