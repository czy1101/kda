---
name: iluvatar-ixkn-analysis
description: Verify Iluvatar V150 optimization mechanisms using attributable Triton cache IR, LLVM IR, native assembly, ixobjdump, and discovered ixKN counters. Use to distinguish generated instructions from dynamically executed paths and to explain compute, memory, shared-conflict, or occupancy behavior; measure speedups separately without profiling.
---

# V150 compiler and ixKN evidence

Use the selected backend's live tool-discovery record and the task's configured
profile command. The tool is `ixkn-cli` (ixKN, IXplorer Kernel Analyzer). A tool
declaration or executable probe is not proof that capture and export will work.
Read the knowledge base's “8. 编译管线与 IR”, “9.6 ixKN CLI”, and the relevant
[hardware topic](../Iluvatar/天数V150_硬件特性与计算搬运单元.md) when needed.

## Attribute low-level artifacts to the candidate

Capture the exact launched specialization rather than an arbitrary cache file.
Record candidate/source identity, cache key if available, signature, shape,
dtype, host rank/strides/offset alignment, masks/consumer, launch options,
environment switches, compiler/driver versions, and selected kernel name.
For auto-tuning, identify the chosen configuration and the measured launch.

Use the returned Triton compiled kernel's `asm` mapping when available:
`ttir`, `ttgir`, `llir`, `asm`, and binary entries actually exposed by the
installed backend. Preserve resource metadata (`n_regs`, `n_spills`, shared)
with those artifacts. Missing keys are an evidence limit, not permission to
fabricate PTX/SASS or parse another candidate's files.

If debugging lowering requires a fresh dump, the known installation supports
`TRITON_KERNEL_DUMP=1`, `TRITON_DUMP_DIR=<task-output-directory>`, and
`TRITON_ALWAYS_COMPILE=1`. Set these only for an attributable diagnostic run;
keep compilation and debug dumping outside official timing. Cached artifacts
are usable only with provenance tying them to the actual specialization.
For SME selection comparisons, use isolated processes and explicit settings.

Native V150 assembly uses `ml_*` / `sl_*`. For compiled binaries the verified
entry is `ixobjdump --dump-sass <binary-or-fatbin>` (also `--sass`), not a
presumed NVIDIA disassembler. Some addresses were omitted even on successful
decode; missing decoded text alone is not proof of absent instructions.

## Choose evidence for the claim

| Claim | Compiler evidence | Dynamic evidence when needed |
|---|---|---|
| TCU is selected and used | Operand types/layout and `ml_matrix_mad_*` | Matrix instruction counts / TCU metrics for the matched kernel |
| Packed FMA or native math is used | `ml_fma_v2f16/v2bf16`, `ml_exp/sqrt/rsq/rcp_*`; surrounding converts | Corresponding available instruction-class counters; coarse classes do not prove a specific mnemonic |
| SME replaces ordinary staging | SME calls/loads, consumer, `g2scnt` waits, LSA/SLB changes | Relevant instruction/memory/resource deltas; ordinary traffic counters alone do not uniquely identify SME |
| Alignment or cache policy helps | Actual load width/encoding, true addresses | Matched working-set cache/request/load data and unprofiled timing |
| Shared swizzle helps | Address mapping for both producer and consumer | Shared load/store conflicts separately, plus resources and ordinary timing |
| Resource tuning helps | VRF/SRF/shared, spills, stage buffers and waits | Occupancy/resource limits, warp/scheduler stalls, workload timing |
| Concurrency or transfer overlap occurs | Dependencies, buffer lifetime, wait placement | Compatible discovered timeline or a controlled overlap experiment; static async names are insufficient |

Do not demand every counter for every candidate. If a named metric is absent,
record that limitation and use supported evidence. A compiler artifact proves
generation; counters can show dynamic activity but may combine many instruction
types. Neither alone proves correctness or net benefit.

## Bounded ixKN collection

The CoreX 4.4.0 knowledge base verified these read-only discovery commands:
`ixkn-cli --version`, `--help`, `--list-sets`, `--list-sections`. Confirm current
flags before adapting a task's authorized profiling command. A known template:

```bash
ixkn-cli --section Memory,Instruction,SpeedOfLight \
  -k '<matched-kernel-name-or-regex>' -s 5 -c 1 \
  ./<contracted-program> <arguments>
```

Choose skip/count for the actual warmup and launch sequence; they are not fixed
requirements. Use only sections needed for the hypothesis: Memory, Instruction,
SpeedOfLight, Occupancy, LaunchStats, WarpStateStats, SchedulerStats, or
ComputeWorkload as exposed locally. Use `--csv` for attributable structured text
and `-o` for the raw report when supported. Keep raw reports and stderr, record
filters, replay policy, exit status, and export/parser failures. Dispatch through
KDA's controlled commands in the contracted environment; do not bypass transport
or add an unapproved remote command. Do not install or reconfigure tools.

Compare baseline and candidate with matched shape, dtype, launch attribution,
device conditions, warmup, section set, filtering, and replay policy. Collect
only correct kernels. Inspect a matched copy/control baseline when attributing
instruction classes: copy can have BFU counts, while measured `exp2` adds SFU
activity. Multi-Lane ALU includes indexing/address work, not just model FLOPs.

Global-memory instruction counts are not HBM bytes; L1/L2 hit rates cannot be
added. ixKN requests/throughput do not inherit NVIDIA sector/DRAM definitions.
Shared write conflicts alone do not show consumer read cost. Preserve field
names, units, unavailable values, and tool version; do not invent formulas.

Profiler replay changes execution time, cache state, and synchronization.
Profiled application Event averages and ixKN Duration are diagnostic; compute
performance gains from separate unprofiled contracted benchmarks. Report
generated path, dynamic observation, correctness, resource tradeoffs, and
measured benefit separately, with evidence paths in the task workspace.
