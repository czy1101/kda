---
id: hygon-profiling-tools
kind: guide
status: confirmed
device: BW
arch: gfx936
scope: "Exact commands, artifacts, and failure modes for rocprofv2, rocprof v1, hipprof, rocsys, and the device probes on BW/gfx936"
last_validated: 2026-11-03
revalidate_after: 2027-02-28
---

# Hygon profiling tools: commands that actually work

This is the command-level reference for profiling on BW/gfx936. It records what
was executed and what came out, not what the tool documentation claims. The
workflow around these commands is in
[profiling-workflow.md](profiling-workflow.md); the hardware limits are in
[gfx936-validation.md](gfx936-validation.md).

Validated on 2026-11-03 in the `jiangrenkai-dev` container
(DTK `26.04-DCC2602-0317`) and on the bare-metal host
`bm-srwl-nj-zone3-d-bw1000-64g-2-8` (DTK `/opt/dtk`), 8×BW.

## 0. Environment, before any command

The DTK environment is not optional: several tools fail to load without it, and
`rocprofiler/bin` is never on `PATH` by default.

```bash
set +u
source /opt/dtk-26.04-DCC2602-0317/env.sh     # bare metal: source /opt/dtk/env.sh
set -u
export PATH=/opt/dtk-26.04-DCC2602-0317/rocprofiler/bin:$PATH   # rocprofv2/rocprof/rocsys
unset ROCR_VISIBLE_DEVICES DCU_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export HIP_VISIBLE_DEVICES=<device>           # the task contract owns this mask
```

Two verified traps:

- `env.sh` adds `/opt/dtk/bin`, `/opt/dtk/hip/bin`, and the libraries, but not
  `rocprofiler/bin`. Without the explicit `PATH` addition, `rocprofv2`,
  `rocprof`, and `rocsys` are "command not found" on the bare-metal host.
- `hipprof` aborts with `libperfetto.so.5: cannot open shared object file`
  unless `env.sh` was sourced first.

## 1. Which tool for which question

| Question | Tool | Verified |
|---|---|---|
| Which kernels run, for how long, with what resources? | `rocprofv2 --kernel-trace` | yes |
| HIP/HSA API timeline inside one process | `rocprofv2 --hip-trace` / `--sys-trace` | invocation accepted; use for API attribution |
| Numeric hardware counters (VMEM/VALU/LDS/SQ_*) | **`rocprof` v1** | yes — see the counter finding below |
| Launch gaps, synchronization, host/device overlap | `hipprof` | yes |
| Selected capture window of a long-running job | `rocsys` | controller only |
| Architecture, CU/LDS/Wave size | `rocminfo` | yes |
| Clocks, memory, utilization | `hy-smi` or `rocm-smi` | yes |
| Streaming-bandwidth calibration | `rocm-bandwidth-test` | present |
| Triton-level kernel profiling | `proton` / `triton.profiler` | **no — import fails on this image** |

## 2. `rocprofv2` — kernel traces

```bash
mkdir -p profile/trace
rocprofv2 --kernel-trace -d profile/trace -o kernels python workload.py
```

Verified artifact: `profile/trace/results_kernels.csv` with one row per dispatch:

```text
Dispatch_ID,GPU_ID,Queue_ID,Queue_Index,PID,TID,GRD,WGR,LDS,SCR,Arch_VGPR,ACCUM_VGPR,SGPR,Wave_Size,SIG,OBJ,Kernel_Name,Start_Timestamp,End_Timestamp,Correlation_ID
```

Useful columns: `Kernel_Name` (group by it), `GRD`/`WGR` (grid and workgroup),
`LDS`, `Arch_VGPR`/`SGPR`, `Wave_Size`, and the timestamps (compute duration
from the difference; do not assume a fixed clock).

Other modes accepted by the same binary: `--hip-trace`, `--hsa-trace`,
`--roctx-trace`, `--sys-trace`, `--list-counters`, `-i <pmc.txt>`,
`--plugin file|perfetto|att|ctf`, `-fi` (flush interval), `-tp` (trace period).
`--list-counters` works and lists gfx936 counters (for example `SQ_INSTS_VALU`,
`SQ_INSTS_MMOP`, `SQ_WAVES`), but a listable counter is not a capturable one.

**Counter capture with v2 is broken on the validated image.** `rocprofv2 -i
pmc.txt -d out -o counters <app>` exits 0, prints nothing, produces no file, and
does not even surface the app's stdout. Do not treat that as "zero counters".

## 3. `rocprof` v1 — hardware counters

The v1 path is the only one that produced numeric counters:

```bash
printf 'pmc: SQ_INSTS_VALU\n' > pmc.txt
rocprof -i pmc.txt -o counters.csv python workload.py
```

Verified artifact: `counters.csv` with one row per kernel and the counter as a
column:

```text
Index,KernelName,gpu-id,queue-id,queue-index,pid,tid,grd,wgr,lds,scr,arch_vgpr,accum_vgpr,sgpr,wave_size,sig,obj,SQ_INSTS_VALU
```

Notes:

- v1 replays the workload once per PMC group, so it is slower than a trace and
  changes timing; never use a v1 counter run as the latency measurement.
- The counter catalogue is large (`rocprof --list-basic` printed ~39k lines on
  this image). Use `--list-basic` to find names, then capture a small group.
- The input file syntax is `pmc: <name> <name> ...`, not the v2 line-per-counter
  format. Mixing them silently captures nothing.

## 4. `hipprof` — timelines and API statistics

```bash
hipprof -o profile/timeline.json --output-type 0 python workload.py
```

`--output-type 0` writes JSON (`1` = html, `2` = pftrace, the default). Verified
behavior: it prints HIP API statistics (calls, total/average ns, percentage) and
a HIPOPS kernel table, then writes the trace. The example workload produced 428
events in 18 s; `hipModuleLoadData` dominated because the framework JITs its
kernels at startup — expect that on the first run, not on a warm one.

Use it for: launch gaps, synchronization, host/device overlap, and the
first-run-versus-steady-state difference. It does not replace kernel-level
counters for memory or arithmetic analysis.

## 5. `rocsys` — capture control

`rocsys` does not profile by itself; it starts and stops a session around a
`rocprofv2` capture:

```bash
rocsys --session <name> start
rocsys --session <name> stop
rocsys --session-new <name> launch <app command>
```

Use it when the interesting window is a small part of a long run. The capture it
controls is still `rocprofv2`; the same counter limitation applies.

## 6. Device probes

```bash
rocminfo | grep -E 'Name:|gfx[0-9]+'    # agents, ISA target
hy-smi                                   # HCU temperature, power, VRAM%, utilization
rocm-smi --showproductname               # card series/vendor
rocm-bandwidth-test -t 2                 # D2D streaming copy (calibration anchor)
```

Record the device mask and the probe output next to every capture: a profile
taken with a different `HIP_VISIBLE_DEVICES` is not comparable.

## 7. Not available, and why it matters

- `rocprofv3`, `omniperf`, `omnitrace`, `ncu`, `nsys`, `perf`: not installed.
  The analysis path is the three tools above plus the device probes.
- `proton` exists at `/flagos/bin/proton`, but `import triton.profiler` fails
  with `cannot import name 'amd' from triton._C.libtriton` on the FlagGems
  Triton build in this image. Retry after the Triton build changes; until then
  use `rocprofv2` for kernel-level and `torch.profiler` for framework-level
  attribution.

## 8. Failure modes seen

| Symptom | Cause | Action |
|---|---|---|
| `rocprofv2: command not found` | `rocprofiler/bin` not on `PATH` | add it after sourcing `env.sh` |
| `hipprof: libperfetto.so.5: cannot open shared object file` | `env.sh` not sourced | source it before running |
| `rocprofv2 -i` exits 0 with no output | v2 counter path broken on this image | use `rocprof` v1 counters, or fall back to trace + HSACO |
| v1 capture produced no CSV | PMC file used the v2 syntax, or the workload failed | use `pmc: <name>` and keep the app's own exit code |
| Trace rows missing a kernel | capture started before the workload warmed up | warm up, then capture the exact benchmark iteration |
| Counters look implausible | v1 replays and changes timing/state | never measure latency from a v1 run |

## 9. What to keep as evidence

For each capture, preserve under the task workspace:

- the exact command, tool, and environment (`profile/<candidate-id>.md`);
- the raw artifact (`results_kernels.csv`, `counters.csv`, `timeline.json`);
- the device probe output and the device mask;
- one falsifiable hypothesis derived from the capture.

A capture that produced no parseable artifact is a capability limit, not a
performance result; record the failure and move to the next available tool.
