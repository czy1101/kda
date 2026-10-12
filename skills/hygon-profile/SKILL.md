---
name: hygon-profile
description: Profile Triton or HIP operators on Hygon gfx936/BW devices with rocprofv2 and hipprof, then turn structured traces into an evidence-backed optimization hypothesis. Use only after correctness and a reproducible baseline exist.
---

# Hygon gfx936 profiling

Read the task contract and `backends/hygon-gfx936.yaml` first. This skill
describes the profiling route; it does not replace the correctness command,
benchmark contract, or candidate promotion rules.

## Tool roles

Use the tools for the question they can answer:

| Question | Tool | Evidence |
|---|---|---|
| Which kernels launched, how long, and with what grid/resources? | `rocprofv2 --kernel-trace` | `results_kernels.csv` |
| Which hardware counters are available or useful? | `rocprofv2 --list-counters`, then a small `-i` counter run | counter CSV or explicit unavailable result |
| Are launch gaps, synchronization, or host/device overlap limiting? | `hipprof --hip-trace --stats` | timeline/statistics output |
| Is the device in the expected state? | `rocm-smi`/`hy-smi`, `rocminfo` | preflight text; not a kernel-performance claim |

Do not treat tool discovery as proof that a capture is usable. A tool is
usable only after its capture and structured export complete for the target
workload.

## Environment and device preflight

The task's `environment.setup` should activate the DTK runtime and make the
device selection explicit. For the current BW/gfx936 image, the setup is
equivalent to:

```bash
source /opt/dtk-26.04-DCC2602-0317/env.sh
export KDA_DTK_ROOT=/opt/dtk-26.04-DCC2602-0317
export PATH="$KDA_DTK_ROOT/rocprofiler/bin:$KDA_DTK_ROOT/hip/bin:$KDA_DTK_ROOT/bin:$PATH"
export HIP_VISIBLE_DEVICES=6
unset ROCR_VISIBLE_DEVICES DCU_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
```

Use the logical device exposed after `HIP_VISIBLE_DEVICES` is applied. Record
the following before profiling:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("device_count", torch.cuda.device_count())
print("device", torch.cuda.get_device_name(0))
PY
rocm-smi --showproductname --showmeminfo vram --showuse --showclocks
# `hy-smi` is the native Hygon alias on images where rocm-smi is only a wrapper.
hy-smi --showproductname --showmeminfo vram --showuse --showclocks
rocminfo | grep -m1 -E 'Name: *gfx|Marketing Name:'
```

If `rocm-smi` is not present, record that capability as unavailable and
continue; do not install or reconfigure a profiler during an optimization
run.

## Kernel trace procedure

1. Run the correctness command and the baseline benchmark.
2. Warm up the exact candidate workload outside the capture so Triton
   compilation and allocator initialization are not attributed to the kernel.
3. Capture one deterministic workload with `rocprofv2`:

```bash
work=profile/hygon-rocprofv2/candidate
rm -rf "$work"
mkdir -p "$work"
rocprofv2 --kernel-trace -d "$work/trace" -o kernels \
  python path/to/workload.py
```

The workload must use the same shape, dtype, layout, warmup, repetitions,
device mask, and stream policy as the benchmark. Preserve the raw
`results_kernels.csv` under `profile/`; do not rely only on a pasted terminal
summary.

The key columns are `Kernel_Name`, `GRD`, `WGR`, `LDS`, `Arch_VGPR`, `SGPR`,
`Start_Timestamp`, and `End_Timestamp`. Calculate duration as
`End_Timestamp - Start_Timestamp` and group by `Kernel_Name`. For a Triton
kernel, record whether the candidate is limited by launch count, LDS/register
resource pressure, or long memory/compute duration. Do not infer occupancy
from a single resource column without a matching counter or compiler report.

## Counter procedure

Counter names are DTK/image dependent. Discover them first:

```bash
rocprofv2 --list-counters
```

Select only counters that are listed on the target image. Run a minimal
counter capture with a one-line input file:

```bash
printf '%s\n' <COUNTER_NAME> > profile/hygon-rocprofv2/counters.txt
rocprofv2 -i profile/hygon-rocprofv2/counters.txt \
  --plugin file -d profile/hygon-rocprofv2/counter-trace -o counters \
  python path/to/workload.py
```

If a counter run fails while kernel trace succeeds, keep the trace evidence,
record the exact counter error, and do not label the missing metric as zero.

## Timeline fallback

Use `hipprof` when the hypothesis concerns host gaps, synchronization, or
overlap rather than instruction-level kernel resources:

```bash
work=profile/hygon-hipprof/candidate
rm -rf "$work"
mkdir -p "$work"
hipprof --hip-trace --stats -o "$work" python path/to/workload.py
```

Preserve the generated trace/statistics files and record the command. A
timeline can show that a kernel is launch-bound, but it does not by itself
explain a kernel's memory or arithmetic bottleneck; combine it with
`rocprofv2` or compiler evidence.

## Candidate/reference comparison

For a candidate and a reference, use identical semantic inputs and device
conditions. Capture them in separate directories:

```text
profile/hygon-rocprofv2/reference/trace/results_kernels.csv
profile/hygon-rocprofv2/candidate/trace/results_kernels.csv
```

Compare end-to-end elapsed time first. Then compare kernel counts, launch
durations, and resource columns. Do not require one-to-one kernel matching when
one implementation fuses or splits work.

## Optimization hypothesis and evidence

Every profile run must end with one falsifiable hypothesis, for example:

> The candidate spends more time in repeated task-map launches than in the
> producer kernel. Caching the schedule for unchanged `kv_lens` should remove
> those launches while preserving the same output; the hypothesis is falsified
> if the trace still contains the same launch count or the benchmark regresses.

Write `profile/<candidate-id>.md` with the command, workload, tool/version,
kernel summary, resource observations, limitations, and hypothesis. Keep failed
counter captures and regressed candidates in the task workspace evidence.
