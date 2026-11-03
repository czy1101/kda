---
id: hygon-profiling-workflow
kind: guide
status: confirmed
device: BW
arch: gfx936
scope: "Capture and read rocprofv2 kernel traces, counter exports, hipprof timelines, and device probes"
last_validated: 2026-11-03
revalidate_after: 2027-02-28
---

# Hygon gfx936 profiling workflow

For the exact command per tool, the artifact it produces, and its failure modes,
see [profiling-tools.md](profiling-tools.md). This document covers the sequence
and the evidence rules.

KDA treats Hygon profiling as a backend capability selected from live probes,
not as a hard-coded assumption. Set a task target such as:

```yaml
target:
  backend: hygon
  device: BW
  arch: gfx936
  profile: hygon-gfx936
  language: triton
```

The backend profile is [backends/hygon-gfx936.yaml](../../hygon-gfx936.yaml)
and the detailed executable skill is [skills/hygon-profile/SKILL.md](../../../skills/hygon-profile/SKILL.md).
The validated BW/gfx936 hardware and ISA calibration is recorded in
[gfx936-validation.md](gfx936-validation.md). Treat that document
as a reproducibility anchor, not as a vendor peak specification.

For reusable HY3 prefill structure and dispatch guidance, see
[hy3-prefill-gqa-split-kv.md](hy3-prefill-gqa-split-kv.md). It
separates portable GQA/split-KV patterns from Hygon-specific constraints and
does not replace task-local profiling evidence.

For the reusable matrix-tile, multi-row fusion, and MMAC-versus-vector path
selection method, see
[mmac-fma-path-selection.md](mmac-fma-path-selection.md). The
method separates the HY3 Decode example from the portable validation rules.

## Required tools during performance analysis

For a Hygon performance bottleneck analysis, collect at least one usable
profiler capture before making a bottleneck claim or promoting a performance
candidate. Start with a `rocprofv2 --kernel-trace` capture of the candidate
and, when available, the comparable reference. If that capture is unavailable
or unusable, try `rocprof` only after a task-local smoke capture produces a
parseable trace or counter export. Record the exact failure when no profiler
capture can be produced; benchmark timing alone is not profiler evidence.

| Tool | Use | Requirement |
|---|---|---|
| `rocprofv2` | Kernel trace | Default for kernel-level performance analysis. |
| `rocprof` v1 | Hardware counters (and a legacy trace) | The only counter path that produced a parseable export on the validated image; keep its capture separate from latency measurement. |
| `hipprof` | Timeline, synchronization, launch gaps, host/device overlap | Use for timeline questions or when timeline evidence is needed to explain latency. |
| `rocsys` | Selective start/stop control of a `rocprofv2` profiling session | Use for long-running workloads needing a selected capture window; not an independent profiler. |
| `rocminfo` | Runtime agent and architecture information | Run during Hygon preflight/profile setup. |
| `rocm-smi` or `hy-smi` | Device state, clocks, memory, and utilization | Run during Hygon preflight/profile setup. |

Do not treat tool discovery or `--list-basic`/`--list-counters` output as proof
that a capture is usable. Only use metrics after capture and structured export
complete for the target workload. On 2026-11-03 in `jiangrenkai-dev`,
`rocprofv2 --list-counters` listed gfx936 counters while `rocprofv2 -i` exited
zero without producing any file, and `rocprof` v1 produced a counter CSV from
the same workload. The counter route is image-dependent: smoke-test it every
time.

## Recommended KDA sequence

1. Define the task workspace and task contract outside this repository.
2. In `environment.setup`, source the DTK runtime, set the Hygon device mask,
   and make the profiler paths available.
3. Run correctness and the baseline before profiling.
4. Discover tools in the contracted environment:

   ```bash
   python3 <kda>/scripts/kernelpilot.py discover-tools \
     --workspace <task-workspace>
   python3 <kda>/scripts/kernelpilot.py strategy \
     --workspace <task-workspace>
   ```

5. Prefer `rocprofv2` for kernel traces and structured CSV. Use `hipprof` for
   timeline questions. Run `rocm-smi` (or native `hy-smi`) and `rocminfo` as
   device-state probes, not as performance profilers.
6. Capture the candidate through the contracted profile command:

   ```bash
   python3 <kda>/scripts/kernelpilot.py run profile \
     --workspace <task-workspace> --variant candidate
   ```

7. If a comparable reference is configured, capture it with the same workload:

   ```bash
   python3 <kda>/scripts/kernelpilot.py run profile \
     --workspace <task-workspace> --variant reference
   ```

8. Preserve raw reports in `profile/`, summarize them in
   `profile/<candidate-id>.md`, and write one falsifiable optimization
   hypothesis before editing the implementation.
9. Run correctness before and after each candidate, then run the required
   benchmark matrix. Promote only a correct candidate with measured evidence.

## Validated BW/gfx936 constraints

The current reference image reports Wave64, 80 CU, 64 KiB group/LDS, 32 KiB
L1, and 8 MiB L2. Use these values to bound a sweep, not to infer occupancy or
performance ratios. Four `num_warps` maps to a 256-thread workgroup on this
backend; eight maps to 512 threads. Check the generated HSACO and the trace
resource columns after each change.

The same validation measured approximately 1.35 TB/s for large D2D copies and
about 305 TFLOPS for FP16/BF16 `torch.mm` at N=8192. These are environment
calibration measurements. They must remain separate from HBM theoretical peak,
native FP8 peak, and operator latency in task reports.

The current DTK image lists gfx936 PMC names, but a counter capture must still
produce a parseable result file before any VMEM/VALU/LDS/occupancy claim is
made. If capture produces no file, retain the kernel trace and HSACO evidence
and mark the numeric counter capability unavailable.

## Task profile configuration

The task contract owns workload-specific commands. A minimal configuration is:

```yaml
profile:
  tool: auto
  tools:
    rocprofv2:
      candidate_command: >-
        rm -rf profile/hygon-rocprofv2/candidate &&
        mkdir -p profile/hygon-rocprofv2/candidate &&
        rocprofv2 --kernel-trace
        -d profile/hygon-rocprofv2/candidate/trace -o kernels
        python path/to/profile_workload.py
      reference_command: >-
        rm -rf profile/hygon-rocprofv2/reference &&
        mkdir -p profile/hygon-rocprofv2/reference &&
        rocprofv2 --kernel-trace
        -d profile/hygon-rocprofv2/reference/trace -o kernels
        python path/to/reference_profile_workload.py
    hipprof:
      candidate_command: >-
        rm -rf profile/hygon-hipprof/candidate &&
        mkdir -p profile/hygon-hipprof/candidate &&
        hipprof --hip-trace --stats
        -o profile/hygon-hipprof/candidate
        python path/to/profile_workload.py
```

Use `rocprofv2 --list-counters` before adding a counter capture. Counter names
vary across DTK images, so they belong in the task workspace evidence and must
not be assumed by the reusable backend profile.

## Reading traces and counters

For a deterministic kernel trace, warm up the exact workload first, then use
the same input, shape, dtype, device mask, stream policy, and repetition policy
as the benchmark. Preserve `results_kernels.csv` under the task workspace's
`profile/` directory. Useful columns include `Kernel_Name`, `GRD`, `WGR`,
`LDS`, `Arch_VGPR`, `SGPR`, `Start_Timestamp`, and `End_Timestamp`; compute
duration from the timestamp difference and group by kernel name. Do not infer
occupancy from one resource column alone.

Discover counter names on the actual image with `rocprofv2 --list-counters`,
then make a small capture using only listed counters. If capture or export
fails, preserve the error and do not treat the missing value as zero. Keep the
kernel trace and generated-code evidence as fallbacks.

Use `hipprof --hip-trace --stats` for launch gaps, synchronization, and
host/device overlap questions. A timeline can establish that launches or
host-side gaps dominate, but it cannot by itself explain a kernel's memory or
arithmetic bottleneck; combine it with kernel traces or compiler artifacts.

For each capture, finish with one falsifiable optimization hypothesis and
write `profile/<candidate-id>.md` with the command, workload, tool/version,
kernel/resource summary, limitations, and hypothesis. Compare candidate and
reference end-to-end latency first, then compare kernel counts, durations, and
resource columns. Do not require one-to-one kernel matching when an
implementation fuses or splits work.

## Tool limitations

- `rocprofv2 -i` counter capture produced no file on the validated image; use
  `rocprof` v1 for counters and keep the trace path for timing.
- `rocprof` v1 replays the workload per PMC group, so its run is not a latency
  measurement. Some DTK images also contain a stale hard-coded
  `/opt/dtk-26.04/rocminfo` path; smoke-test the capture before relying on it.
- `hipprof` needs the DTK environment on `LD_LIBRARY_PATH`; a bare invocation
  fails to load `libperfetto.so.5`.
- `rocsys` only controls a session; it does not collect data by itself.
- `proton` / `triton.profiler` is unusable on the FlagGems Triton build in the
  validated image (`cannot import name 'amd' from triton._C.libtriton`).
- Missing counters or unavailable timeline tools are capability limits, not
  evidence of a performance improvement.
