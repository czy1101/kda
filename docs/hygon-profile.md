# Hygon gfx936 profiling workflow

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

The backend profile is [backends/hygon-gfx936.yaml](../backends/hygon-gfx936.yaml)
and the detailed procedure is [skills/hygon-profile/SKILL.md](../skills/hygon-profile/SKILL.md).

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

## Tool limitations

The legacy `rocprof` v1 executable is intentionally not the default route:
some Hygon DTK images contain a stale hard-coded `/opt/dtk-26.04/rocminfo`
path. If a task proves that v1 works, it may be used as a task-local command,
but the failure or success must be recorded. Missing counters or unavailable
timeline tools are capability limits; they are not evidence of a performance
improvement.
