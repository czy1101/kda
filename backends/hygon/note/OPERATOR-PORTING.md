---
id: hygon-operator-porting
kind: guide
status: provisional
device: BW
arch: gfx936
scope: "Port a new operator to Hygon BW/gfx936 in Triton and get it through the KDA evidence gates"
revalidate_after: 2027-01-31
---

# Porting an operator to BW/gfx936

A checklist for taking an operator that already runs on NVIDIA (or only on CPU)
and getting a correct, measured, promotable Hygon implementation. It assumes the
KDA task workflow: the contract, evidence files, and gates are described in
`workflows/kernel-optimization.yaml` and `AGENTS.md`.

Use [METHODS.md](METHODS.md) after this checklist: this tells you how to stand up
the operator and its harness; METHODS tells you which experiment to run next.

## 0. Preconditions

- [ ] The task contract exists: `python scripts/kernelpilot.py init <operator-repo>`,
      then fill `.kernelpilot/task.yaml`.
- [ ] `execution` is set: `local`, or `ssh` with `host`, `workspace`, and the
      `container` that owns the DTK environment.
- [ ] `environment.setup` reproduces the exact toolchain: DTK `env.sh`, `PATH`,
      `LD_LIBRARY_PATH`, and the device mask (`HIP_VISIBLE_DEVICES`). Do not rely
      on a login shell.
- [ ] `environment.preflight` prints something that proves both the import and
      the device: `python -c "import torch; print('OK', torch.cuda.get_device_name(0))"`.
- [ ] `constraints.allowed_paths` lists exactly the files the agent may edit;
      `forbidden_paths` lists the reference, tests, and benchmark adapter.
- [ ] `goal` states the metric, direction, aggregation, and either
      `relative_to_baseline` or `target_value`.

## 1. Understand before editing

- [ ] Read the reference implementation and its tests.
- [ ] Write `docs/draft.md`: inputs, layouts, dtypes, masks, reduction axes, and
      the exact shapes the operator must support.
- [ ] Write `docs/plan.md` with the candidate order you intend to try.
- [ ] Run the reference once to confirm the harness works before any edit.

## 2. Numerics contract

- [ ] Decide the accumulation dtype. BF16/FP16 inputs usually accumulate in
      FP32; write the tolerance down before comparing.
- [ ] Compare against a reference computed at higher precision, not against the
      previous Hygon output — a wrong kernel must not be able to bless itself.
- [ ] Check the reduction order explicitly for masks, softmax, and norms: Wave64
      reductions differ from NVIDIA's Wave32, so tie-breaking and small-value
      differences are expected and must be within tolerance.
- [ ] Test the fallback path directly, not only the fast path.

## 3. Launch-parameter worksheet

Validated limits for this image: Wave64, 80 CUs, 40 waves per CU, 1024
work-items per workgroup, 64 KiB LDS per workgroup, 32 KiB L1, 8 MiB L2.

- [ ] Threads = `num_warps × 64`. `num_warps=4` is 256 threads, `8` is 512, and
      the workgroup cap is 1024 threads (`num_warps=16`).
- [ ] Choose block sizes that divide the real dimensions; a partially used tile
      pays the padding cost in both registers and LDS.
- [ ] Budget LDS per stage before adding `num_stages`:

```text
bytes_per_stage = (BM × BK + BK × BN) × bytes_per_element
total_bytes     = bytes_per_stage × num_stages        must stay <= 65536
```

| Tile (bf16) | Per stage | 2 stages | Fits 64 KiB? |
|---|---:|---:|---|
| BM=64, BK=64, BN=64 | 16 KiB | 32 KiB | yes |
| BM=128, BK=64, BN=64 | 24 KiB | 48 KiB | yes |
| BM=128, BK=128, BN=64 | 48 KiB | 96 KiB | **no** — reduce `num_stages` or the tile |
| BM=128, BK=128, BN=128 | 64 KiB | 128 KiB | **no** |

- [ ] `num_stages > 1` does not by itself create overlap on this toolchain: the
      backend reported `use_async_copy=false` and `use_pingpong=false` for
      gfx936. Check the generated code before believing a pipeline exists.

## 4. Dispatch and packaging guard

- [ ] One stable operator identity: public API name, `quality.naming.op_name`,
      pytest marker, registry id, and unified dispatcher entry must agree.
- [ ] Relative imports become repository-qualified imports
      (`from flag_attn.runtime.backend._hygon.FLA.index import ...`); the built-in
      audit rejects `from .index import ...`.
- [ ] No dispatch table keyed by benchmark shape. If a shape needs a special
      path, it needs an algorithmic reason, a guard, and `generalization`
      evidence.

## 5. Correctness panel

Run at least these before the first benchmark:

- [ ] the smallest supported shape;
- [ ] a typical shape from the target workload;
- [ ] an odd/non-power-of-two dimension that exercises masking;
- [ ] the largest supported shape (memory and index arithmetic);
- [ ] a shape outside the optimized domain that must take the fallback.

Use `python scripts/kernelpilot.py run correctness` so the contracted environment
is applied. Run `python scripts/kernelpilot.py run quality` before the baseline
and after every candidate.

## 6. Baseline and first benchmark

- [ ] Record the baseline candidate in `runs/candidates.jsonl` and its rows in
      `runs/benchmark.csv` with the contract's `benchmark.shapes`.
- [ ] Use the contract's warmup and repeat counts; keep them identical between
      candidates and baseline.
- [ ] Run A/B in alternating order where the harness allows it, so a clock or
      thermal drift cannot masquerade as a win.
- [ ] Report median and dispersion; a single fast run is not a result.

## 7. Profile before the second candidate

- [ ] `python scripts/kernelpilot.py discover-tools` and keep
      `runs/analysis-capabilities.json`.
- [ ] Capture `rocprofv2 --kernel-trace` for the candidate, and for the reference
      when the comparison is enabled.
- [ ] Split the operator time by kernel; know which kernel you are optimizing
      before changing it.
- [ ] Treat counters as unavailable unless a task-local capture produces a
      parseable CSV. Missing counter output is not zero.

## 8. Iterate one hypothesis at a time

- [ ] One coherent change per candidate, with the hypothesis written down before
      the edit.
- [ ] Deploy through the configured transport (`remote pull` / `remote push`),
      never by direct `ssh`/`scp`.
- [ ] After each candidate: correctness, quality, benchmark, record, then decide.
- [ ] On rejection, restore the best correct checkpoint and record why the
      candidate failed.

## 9. Promotion and knowledge

- [ ] `python scripts/kernelpilot.py checkpoint save-best --candidate <id>`
      refuses unless the ledgers back the candidate; if it refuses, the run is
      not finished.
- [ ] Publish a confirmed note only if the gain is material and the scope is
      understood (`scripts/record_knowledge.py`).
- [ ] Record a measured dead end with `scripts/record_rejection.py` so the next
      person does not repeat it.
- [ ] If nothing improved, say so: an evidence-backed "target not reached" is an
      acceptable result.

## Common Hygon failure modes

| Symptom | Likely cause | Check |
|---|---|---|
| Kernel compiles but is much slower than expected | Vector FMA path where a complete M=16 tile was available, or the opposite | HSACO instruction mix |
| `num_stages` sweep changes nothing | No ping-pong/asynchronous copy generated on gfx936 | Compiler flags and HSACO |
| Occupancy collapses when the tile grows | 64 KiB LDS or VGPR limit crossed | LDS and register columns in the trace |
| "Native FP8 win" that does not reproduce | FP8 lowered to an FP16 MMAC path | Disassembly, not the dtype |
| Counter capture returns nothing | `rocprofv2 -i` produced no result file in this image | Presence of the CSV, not the exit code |
| Correct on one shape, wrong on another | Mask, page table, or fallback branch not exercised | The correctness panel in step 5 |
| Timings differ between runs | Clocks, other tenants, or a different device mask | `hy-smi`/`rocm-smi` state and the contracted `HIP_VISIBLE_DEVICES` |

## Contract skeleton to copy

```yaml
version: 1
task:
  name: <operator>-<device>
  description: <one line>
target:
  backend: hygon
  device: BW
  arch: gfx936
  profile: hygon-gfx936
  language: triton
execution:
  transport: ssh
  host: <ssh-config-host>
  workspace: /path/in/container
  container: <docker-container>
environment:
  shell: bash
  setup: |
    source /opt/dtk-26.04-DCC2602-0317/env.sh
    export PYTHONPATH=<remote-workspace>/src:${PYTHONPATH:-}
    export HIP_VISIBLE_DEVICES=<device>
implementation:
  path: <repository-relative implementation path>
reference:
  path: <repository-relative reference path>
correctness:
  command: python <correctness harness>
benchmark:
  command: python <benchmark harness>
  warmup: 20
  repeat: 100
  shapes: [<shape-a>, <shape-b>]
baseline:
  name: original
goal:
  metric: latency
  direction: minimize
  aggregation: geomean_ratio
  relative_to_baseline: 0.9
  max_regression: 0.02
constraints:
  allowed_paths: [<implementation path>]
  forbidden_paths: [<reference path>, <test path>, <benchmark path>]
stop:
  max_candidates: 8
  max_profile_runs: 4
  no_improvement_rounds: 3
```
