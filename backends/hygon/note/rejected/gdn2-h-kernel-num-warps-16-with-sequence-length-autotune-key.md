---
id: gdn2-h-kernel-num-warps-16-with-sequence-length-autotune-key
kind: rejection
status: rejected
device: BW
arch: gfx936
operator: GDN2
scope: Hygon Triton H-kernel; add T to autotune key and expand H_KERNEL_NUM_WARPS
  to [2,4,8,16] for full 24-shape FP16/BF16 matrix.
rejected_on: '2026-11-02'
retry_when: Retry only if reliable profiling shows a distinct long-sequence H-kernel
  bottleneck and a non-overfit workload-class dispatch passes multiple shapes with
  no >2% regressions.
---

# GDN2 H-kernel num_warps=16 with sequence-length autotune key

> Status: rejected on `2026-11-02`. This is negative knowledge: it records a measured
> failure, not an optimization result. Do not cite it as a win.

## Attempted

- Device/architecture: `BW` / `gfx936`
- Operator: `GDN2`
- Scope: Hygon Triton H-kernel; add T to autotune key and expand H_KERNEL_NUM_WARPS to [2,4,8,16] for full 24-shape FP16/BF16 matrix.

## Hypothesis

Separate autotune choices by sequence length may retain the long-sequence 16-warp gain without short-shape regressions.

## What happened

All 24 accuracy checks passed. Geometric candidate/baseline ratio was 1.0026 (~0.26% slower); task sum_ratio was 0.9951 (~0.49% faster). However, 9/24 cases regressed >0.5%, max regression 4.54%, exceeding the per-shape guard. B2/T16384/H16/K=V=128 improved 4.3-4.4% both dtypes.

## Why it was rejected

Adding T to the key did not prevent regressions; reverted the warps list and tuner key.

## When to retry

Retry only if reliable profiling shows a distinct long-sequence H-kernel bottleneck and a non-overfit workload-class dispatch passes multiple shapes with no >2% regressions.

## Evidence

- `../../../../../../kda-gdn2-task/runs/candidate-numwarps16-tkey.csv`
