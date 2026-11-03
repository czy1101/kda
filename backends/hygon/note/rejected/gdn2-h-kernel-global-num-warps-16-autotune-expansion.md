---
id: gdn2-h-kernel-global-num-warps-16-autotune-expansion
kind: rejection
status: rejected
device: BW
arch: gfx936
operator: GDN2
scope: Hygon Triton H-kernel; H_KERNEL_NUM_WARPS expanded from [2,4,8] to [2,4,8,16]
  for the full 24-shape FP16/BF16 matrix.
rejected_on: '2026-11-02'
retry_when: Retry only with a principled low-cardinality workload-class dispatch (not
  global autotuning) and paired validation showing long low-grid K=V=128 gain while
  every other shape stays within 2% regression.
---

# GDN2 H-kernel global num_warps=16 autotune expansion

> Status: rejected on `2026-11-02`. This is negative knowledge: it records a measured
> failure, not an optimization result. Do not cite it as a win.

## Attempted

- Device/architecture: `BW` / `gfx936`
- Operator: `GDN2`
- Scope: Hygon Triton H-kernel; H_KERNEL_NUM_WARPS expanded from [2,4,8] to [2,4,8,16] for the full 24-shape FP16/BF16 matrix.

## Hypothesis

The profiled H-kernel's 280-byte scratch allocation and 256 architectural VGPRs might benefit from more warps distributing per-thread live state.

## What happened

All 24 accuracy checks passed. Geometric candidate/baseline time ratio was 0.9992 (~0.08% faster); task sum_ratio was 0.9941 (~0.59% faster). However, 6/24 cases regressed by >0.5%; BF16 B2/T512/H8 regressed about 2.84%, exceeding the 2% guard. B2/T16384/H16/K=V=128 improved 4.4-4.6% in both dtypes.

## Why it was rejected

The global autotune expansion did not yield robust full-matrix benefit and violated the allowed per-shape regression limit. Restored [2,4,8].

## When to retry

Retry only with a principled low-cardinality workload-class dispatch (not global autotuning) and paired validation showing long low-grid K=V=128 gain while every other shape stays within 2% regression.

## Evidence

- `../../../../../../kda-gdn2-task/runs/candidate-numwarps16-global.csv`
