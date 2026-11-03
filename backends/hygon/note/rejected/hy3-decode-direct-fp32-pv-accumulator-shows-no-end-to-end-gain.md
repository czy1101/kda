---
id: hy3-decode-direct-fp32-pv-accumulator-shows-no-end-to-end-gain
kind: rejection
status: rejected
device: BW
arch: gfx936
operator: HY3 BF16 decode attention
scope: Pass alpha-scaled FP32 history directly as the tl.dot accumulator in the MTP1
  GQA kernel, uniform MTP1 kernel, and fused MTP/GQA producer
rejected_on: '2026-11-03'
retry_when: Retry only if a compiler or TTGIR/HSACO change can show that scaled history
  is accumulated in the matrix path without an added layout conversion or extra live
  state, followed by a paired full-panel win.
---

# HY3 decode direct FP32 PV accumulator shows no end-to-end gain

> Status: rejected on `2026-11-03`. This is negative knowledge: it records a measured
> failure, not an optimization result. Do not cite it as a win.

## Attempted

- Device/architecture: `BW` / `gfx936`
- Operator: `HY3 BF16 decode attention`
- Scope: Pass alpha-scaled FP32 history directly as the tl.dot accumulator in the MTP1 GQA kernel, uniform MTP1 kernel, and fused MTP/GQA producer

## Hypothesis

Using the historical output as the MMA accumulator may shorten the independent dot-result lifetime and reduce vector accumulation overhead while preserving the online-softmax recurrence

## What happened

108/108 BF16 cases passed; FA/HY3 geometric mean 3.8544x baseline versus 3.8552x candidate; HY3 latency geometric mean 0.2751 ms versus 0.2763 ms (candidate 0.41% slower). rocprofv2 showed nearly unchanged producer/uniform/combine duration and unchanged VGPR/LDS/grid.

## Why it was rejected

The full-panel result is within noise and slightly slower, with no resource or trace evidence of a useful change. The candidate was reverted.

## When to retry

Retry only if a compiler or TTGIR/HSACO change can show that scaled history is accumulated in the matrix path without an added layout conversion or extra live state, followed by a paired full-panel win.

## Evidence

- `D:/Desktop/codex/.hygon-control/decode/runs/d01-benchmark.txt; D:/Desktop/codex/.hygon-control/decode/runs/d01-profile-export.txt`
