---
id: decode-cta-local-two-half-pv-dot-regresses-uniform-workloads
kind: rejection
status: rejected
device: BW
arch: gfx936
operator: HY3 BF16 decode attention
scope: Within each 64-token score tile, gather probabilities into two 32-token slices,
  load V in two halves, and accumulate both PV dots into the same FP32 acc; applied
  to MTP1/uniform split and fused MTP producers without changing softmax or KV semantics.
rejected_on: '2026-11-03'
retry_when: Retry only if a Triton lowering can consume a sliced V/P operand without
  extra gather/register cost, and then demonstrate a paired multi-workload win.
---

# Decode CTA-local two-half PV dot regresses uniform workloads

> Status: rejected on `2026-11-03`. This is negative knowledge: it records a measured
> failure, not an optimization result. Do not cite it as a win.

## Attempted

- Device/architecture: `BW` / `gfx936`
- Operator: `HY3 BF16 decode attention`
- Scope: Within each 64-token score tile, gather probabilities into two 32-token slices, load V in two halves, and accumulate both PV dots into the same FP32 acc; applied to MTP1/uniform split and fused MTP producers without changing softmax or KV semantics.

## Hypothesis

Reducing live V tile size may lower register pressure while preserving CTA count and avoiding cross-CTA partial/merge work.

## What happened

The 12-case uniform_512 correctness selection passed. Geometric mean HY3 latency increased 4.74% versus saved baseline (candidate/baseline 1.0474); FA/HY3 geometric mean was 0.5468x. MTP1 rows were 6.5-7.9% slower and MTP2 2.9-4.8% slower; MTP3 2.1-2.9% slower. rocprofv2 showed producer VGPR 168->176 and uniform producer 136->152; LDS fell 16->8 KiB, but increased VGPR and an extra PV dot/gather outweighed that resource reduction.

## Why it was rejected

The segmented V loads did not lower register use, and the extra gather/dot work raised end-to-end latency across the selected MTP/layout panel. Candidate was reverted.

## When to retry

Retry only if a Triton lowering can consume a sliced V/P operand without extra gather/register cost, and then demonstrate a paired multi-workload win.

## Evidence

- `D:/Desktop/codex/.hygon-control/decode/runs/d04-pv-segment-selection.log`
- `D:/Desktop/codex/.hygon-control/decode/runs/d04-pv-seg-profile-export.txt`
- `D:/Desktop/codex/.hygon-control/decode/runs/baseline-full.txt`
