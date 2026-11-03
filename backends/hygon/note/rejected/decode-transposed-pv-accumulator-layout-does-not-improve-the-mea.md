---
id: decode-transposed-pv-accumulator-layout-does-not-improve-the-mea
kind: rejection
status: rejected
device: BW
arch: gfx936
operator: HY3 BF16 decode attention
scope: Changed persistent PV accumulation from P[Q,KV] @ V[KV,D] -> acc[Q,D] to V.T[D,KV]
  @ P.T[KV,Q] -> acc_t[D,Q] in the fused producer and uniform split-KV kernels; kept
  softmax, masks, split policy, and benchmark cases unchanged.
rejected_on: '2026-11-03'
retry_when: Retry only with a change to MMA operand/layout lowering that reduces synchronization
  or measured kernel time, and compare identical MTP/layout specializations with at
  least the full correctness and benchmark panels.
---

# Decode transposed PV accumulator layout does not improve the measured panel

> Status: rejected on `2026-11-03`. This is negative knowledge: it records a measured
> failure, not an optimization result. Do not cite it as a win.

## Attempted

- Device/architecture: `BW` / `gfx936`
- Operator: `HY3 BF16 decode attention`
- Scope: Changed persistent PV accumulation from P[Q,KV] @ V[KV,D] -> acc[Q,D] to V.T[D,KV] @ P.T[KV,Q] -> acc_t[D,Q] in the fused producer and uniform split-KV kernels; kept softmax, masks, split policy, and benchmark cases unchanged.

## Hypothesis

Keeping the output accumulator in the transposed MMA orientation may reduce layout conversions and accumulator movement.

## What happened

The 12-case uniform_512 correctness selection passed. Candidate/baseline geometric mean HY3 latency was 1.0084, a 0.84% regression; FA/HY3 geometric mean was 0.5653x. TTGIR confirms transposed PV MMA. In the matched K=64 producer specialization, local_alloc fell 27->18 and MMAC instructions 360->144; shared memory stayed 16 KiB, but LDS writes rose 243->306 and barriers 107->170. The K=32 producer kept 180 MMAC instructions, while LDS writes rose 135->234 and barriers 89->125. This codegen shift did not yield end-to-end gain.

## Why it was rejected

Short decode did not improve overall, and additional synchronization/store work offset the reduced layout allocations in the measured panel. The candidate was restored.

## When to retry

Retry only with a change to MMA operand/layout lowering that reduces synchronization or measured kernel time, and compare identical MTP/layout specializations with at least the full correctness and benchmark panels.

## Evidence

- `D:/Desktop/codex/.hygon-control/decode/runs/d03-transposed-pv-selection.txt`
- `D:/Desktop/codex/.hygon-control/decode/runs/d03-codegen-reference-exact.log`
- `D:/Desktop/codex/.hygon-control/decode/runs/d03-codegen-candidate-exact.log`
