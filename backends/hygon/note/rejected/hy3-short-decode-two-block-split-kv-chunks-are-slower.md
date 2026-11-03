---
id: hy3-short-decode-two-block-split-kv-chunks-are-slower
kind: rejection
status: rejected
device: BW
arch: gfx936
operator: HY3 BF16 decode attention
scope: Override HY3_BLOCKS_PER_CHUNK=2 for the uniform_512 BF16 decode panel, retaining
  multiple KV chunks rather than the rejected one-chunk setting
rejected_on: '2026-11-03'
retry_when: Retry only if a profile shows task-map or combine overhead materially
  dominates uniform_512 and a new task packing policy preserves per-CU producer parallelism
  without increasing serial work.
---

# HY3 short decode two-block split-KV chunks are slower

> Status: rejected on `2026-11-03`. This is negative knowledge: it records a measured
> failure, not an optimization result. Do not cite it as a win.

## Attempted

- Device/architecture: `BW` / `gfx936`
- Operator: `HY3 BF16 decode attention`
- Scope: Override HY3_BLOCKS_PER_CHUNK=2 for the uniform_512 BF16 decode panel, retaining multiple KV chunks rather than the rejected one-chunk setting

## Hypothesis

A middle split-KV granularity may reduce task-map and merge overhead while preserving enough CTA parallelism

## What happened

All 12 uniform_512 cases passed. Candidate MTP1 latency was about 0.167-0.169 ms across NHD/HND and static/dynamic versus about 0.130-0.134 ms in the baseline. MTP2/3 were about 0.198-0.207 ms versus about 0.187-0.195 ms. FA/HY3 ratios fell to 0.436-0.572x.

## Why it was rejected

The two-block chunk increased short-case latency across MTP/layout combinations, so it fails the focused selection gate and was not applied to the full workload panel.

## When to retry

Retry only if a profile shows task-map or combine overhead materially dominates uniform_512 and a new task packing policy preserves per-CU producer parallelism without increasing serial work.

## Evidence

- `D:/Desktop/codex/.hygon-control/decode/runs/d02-blocks2-selection.txt`
