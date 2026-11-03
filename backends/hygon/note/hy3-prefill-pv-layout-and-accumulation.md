---
id: hygon-hy3-prefill-pv-layout-and-accumulation
kind: guide
status: provisional
device: BW
arch: gfx936
operator: "HY3 BF16 prefill attention"
scope: "Evidence and transfer limits for PV orientation, segmented PV, accumulator reuse, and V loading"
last_validated: 2026-10-29
revalidate_after: 2027-01-31
---

# HY3 prefill PV layout, accumulation, and loading notes

This note consolidates structural experiments reported for Hygon HY3 BF16
prefill and related Parallax kernels. It separates measured wins from
hypotheses and cross-operator evidence. A result from Parallax is a reason to
test the same compiler-level question in HY3; it is not proof that HY3 benefits.

## Measured results

| Change | Reported result | What the result supports |
|---|---|---|
| HY3 BF16 PV reduction `PV128` → four `PV32` operations, with V loaded in segments | Same BF16 semantics; 12-case geometric-mean speedup `1.483x`, 8 long-sequence cases `1.806x`; sampled spills `42 → 0` | This is the strongest reported HY3 result. Segmenting the reduction and V loads can reduce live state and improve the tested workload. Keep the tested dispatch domain and recheck the full shape/layout matrix before broadening it. |
| Parallax D128 transposed PV direction (`V.T @ P.T`) | Target kernel `47.055 → 33.249 ms` (`-29.34%`); 32-case latency geometric mean `-3.76%`; spills `44 → 27`, registers remained 256 | Direction and operand layout can materially affect generated MMA mapping. This is cross-operator evidence only. D64 regressed by about `7.39%`, so do not apply by head dimension alone. |
| Parallax PV32 → two PV16 operations while loading V as a whole tile | D128 target kernel `33.273 → 30.426 ms`; complete 32-case latency geometric mean `-1.61%`; spills `27 → 15` | Finer accumulation helped the target kernel, but the end-to-end aggregate win was small. Kernel-only improvement is not enough to promote a default. |
| HY3 finite V32 prefetch | Reported as beneficial in one prior experiment; no complete numeric panel is preserved here | Keep as a bounded follow-up candidate. Compare just-in-time loading against only the next V segment; do not infer that earlier or wider prefetch is better. |

The reported HY3 reverse-PV control regressed all 16 long-capacity cases, but it
also changed the load direction. Treat it as evidence against that combined
variant, not as an isolated test of PV orientation. The Parallax percentages
above come from separate rounds and must not be added together.

## Reusable experiment design

### PV orientation

Compare `P[Q,KV] @ V[KV,D] -> acc[Q,D]` with
`V.T[D,KV] @ P.T[KV,Q] -> acc_t[D,Q]`. Keep the selected accumulator direction
through the loop, broadcast online-softmax alpha along the query axis, and
restore the output layout once at the end. The arithmetic and output element
count are unchanged. Inspect TTGIR, LLVM/HSACO, operand layouts, accumulator
layout, and shared-memory conversions; `tl.trans` is not evidence that a
conversion is free or that bank conflicts were removed.

Run this as an isolated A/B: keep V loading, softmax, tile sizes, and launch
configuration fixed. Cover both head dimensions and both layouts before
adding any metadata-based dispatch.

### Segmented PV and accumulator lifetime

Within a CTA, partition the MMA reduction axis and repeatedly accumulate into
one output accumulator, for example `acc = dot(a0, b0, acc)` followed by
`acc = dot(a1, b1, acc)`. Independently compare whole-tile V loading with
segmented V loading. Do not create multiple full output temporaries. This is
CTA-local reduction partitioning, not cross-CTA split-KV: it needs no partial
output/LSE merge.

When evaluating historical-output reuse, first write the exact online-softmax
algebra for the current kernel. An FP32 historical output cannot be inserted
directly into an integer MMA accumulator whose scale changes per segment.
Validate output tolerance and accumulation scale before treating this as a
candidate.

### Operand lifetime and prefetch

Test just-in-time V loads, one-segment lookahead, and a small bounded prefetch
window separately. Use generated code and profile resource data to explain a
latency change. A zero-spill result alone is not a win criterion; the full
operator latency and shape panel decide.

## Transfer boundaries and unvalidated ideas

- The transposed-PV result is from Parallax. HY3 must be independently measured
  with its own softmax, page/mask path, accumulator type, and launch schedule.
- A prior HY3 contiguous BF16 counter-control changed load direction together
  with PV direction, so it cannot isolate which change caused the regression.
- Directly feeding historical output into the first PV MMA is a hypothesis,
  not a recorded win. Re-derive the scale and alpha relationship for each
  quantization mode.
- Earlier `V0`/`V0mid` loading variants were reported as regressions. Retain
  those as rejected scheduling ideas unless a profile identifies a different
  bottleneck or the compiler changes.
- GQA query-row packing and split-KV dispatch are covered separately in
  [HY3 prefill GQA and split-KV reference](hy3-prefill-gqa-split-kv.md). They
  have their own workload and workspace guards; do not combine their reported
  gains with the PV results here.

## Promotion requirements

This guide remains provisional because its source summaries do not include the
full per-shape benchmark tables, correctness artifacts, and profile bundle in
this checkout. Before promoting an individual technique to a confirmed result,
record it as a separate KDA candidate with paired baseline/candidate runs,
correctness across the intended dispatch domain, compiler/profile evidence,
and end-to-end aggregation. Keep materially different code changes as
separate candidates so their effects are attributable.

## Evidence references

The following report paths were included with the prior optimization analysis;
they are provenance references and may not be present on this workstation:

- `E:/download/kda-main/outputs/20261029-fwdtranspose02/result.json`
- `E:/download/kda-main/outputs/20261029-fwdsplit01/result.json`
- `E:/download/kda-main/outputs/metax-hy3-bf16-prefill-20261028/report.html`
