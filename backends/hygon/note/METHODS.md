---
id: hygon-methods-index
kind: guide
status: confirmed
device: BW
arch: gfx936
scope: "Route a measured Hygon bottleneck to the experiment that tests it, and to the evidence that decides it"
revalidate_after: 2027-01-31
---

# Choosing the next Hygon experiment

This is the index for operator optimization on BW/gfx936. It does not replace
the detailed notes; it tells you which note applies once you have evidence for a
bottleneck class, and what measurement decides the question.

## How to use it

1. Confirm the backend, device, and toolchain from the task contract and live
   preflight. Do not reuse another image's numbers.
2. Run correctness and record a baseline before touching the kernel.
3. Take a `rocprofv2` kernel trace (see
   [profiling-workflow.md](profiling-workflow.md)) and read the kernel split:
   which kernel dominates, and how its time divides between memory, compute,
   LDS, and launch overhead.
4. Find the matching row below. Run its **first experiment**. Read the linked
   note only after that experiment is justified by evidence.
5. Record the candidate, the numbers, and the decision, including a rejection
   with its retry condition.

A row is a hypothesis to test, not an instruction to apply. Most rows also
describe what the evidence would look like if the hypothesis is wrong.

## Bottleneck to experiment

| # | Bottleneck class | Evidence that selects it | First experiment | Hygon specifics | Deciding evidence |
|---|---|---|---|---|---|
| 1 | Matrix path under-used | `tl.dot` lowers to vector FMAs, or the M tile is partially used; compute-bound with low achieved TFLOPS | Compute `M_eff = rows × heads_per_group` and `occupancy = M_eff / M_tile`; if `M_eff == 16` try the MMAC path, if smaller try a vector/FMA candidate | BF16 MMAC tile observed at M=16 with 64x64x16 shape; FP8 does not inherit the BF16 lowering | Same-shape A/B latency + HSACO instruction mix (see [mmac-fma-path-selection.md](mmac-fma-path-selection.md)) |
| 2 | Global-memory traffic from an intermediate | Trace shows a separate kernel or a large FP32 intermediate that is written and re-read | Fuse the producer and consumer so the intermediate stays in registers/LDS; verify the fused stage computes the same tile | One CTA must hold the complete intermediate tile; K=V=64 worked, K=V=128 regressed (see [rejected/](rejected/)) | End-to-end latency plus a correctness check on every fused shape |
| 3 | Reuse that LDS staging could exploit | Kernel re-reads the same K/V or weight tile across iterations; occupancy has headroom | Software-managed LDS staging with two buffers and explicit producer/consumer ordering | 64 KiB LDS per workgroup; no direct global-to-LDS path on this toolchain, so staging costs a VGPR hop | Latency plus LDS/VGPR/occupancy columns; staging is not TMA ([gfx936-validation.md](gfx936-validation.md)) |
| 4 | Register pressure or occupancy | Trace/HSACO shows high VGPR use, spills, or too few waves per CU for the tile size | Reduce the tile or the live range; compare a smaller tile with more waves against a larger tile with fewer | 40 waves per CU max, 1024 work-items per workgroup; more `num_warps` is not free under Wave64 | Same-shape latency and the resource columns together, never latency alone |
| 5 | Launch-bound or grid tail | Kernel duration is small relative to the operator, or grid size is not a multiple of the CU count | Merge stages into one kernel, use multi-row programs to raise per-CTA work, or round the grid to full waves | 80 CUs; a grid of 100 blocks leaves a short second wave that costs like a full one | Timeline from `hipprof` plus end-to-end latency ([vector-reduction-and-launch-bound.md](vector-reduction-and-launch-bound.md)) |
| 6 | Reduction-bound (softmax, norms, sums) | Trace shows one row per CTA with serialized accumulation, or low achieved bandwidth on a bandwidth-light kernel | Split the row across a block, reduce in two stages when the row exceeds one wave's reach, avoid per-element atomics | Wave64 reduction order differs from Wave32; block size must be a multiple of 64 | Kernel time and the reduction tree in the source, plus output tolerance checks |
| 7 | Irregular or paged access | Trace shows many short memory transactions, page-table loads on the critical path, or heavy masking | Coalesce the page-table lookup, hoist masks, group rows by page, or transpose the layout | L1 is 32 KiB and L2 is 8 MiB: a working set above L2 loses all reuse | Memory-pipeline counters when available; otherwise kernel-trace shape plus a controlled layout A/B |
| 8 | Too little parallelism (split-K tail) | One CTA per output row and a long serial K loop; trace shows the tail dominating | Split K into segments, combine in a second stage, size segments from waves available | Combine cost is real: measure it, do not assume more segments is better ([hy3-prefill-gqa-split-kv.md](hy3-prefill-gqa-split-kv.md)) | Same-shape latency across segment counts plus a numerics check on the combine |
| 9 | dtype/lowering mismatch | FP8 input but FP16 MMAC in the HSACO; or a claimed "native FP8" win with no instruction evidence | Inspect the generated ISA before optimizing; compare the FP8 and BF16 paths on identical shapes | PyTorch FP8 `torch.mm` is not implemented in the validated image; that is a framework limit, not a hardware one | HSACO disassembly plus same-shape latency; never infer from the dtype alone |
| 10 | Configuration space, not structure | The kernel is structurally fine but launch parameters are untuned for the shape family | Sweep `num_warps`, `num_stages`, and block sizes within the resource budget, then fix one default per workload class | `num_warps` is a count of 64-wide waves: 4 means 256 threads ([tune-configs.md](tune-configs.md)) | Per-shape latency table across the sweep, with a guarded fallback for shapes outside it |

## Turning a hypothesis into a claim

A candidate needs all four before it can be promoted:

1. **Correctness** on the reference, including the fallback path and at least one
   shape outside the optimized domain.
2. **Generated code** when the hypothesis is about lowering (MMAC vs vector,
   FP8 vs FP16): inspect the HSACO of the kernel that was actually timed.
3. **Profiling** that explains the change, not just accompanies it. On this
   backend a kernel trace is required; a counter claim additionally needs a
   parseable counter export.
4. **Same-shape A/B** latency with identical warmup, repeat, device mask, and
   allocator state, reported per shape and aggregated with the task's
   `goal.aggregation`.

If any of the four is missing, the honest outcome is "target not reached,
bottleneck unconfirmed" — not a promoted candidate.

## Anti-overfitting rules

- Do not encode a list of benchmark shapes as the dispatch policy.
- Do not select MMAC because the disassembly contains one MMAC instruction.
- Do not copy a BF16 path to FP8 without checking its lowering and resources.
- Do not report a kernel-only win as an operator win before launch, combine,
  and synchronization costs are included.
- Keep an explicit fallback for partial tiles and unsupported layouts.
- Record the rejection and the workload panel that rejected it.

## What this library does not cover yet

The following are open, not "known to be impossible":

- vendor/theoretical HBM peak and cross-card P2P bandwidth;
- native FP8 matrix peak, and whether a future DTK lowers FP8 MMAC;
- numeric VMEM/VALU/LDS/occupancy counters when `rocprofv2 -i` produces no file;
- whether a dedicated direct global-to-LDS path exists in a private toolchain;
- attention variants outside the recorded GLA and HY3 scopes.

## Where the detail lives

| Question | Note |
|---|---|
| What are the validated hardware limits and calibration numbers? | [gfx936-validation.md](gfx936-validation.md) |
| How do I capture and read a profile? | [profiling-workflow.md](profiling-workflow.md) |
| MMAC or vector FMA, and when to fuse rows? | [mmac-fma-path-selection.md](mmac-fma-path-selection.md) |
| GQA reuse and split-KV patterns | [hy3-prefill-gqa-split-kv.md](hy3-prefill-gqa-split-kv.md) |
| Reduction, memory-bound, and launch-bound patterns | [vector-reduction-and-launch-bound.md](vector-reduction-and-launch-bound.md) |
| How do I express launch parameters in Triton configs? | [tune-configs.md](tune-configs.md) |
| What did we already try that failed? | [rejected/](rejected/) |
| How do I port a new operator end to end? | [OPERATOR-PORTING.md](OPERATOR-PORTING.md) |
