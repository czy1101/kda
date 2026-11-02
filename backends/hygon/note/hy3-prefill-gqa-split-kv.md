---
id: hygon-hy3-prefill-gqa-split-kv
kind: guide
status: provisional
device: BW
arch: gfx936
operator: "hy3 prefill attention"
scope: "Portable GQA K/V reuse and split-KV patterns plus the dispatch policy that gates them"
revalidate_after: 2027-01-31
---

# HY3 prefill GQA and split-KV reference

This document records reusable, evidence-backed patterns for optimizing a
paged/block-sparse prefill attention kernel with pure Triton. It separates
portable algorithmic guidance from Hygon BW/gfx936 constraints. The measured
latencies below are workload evidence from one BW environment, not universal
performance guarantees and not dispatch constants to copy blindly.

## Scope and execution paths

The reference operator has FP8 Q/K/V or paged K/V cache, NHD/HND layouts,
optional block masks, GQA, online softmax, and a BF16 output. Three execution
paths are useful:

- **Base:** one program processes a query tile and walks the complete KV range.
- **GQA reuse:** one program processes multiple query rows/heads that share a
  KV head, so K/V tiles can be reused within the program.
- **Split-KV:** the KV range is divided into segments. A main kernel writes
  partial softmax state and accumulators; a reduction kernel merges segments
  with a numerically stable online-softmax formula.

No path is universally fastest. Dispatch must consider Q and KV lengths,
batch, query/KV head counts, layout, page size, mask sparsity, quantization
mode, and temporary workspace.

## Portable GQA reuse pattern

### Algorithm

For GQA, several query heads share one KV head. A multi-row program loads a KV
tile once where possible, maintains independent dot/softmax/accumulator state
for each query row, and writes each query head separately.

### Engineering rules

1. Start with a modest query-row group, such as `GROUP_M=2`, and a bounded
   query tile such as `BLOCK_M=64`.
2. Require a valid `Hq/Hkv` grouping and reject unsupported groups before
   launching a kernel.
3. Prefer unmasked inputs first. Different mask rows can require different
   KV loads and erase the reuse benefit.
4. Keep the GQA candidate behind a dispatch guard until NHD/HND, batch, page,
   and head-count combinations have independent evidence.
5. Treat a larger group as a separate candidate: more reuse can be outweighed
   by register pressure, accumulator footprint, or lower occupancy.

### Failure modes

- Register spilling or reduced occupancy offsets global-memory savings.
- A group tuned for one Q/KV shape regresses another shape.
- Heterogeneous batches and masks prevent effective sharing.
- Compilation and cache warm-up are confused with steady-state latency.

## Portable split-KV pattern

### Algorithm

The main kernel computes each `(batch, query-head, query-tile, KV-segment)`
independently and stores partial state:

- FP32 `partial_m` and `partial_l` for the online softmax;
- a partial output accumulator, commonly BF16 when the error budget permits.

The reduction kernel first finds the global maximum `m`, then applies
`exp2(segment_m - global_m)` to each segment's `l` and accumulator, and finally
normalizes and stores the output. Tail tiles and empty segments must be masked.

### Segment sizing and workspace

If `segment_tiles` counts 128-token KV tiles, a first-order workspace estimate
is:

```text
q_tiles       = ceil(Q / BLOCK_M)
kv_tiles      = ceil(KV / BLOCK_N)
num_segments  = ceil(kv_tiles / segment_tiles)
rows          = batch * Hq * q_tiles * num_segments * BLOCK_M
bytes         ~= rows * (8 + 2 * HEAD_DIM)
```

The estimate accounts for FP32 `m/l` and a BF16 accumulator. Add allocator,
alignment, and temporary-buffer headroom. Use a workspace budget and fall back
to base when the estimate is unsafe. A larger segment reduces partial storage
and reduction work but also reduces independent programs; a smaller segment
does the opposite. Candidate values such as 64, 128, and 256 should be
profiled rather than selected by KV length alone.

### Numerical and correctness requirements

- Merge all segments with the stable max/rescale formula.
- Check finite outputs, absolute/relative error, short sequences, tail tiles,
  page boundaries, both quantization modes, and both layouts.
- Compare base, explicit split, and automatic dispatch on identical metadata
  and masks.
- Preserve a base fallback for unsupported or uncertain workloads.

## Portable dispatch policy

Keep dispatch host-side and metadata-driven. Useful metadata includes Q/KV
lengths, batch, Hq/Hkv, layout, page size, mask presence, a caller-declared
sparsity bucket, and the workspace limit. A conservative order is:

1. Honor explicit `base`, `split`, or `gqa` controls used for controlled
   profiling.
2. Validate grouping, layout, mask, quantization, and workspace safety.
3. Prefer split-KV only when long KV and insufficient base parallelism are
   supported by measurements.
4. Try GQA only in an independently validated, usually unmasked regime.
5. Use base for the remaining cases.

Do not read a large device mask into host memory merely to make a dispatch
decision. If automatic mask dispatch is needed, compute or pass a compact
sparsity bucket as part of the workload metadata.

## Hygon BW/gfx936-specific guidance

### Runtime and compiler

The Hygon device is exposed through a CUDA-compatible PyTorch API, but it is
not an NVIDIA GPU. Device capability values must not be used as an NVIDIA SM
test. Set the DTK runtime and device mask explicitly in the task contract; a
typical setup is:

```bash
source /opt/dtk-26.04-DCC2602-0317/env.sh
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
export ROCR_VISIBLE_DEVICES=5
export TRITON_HIP_CLANG_PATH=/opt/dtk-26.04-DCC2602-0317/aillvm/bin/clang-18
export FLAG_ATTN_BACKEND=hygon
```

The Hygon Triton/HIP backend accepts `input_precision="ieee"`; do not infer
NVIDIA Ampere/Hopper and emit `tf32`. Treat wave/warp semantics, `num_warps`,
`num_stages`, LDS, VGPR use, and occupancy as backend-specific measurements.
The current BW/gfx936 calibration reference reports Wave64, 80 CU, roughly
64 KiB LDS, 32 KiB L1, and 8 MiB L2. These values bound experiments; they are
not a vendor peak specification.

### HY3-specific policy validated on BW

The current HY3 implementation includes GQA reuse, split-KV main/reduce
kernels, segment sizes 64/128/256, metadata dispatch, and a default 1 GiB
split workspace limit. The production policy is intentionally conservative:

- unmasked long-KV inputs can use split-KV only inside the measured Q/head
  range and when the workspace estimate is safe;
- high-sparsity mask bucket 3 is eligible only for short enough Q; medium
  sparsity remains on base until it has stable evidence;
- `GROUP_M=2` is the default; `GROUP_M=4` remains an explicit experiment;
- HND, batch>1, page-size variants, and quantization modes require their own
  matrix rather than inheriting a batch-1 NHD result.

Representative evidence from the BW environment:

| Workload | Base | Split | Interpretation |
|---|---:|---:|---|
| Q=32768, KV=262K, H=8, quant=1 | ~1281.8 ms | ~1259.3 ms | ~1.8% split gain; workspace safe |
| Q=32768, KV=262K, H=8, quant=0 | ~1536.4 ms | ~1417.5 ms | ~7.7% split gain; workspace safe |
| Q=32768, KV=262K, H=16 | — | — | workspace exceeds the default limit; use base |
| Q=32768, KV=512K, H=8 | — | — | workspace exceeds the default limit; use base |

These measurements justify a guarded dispatch extension, not a universal
split-KV rule. They also demonstrate why the workspace check must remain part
of automatic selection.

Useful HY3 controls for controlled experiments include:

```text
HY3_PREFILL_PATH=auto|base|split|gqa
HY3_PREFILL_SEGMENT_TILES=64|128|256
HY3_PREFILL_MAX_SPLIT_BYTES=<bytes>   # 0 means unlimited; experiment only
HY3_PREFILL_ALLOW_LARGE_SPLIT=1       # explicit unsafe-budget experiment
HY3_PREFILL_MASK_SPLIT=1              # explicit masked split experiment
HY3_PREFILL_GQA_GROUP_M=2|4
HY3_PREFILL_DEBUG_PATH=1
```

Do not bake these controls or one benchmark shape into a generic public
benchmark. Keep them in backend dispatch or the task contract.

## KDA optimization workflow

For a future task using this reference:

1. Read the task contract and the selected backend profile.
2. Run correctness and a reproducible base benchmark without deleting the
   Triton cache before every run.
3. Discover the available Hygon profiler. Prefer `rocprofv2 --kernel-trace`
   for kernel duration/grid/resource evidence and `hipprof` for launch gaps or
   synchronization. Treat `rocminfo`/`rocm-smi` as device-state probes.
4. Write one falsifiable hypothesis, then implement one coherent candidate.
5. Run correctness before and after the candidate, including edge cases and
   numerical checks.
6. Compare base, candidate, and automatic dispatch with alternating order and
   warm-up separated from timed samples.
7. Preserve raw traces and rejected candidates. Promote only a correct,
   reproducible candidate with a measured benefit and safe fallback.

## Remaining research areas

Prioritize page size 32, HND and heterogeneous batch, block-mask tile
compression, per-token scale traffic, Q beyond the validated range, and KV
beyond 512K. Do not remove base fallback or expand GQA/split-KV merely because
one shape improves. The target is a safe workload-aware selector, not universal
use of the most complicated kernel.
