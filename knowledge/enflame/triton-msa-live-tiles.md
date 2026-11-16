# S60-TRITON-007: MSA query groups and bounded score tiles

## Scope and status

Observed2026-10-16 on S60/GCU300, Torch2.10.0+cpu with torch_gcu and Enflame Triton3.6.0. BF16 MSA prefill/decode, D128. Status: independent numerical checks, complete fourteen-shape full-call performance and real-launch-linked LLVM/GCU300 ISA evidence. This is not native FP8 support or a hardware-utilization measurement.

Retained best c2; sum-latency ratio 0.769933053, speedup 1.299x. Baseline was three repeats completed before user steering; candidates were measured once, without routine repeated benchmarking or discarded samples. These values apply only to the retained task contract.

## Existing hardware paths were already active

The actual public APIs launched20 original JIT specializations across42 JIT launches for all fourteen workloads. Records include exact source/JIT/compiled hashes, argument shapes/strides, metadata, LLVM call sites, extracted GCU300 ELF and disassembly. This upgrades S60-TRITON-005's earlier compile-only MSA evidence; native Torch-internal kernels remain outside this capture.

| Stage | Observed computation | Transfer scope |
|---|---|---|
| Prefill standalone index scoring | tops_convgen_m32_n64_k128_bf16_fp32 | Scalar page id, regular FP32/BF16 tile handling |
| Original prefill QK/PV | m8/m16/m32_n128_k128 convgen and bias families | Slice/transpose/memcpy/DTE waits; no BF16 payload gather found |
| Decode scoring and Attention | m8/m16 convgen families | Regular selected-page payload tiles |
| Softmax and TopK/merge | exp2, multiply/max/reduction; m.msf and vector instructions | Algorithm-specific scalar/vector work, no matrix operation required for TopK |

Matched matrix instructions include m.vmm2.mode17.bf. The computation-path names do not prove a separate physical engine per library entry or reveal occupancy/bandwidth/overlap.

## Trigger wider matrix tiles with bounded page-union query groups

The original formal TopK16 prefill dispatched one query per program. Reuse the existing grouped kernel while preserving each query's independent selected-page set, causal mask, head/query tails, BF16 operands and FP32 accumulation. For the newly enabled class require D128, TopK<=16, padded GQA<=32 and page-table capacity<=2*TopK. Use8 queries for padded GQA<=8, otherwise4, with enough query rows. Existing TopK4 with padded GQA=8 behavior remains intact.

Actual affected formal specializations changed from m32 to m128 and m16 to m64 convgen, including accumulator/bias entries. Structured KV transfers remained selected. The initial grouping-only candidate reduced the two affected full calls by20.03% and25.89%, and the complete sum by6.42%.

This is not an unrestricted grouping rule. Pairwise union/rank work grows quadratically in candidate slots, and divergent selections increase masked matrix and softmax work. The page-capacity bound is algorithmic, not a lookup of benchmark batch/sequence tuples. Whether a larger or different bound wins requires new measurements.

## Keep bounded FP32 score tiles live across TopK reductions

The previous streaming selector rescanned scores for each selected slot. For widths<=512 and TopK<=32 within its existing supported-input class, load8 query rows once and update a live FP32 value tile across per-row max/id reductions. At most4096 score elements are live. Preserve separate INT32 IDs, lowest-ID tie order, NaN-to-minus-infinity policy, visible-block masks, init/local promotion, output aliasing and strided buffers. Wider rows and TopK64 retain streaming.

Source liveness does not guarantee register residency; compiler-selected local/shared storage and spills must be inspected. This mechanism removes repeated global-score source loads and amortizes query-program work without changing score arithmetic or TopK semantics.

New row-selector transfer calls observed in the retained actual-launch artifacts:

`tops_deslice_async_2_si32`, `tops_gather_load_fp32`, `tops_memcpy_async`, `tops_slice_deslice_async_2_si32`, `tops_wait_dte`

Selected vector computation calls:

`tops_maxsi_si32`, `tops_maxsi_v1_si32`, `tops_maxsi_v4_si32`, `tops_minsi_si32`, `tops_minsi_v4_si32`, `tops_redmaxnumf_v1_fp32`, `tops_redmaxnumf_v2_fp32`, `tops_redmaxnumf_v4_fp32`

Numerical checks include12 frozen BF16 tests, six original-Attention comparisons and, for the row-selector version, the frozen padding regression plus five exact CPU selector cases. Coverage includes ragged prefixes, GQA6/12/24, TopK3/8/16/32/64, query tails, all-negative-infinity rows, NaNs/ties and noncontiguous buffers. Attention max absolute difference0.00048828125 stayed inside unchanged BF16 tolerances.

## Row-dependent masks can retain gather

The retained row-selector score input still lowers to tops_gather_load_fp32. A two-dimensional affine pointer expression with per-row visibility masks did not establish a regular slice-input path. The measured gain is from live tile reuse and fewer query programs, despite the gather entry. A block-pointer load followed by separate semantic masking is an untested next hypothesis, not a confirmed trigger or speedup.

## Scalar LSE can still lower to gather

The original one-chunk decode merge has two actual tops_gather_load_fp32 calls. LLVM def-use follows the LSE pointer into both calls; this is not BF16 KV payload gather. Even a scalar/one-element source expression can select a gather helper in this specialization. Neither its name nor its static count establishes bulk traffic, cost or a reason to replace it. Inspect exact argument provenance before classifying a call as a payload gather.

## Revalidation and limits

- Recheck source/compiled hashes, actual dispatch, dtype/strides, padded head ratio, TopK, context capacity, row tile and compiler/device-library version.
- Compile/inspect outside timing; run numerical tests in separate processes; measure complete operator cost including selection and grouping overhead.
- No matching GCU profiler profile was available; discovery selected no tool. Compiler artifacts and measurements do not establish physical unit mappings, dynamic execution counts or overlap.
- Native BF16-to-FP8 input conversion still aborted in the earlier MSA check. Software FP8 helpers elsewhere are not enabled by these BF16 changes.
- No new public hardware API was introduced. Newly validated trigger conditions select existing device-library interfaces.

## Sources

Source IDs in sources.json: msa-s60-final-report, msa-s60-baseline-runtime, msa-s60-best-runtime, msa-s60-runtime-artifacts, msa-s60-benchmarks, msa-s60-candidates, msa-s60-baseline-benchmark-raw, msa-s60-best-benchmark-raw, msa-s60-source-proof, msa-s60-attention-guards, msa-s60-selector-guards. Raw artifacts remain in the task workspace; card digests identify the reviewed evidence.
