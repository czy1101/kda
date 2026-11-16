# S60-TRITON-009: six-category execution survey before operator tuning

Observed2026-10-16, GCU300/S60, torch_gcu2.10.0+3.7.20260408, Triton3.6.0. This extends S60-TRITON-005; the current task first surveys isolated mechanisms and then separately validates MSA. Multi-device communication was excluded.

## Interfaces and interpretations

- Matrix: tl.dot with sampled BF16 M8/16/32/64/128,N128,K128; FP16/FP32 M64. Operand layout, num_warps and geometry change device-library decomposition. Actual convgen/matmul call signatures and GCU300 ISA are in summary/artifacts. Do not infer full precision or shape support from this sample.
- Vector/scalar: tl.fma,exp2,log2,rsqrt,where,max/sum and INT32 cumsum are probed independently. Scalar page selection followed by affine payload addresses remains different from a vector of row IDs.
- On-chip reuse: retaining loaded operands vs explicit reload, and softmax Q1/Q8 live tiles, expose generated local/shared allocations and copy effects. Compiler memory attributes are not measured physical capacity, bank organization or register occupancy. A volatile spelling alone does not prove the backend reloaded: inspect the actual IR.
- Transfer: affine pointer/block loads select structured slice/deslice/DTE paths; a transposed layout may select transpose routines. Match the large payload, not a tiny metadata memcpy.
- Pipeline: installed GCU300 compiler explicitly calls add_triton_gcu_pingpong(pm,options.num_stages). This is a validated backend hook; num_stages and tl.debug_barrier outcomes remain empirical. No runtime overlap was established from names, static waits or timing alone.
- Scheduling: GCU300 GCUOptions defaults are warp_size=1,vector_length=512,num_warps=4. num_warps and vector_length are compiler options, not a CUDA32-thread warp model. Fixed global matrix work and varying M tiles compare program granularity without changing mathematical work.

GCU300 defaults advertise no supported FP8 dot dtypes, while GCU400/500 options differ. Compiler max_local/max_shared fields are acceptance limits; do not report them as measured hardware capacity. Generic tensor-descriptor or vendor-foreign async module availability does not establish GCU300 lowering.

## Matched one-sweep observations

CPU numerical validation ran in separate processes. Timings use3 warmups and5 groups of5 event-timed launches; buffers are reused and L2 is not flushed. Logical byte rates are not HBM measurements. Retain all samples; repeat only independently identified anomalies. These are microkernel results, not full MSA speedups.

| Before | After | Speedup | Before/after MAD % |
|---|---|---:|---:|
| matrix_m8_w1 | matrix_m64_w1 | 4.244x | 0.50/6.44 |
| matrix_m64_w4 | matrix_m64_w1 | 1.522x | 1.43/6.44 |
| reuse_reload | reuse_live | 1.971x | 0.96/0.58 |
| rank_matrix | rank_scan | 1.072x | 0.52/0.20 |
| transfer_row_ids | transfer_page_id | 35.519x | 0.03/0.42 |
| transfer_pointer | transfer_block | 0.998x | 0.72/0.31 |
| transfer_rows1 | transfer_rows128 | 107.122x | 0.13/0.72 |
| pipeline_s1 | pipeline_s3 | 0.898x | 1.13/0.37 |
| vector_fma_w4 | vector_barrier_w4 | 1.001x | 0.08/0.10 |
| softmax_n128_q1 | softmax_n128_q8 | 7.273x | 0.11/0.37 |
| softmax_n1024_q1 | softmax_n1024_q8 | 7.264x | 0.08/0.63 |
| softmax_n4096_q1 | softmax_n4096_q8 | 3.854x | 0.61/0.40 |

A ratio with high variability is diagnostic, not a stable tuning recipe. Revalidate task geometry/strides, actual lowering and end-to-end performance before reuse.

The tested stage3 pipeline creates explicit async-load/wait regions and raises local allocation73728->139264 bytes, but is slower. Stage2 retains stage1 local allocation and static wait count. Rank scanning reduces local allocation147968->1024 bytes and static waits6->3. The one-warp debug_barrier is present as gpu.barrier in GCU IR but adds no visible dedicated barrier call in final LLIR; numerical acceptance is not proof of a synchronization-engine trigger. Nondefault vector_length128/256 fail in installed-stack device linking with R_DTU_ADDR16_*_ICALL relocation errors; this is not a measured physical vector-width limit.

## Failures and bounded recovery

- vector_rope: timeout; Initial all-case process made no progress here for several minutes and was terminated through a verified task-only recovery. Compile versus execution root cause unresolved.
- vector_length128: fail; Exception('ld.lld: error: relocation R_DTU_ADDR16_LO_ICALL cannot be used against symbol tops_cvt_v4_fp32_to_v16_fp32__4; recompile with -fPIC\n>>> defined in /tmp/kernel3105739_3105739-8806fb.o\n>>> referenced by llvm-link\n>>>               /tmp/kernel3105739_3105739-8806fb.o:(vectors)\n\nld.lld: error: relocation R_DTU_ADDR16_HI_ICALL cannot be used against symbol tops_cvt_v4_fp32_to_v16_fp32__4; recompile with -fPIC\n>>> defined in /tmp/kernel3105739_3105739-8806fb.o\n>>> referenced by llvm-link\n>>>               /tmp/kernel3105739_3105739-8806fb.o:(vectors)\n\nld.lld: error: relocation R_DTU_ADDR16_LO_ICALL cannot be used against symbol tops_fma_v16_fp32; recompile with -fPIC\n>>> defined in /tmp/kernel3105739_3105739-8806fb.o\n>>> referenced by llvm-link\n>>>               /tmp/kernel3105739_3105739-8806fb.o:(vectors)\n\nld.lld: error: relocation R_DTU_ADDR16_LO_ICALL cannot be used against symbol tops_cvt_v16_fp32_to_v4_fp32__1_of_4; recompile with -fPIC\n>>> defined in /tmp/kernel3105739_3105739-8806fb.o\n>>> referenced by llvm-link\n>>>               /tmp/kernel3105739_3105739-8806fb.o:(vectors)\n\nld.lld: error: relocation R_DTU_ADDR16_LO_ICALL cannot be used against symbol tops_cvt_v16_fp32_to_v4_fp32__3_of_4; recompile with -fPIC\n>>> defined in /tmp/kernel3105739_3105739-8806fb.o\n>>> referenced by llvm-link\n>>>               /tmp/kernel3105739_3105739-8806fb.o:(vectors)\n\nld.lld: error: relocation R_DTU_ADDR16_LO_ICALL cannot be used against symbol tops_cvt_v16_fp32_to_v4_fp32__2_of_4; recompile with -fPIC\n>>> defined in /tmp/kernel3105739_3105739-8806fb.o
- vector_length256: fail; Exception('ld.lld: error: relocation R_DTU_ADDR16_LO_ICALL cannot be used against symbol tops_cvt_v4_fp32_to_v8_fp32__2; recompile with -fPIC\n>>> defined in /tmp/kernel3106222_3106222-a3af6f.o\n>>> referenced by llvm-link\n>>>               /tmp/kernel3106222_3106222-a3af6f.o:(vectors)\n\nld.lld: error: relocation R_DTU_ADDR16_HI_ICALL cannot be used against symbol tops_cvt_v4_fp32_to_v8_fp32__2; recompile with -fPIC\n>>> defined in /tmp/kernel3106222_3106222-a3af6f.o\n>>> referenced by llvm-link\n>>>               /tmp/kernel3106222_3106222-a3af6f.o:(vectors)\n\nld.lld: error: relocation R_DTU_ADDR16_LO_ICALL cannot be used against symbol tops_cvt_v8_fp32_to_v4_fp32__0_of_2; recompile with -fPIC\n>>> defined in /tmp/kernel3106222_3106222-a3af6f.o\n>>> referenced by llvm-link\n>>>               /tmp/kernel3106222_3106222-a3af6f.o:(vectors)\n\nld.lld: error: relocation R_DTU_ADDR16_LO_ICALL cannot be used against symbol tops_cvt_v8_fp32_to_v4_fp32__1_of_2; recompile with -fPIC\n>>> defined in /tmp/kernel3106222_3106222-a3af6f.o\n>>> referenced by llvm-link\n>>>               /tmp/kernel3106222_3106222-a3af6f.o:(vectors)\n\nld.lld: error: relocation R_DTU_ADDR16_LO_ICALL cannot be used against symbol tops_fma_v8_fp32; recompile with -fPIC\n>>> defined in /tmp/kernel3106222_3106222-a3af6f.o\n>>> referenced by llvm-link\n>>>               /tmp/kernel3106222_3106222-a3af6f.o:(vectors)\n\nld.lld: error: relocation R_DTU_ADDR16_HI_ICALL cannot be used against symbol tops_fma_v8_fp32; recompile with -fPIC\n>>> defined in /tmp/kernel3106222_3106222-a3af6f.o\n>>> referenced by llvm-
- bf16_fp8_roundtrip: fail; CompilationError('def fp8_cast(X,Y,N:tl.constexpr):\n    i=tl.program_id(0)*256+tl.arange(0,256)\n    x=tl.load(X+i,i<N,other=0)\n    y=x.to(tl.float8e4nv).to(tl.bfloat16)\n    tl.store(Y+i,y,i<N)\n', <ast.Call object at 0x7e9d7041eed0>, 'type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ()')

The initial suite retained21 completed cases when the XOR-indexed pair rotation stopped progressing for minutes. Only its verified task process was terminated through KernelPilot. Remaining cases used independent120-second limits. Compile-vs-execution cause is unresolved unless a specific compiler exception says otherwise; a timeout is not proof that hardware lacks the operation. Failed cases were excluded from timing.

## Operator integration gate

Choose a coherent mechanism supported by current evidence, then preserve semantic selected pages/causal masks/strides/dtypes. Run independent correctness/generalization tests and the complete frozen workload benchmark. Matrix/DTE activity is already present in MSA; new tuning must reduce actual work/traffic/waits or improve scheduling rather than claim first activation. SIP busy monitoring from S60-EVIDENCE-002 cannot establish matrix or DTE utilization.

## Recurrent matrix kernels: native paths still need layout validation

Follow-up2026-10-20 on GCU300, Triton3.6.0 and
torch_gcu2.10.0+3.7.20260408: an executed recurrent BF16/FP32 matrix
kernel already used convgen and linked m.vmm in both launch configurations.
Changing execution-group configuration while preserving outer arithmetic
changed the lowered local matrix dimensions and slice/deslice/wait
decomposition. Independent numerical guards and full-call measurements
confirmed a useful improvement in that task.

Native-matrix call presence is therefore a starting point, not evidence of
efficient decomposition. For a recurrent loop, repeated small local matrix
and layout-transfer operations can be a candidate bottleneck even when every
dot selects convgen. Compare actual local operand shapes, called primitives,
payload transfers and separately timed phases after changing launch geometry.
The tested improvement had a larger compiler shared-storage attribute;
reduced storage is not a necessary explanation. Static call counts do not
measure dynamic traffic, utilization or transfer/compute overlap.

This is not a recommendation to copy one execution-group count into every
kernel. Tile dimensions, strides, recurrence state, useful independent work
and numerical rounding remain applicability gates. Keep task-specific
configuration choices, workloads and timings outside this card. Evidence:
`chunk-kda-groups-baseline-artifacts`, `chunk-kda-groups-best-artifacts`, and
`chunk-kda-groups-validation-report` in [sources.json](sources.json).

## Supplemental multi-warp barrier check

{
  "before": "vector_fma_w4",
  "after": "vector_barrier_w4",
  "llvm_calls_identical": false,
  "instructions_identical": false,
  "llir_identical": false,
  "gcu_barrier_counts": [
    0,
    1
  ],
  "call_changes": {
    "tops_private_dte_ctr": [
      1,
      2
    ],
    "tops_syncthreads": [
      0,
      1
    ]
  },
  "synchronization_instruction_changes": {
    "l.movs.barrier.all": [
      8,
      14
    ],
    "l.movs.barrier": [
      10,
      12
    ],
    "l.wait.dec": [
      8,
      10
    ]
  }
}

Trigger: tl.debug_barrier with num_warps=4 adds tops_syncthreads and additional l.movs.barrier/l.movs.barrier.all/l.wait.dec instructions in the GCU300 ELF. The one-warp version adds no dedicated barrier call. The four-warp pair has separate numerical validation and one timing sweep;1.22193/1.22028ms is essentially unchanged, not a speedup. Use this only when multi-warp synchronization is required, and validate shared-data ordering separately: this arithmetic-only probe does not establish the full memory-ordering guarantee. Current one-warp MSA needs no added synchronization for this purpose.


## Compound integer reductions can restrict execution-group tuning (2026-10-22)

On GCU300/Triton3.6.0 with the default vector_length512, a compound native-matrix
score kernel followed by unsigned/integer Top-K reductions can fail device
linking after changing execution groups, even though standalone matrix probes
with that group count have executed successfully. Emitted
`tops_maxui_v2_ui32`, `tops_minsi_v2_si32` and, in a streaming sample,
`tops_minui_v2_ui32` calls produce R_DTU_ADDR16_LO_ICALL/HI_ICALL relocation
errors on the installed library/compiler stack. This extends the earlier
nondefault-vector-length failures; default vector length is not a sufficient
whole-kernel compatibility check.

A less aggressive group configuration passes the original small numerical
suite yet encounters the same linking issue on a wider direct reduction tile.
Therefore validate all formal compiler layout classes and untuned dtypes,
strides and tails before promotion. A passing small correctness suite does not
establish buildability of every larger specialization. Preserve partial timing
rows without claiming an aggregate for the failed variant, and restore best.

Retaining the known-compatible decomposition for the affected wider layout
while changing the narrower layout is a source-level compatibility mechanism.
It preserves outer arithmetic, score precision, masks and deterministic order.
Match the generated local matrix/reduction shapes and emitted library calls;
do not substitute a table of public sequence lengths or assume a universal
warp-count rule. Exact tile/group choices and performance remain in task data.

Runtime-bound captures and separately timed full calls confirm a useful bounded
decomposition change while preserving convgen, matrix instructions and affine
payload loads. Static resource attributes and DTE waits are not hardware unit
counts, dynamic traffic, occupancy or utilization. Neither relocation errors
nor their `recompile with -fPIC` text proves absent GPU functionality or permits
changing the installed SDK/driver/library during a source-only task.

Sources: `moba-round2-paths`, `moba-round2-link-failures`, `moba-round2-guards`,
`moba-round2-evaluation` and `moba-round2-report` in [sources.json](sources.json).

## Execution groups also change recurrence and transfer lowering (2026-10-23)

In a native-matrix attention recurrence, an execution-group change can alter
both local convgen tile decomposition and row softmax/reduction/transposition
layout while preserving outer logical tiles and FP32 recurrence work. Matched
actual run-return captures with equal tensor roles, shapes, strides and semantic
constants show a wider local native matrix tile, fewer statically emitted DTE
wait and synchronization call sites, and unchanged declared shared memory.
Separate frozen forward/backward checks, untuned strided/GQA cases and complete-
call event timing support a useful bounded change on the reviewed stack.
The baseline capture is accepted only after its source hash matches the fresh
comparator. Retain structured payload paths and matrix ISA evidence for both
sides; a changed group number alone establishes none of these facts.

This extends the earlier chunk-attention observation to a routed recurrence;
it does not prescribe one universal group count or justify shrinking numerical
work. Static waits, shared bytes and call sites are not dynamic traffic, engine
utilization, occupancy or isolated stall attribution. A faster measured phase
is consistent with the combined lowering change, without proving which changed
component causes the gain. Preserve integer-reduction incompatibility evidence
above; validate all compiler classes before timing or promotion. Exact tuned
choices, shapes and timings remain in task evidence. Sources:
`moba-remain5-matched-paths`, `moba-remain5-environment`,
`moba-remain5-group-evaluation`, and `moba-remain5-group-guards`.

## Tuple reductions need a total order and separate promotion gates (2026-10-23)

On the reviewed GCU300/Triton3.6.0 stack, a custom associative `tl.reduce`
can carry unsigned score keys together with signed indices and a slot field
inside a compound matrix/Top-K kernel. A tested retained-state comparator orders
lower unsigned score first, then higher signed index, then lower slot. This
reproduces the original worst-entry and first-matching-slot selection without
packing values into an int64 key. Inactive lanes need sentinels that cannot win
against active entries under the **complete tuple order**, including when every
active key is maximal. Empty valid lanes and duplicate pairs still require a
deterministic slot tie-break. Associativity follows from selecting one element
under that total order; comparisons must preserve signed/unsigned types.

Frozen numerical tests, independent CPU stable selectors and exact state/slot
comparisons pass for the reviewed three-field expression. State probes cover
empty-to-full insertion, no replacement, mixed rows, equal scores, duplicate
retained pairs, padded ranks and high unsigned keys. The whole-call performance
candidate is nevertheless rejected by an untuned regression cap. Do not infer
that fewer source-level reduction expressions mean fewer hardware passes,
staging operations or a useful speedup. Small changes near a cap need honest
sampling-policy interpretation; no profiler isolated the regression's cause.
Task-specific numbers and shape decisions remain outside this card.

Candidate maximum-key/minimum-index selection is a distinct two-field order;
zero/invalid keys require the original sentinel index. A separately gated
two-field expression also passes frozen tests and the independent exact CPU
selector/state cases on the reviewed stack, yet full-call formal and untuned
classes regress and the candidate is rejected. There is no actual runtime-bound
capture of either rejected candidate in this task; final source-bound captures
document restored source only. A physical cause or fewer emitted transfers is
not established for the rejected expressions. Each expression needs its own
performance gates; compatibility and performance of one retained-state tuple
do not establish either property for a different reduction axis, payload or
compiler layout. Preserve prior unsigned-
reduction linking and dynamic-loop failure evidence; this fixed-loop mechanism
does not supersede those boundaries or prove a universal implementation.

Evidence in sources.json: `moba-reduce6-r601-source`,
`moba-reduce6-r601-correctness`, `moba-reduce6-r601-state`,
`moba-reduce6-r601-evaluation`, `moba-reduce6-r602-source`,
`moba-reduce6-r602-correctness`, `moba-reduce6-r602-state` and
`moba-reduce6-baseline-state`.

## Query-row tiling can also expose integer-reduction link limits (2026-10-23)

On the reviewed GCU300/Triton3.6.0 stack, changing only the streaming query-row
and retained-state budget, while keeping score columns and execution groups,
passes the original forward/backward suite but fails a separate BF16 packed
streaming selector class during device linking. The linker reports
R_DTU_ADDR16_LO_ICALL / HI_ICALL against tops_maxui_v2_ui32,
tops_minsi_v2_si32 and tops_minui_v2_ui32. The preceding non-batched selector
control passes. This extends the earlier execution-group/layout limitation:
query-axis tiling alone can change integer reduction library selection.

Do not infer that larger query tiles work from native dot geometry or the
original operator suite alone. Cover the affected streaming branch, ragged
packed sequences and integer tie/selection work before timing. This is a
version/layout-scoped linking failure, not proof of absent hardware reduction
or permission to rebuild the SDK with PIC. The candidate was rejected before
performance measurement and the previous correct checkpoint restored. There
is no candidate runtime ISA or speedup claim for this failing class. Task
constants, shapes and raw failure remain in external hashed evidence.

Sources: moba-query9-source, moba-query9-frozen-suite,
moba-query9-selector-failure and moba-query9-candidates in sources.json.

## Joint query-axis and execution-group revalidation (2026-10-23)

Follow-up on the preceding query-axis link failure demonstrates that the
limitation belongs to a generated layout class. Repartitioning the expanded
streaming query tile across more execution groups allows the previously
failing packed BF16 selector to compile and execute with exact CPU ordering.
An alternative smaller query tile with the original execution-group policy
also executes that class. Both alternatives pass separate FP16 strided cases,
high-rank/tail selectors, pinned-original random-index comparisons and exact
unsigned retained-state tests on the reviewed GCU300/Triton3.6.0 stack.

Therefore a failed integer-reduction relocation is a reason to revalidate
joint tile/group decomposition, not to declare the larger algorithm or chip
incapable. No SDK or library rebuild was used. Compatibility of these tested
expressions is not a general group-count recipe: matrix local shape, shared
storage, staging and program parallelism may change together. Native matrix
and structured transfer coverage does not establish performance improvement;
complete-call formal and untuned regression gates remain independent.

The expanded query/group layout retains the parent's local native matrix
callee while increasing declared shared storage. The smaller-query alternative
retains native matrix ISA but changes the local matrix decomposition and
reduces declared shared storage, yet fails complete-call and untuned performance
gates. Lower resource metadata and matrix-path presence therefore do not imply
better scheduling or speed; these combined changes do not isolate a cause.

Formal returned-kernel captures are source-bound, but they do not capture
the separate packed correctness specialization's exact callee. No claim of
physical utilization, isolated instruction cost or statistical significance
follows from these compiler/runtime checks. Tuned constants, shape tables,
timings and decisions remain in the external task workspace.

Sources use the moba-querylayout10-* prefix in sources.json.
