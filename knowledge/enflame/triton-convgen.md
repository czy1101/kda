# S60-TRITON-001: tile padding and local alignment can select a different dot path

## Scope and status

Observed 2026-10-14 on S60/GCU300, Enflame Triton 3.6.0, BF16 inputs and FP32
accumulation. Compiler metadata: `num_warps=4`, `num_stages=1`,
`vector_length=512`. Driver/SDK versions were not established by this card;
capture them for a reproduction. Status: compiler-confirmed path switch and
historical phase timing. Physical execution-unit mapping is unverified.

## Observation

For the newer bounded local-shape selection predicate, see the
2026-10-21 controlled-boundary follow-up at the end of this card.

The source still uses `tl.dot`; `convgen` is a backend-selected device-library
entry, not an API to substitute into the Triton source.
No general "enable convgen" switch has been verified. The same-sized dot in
a standalone kernel still needs its own lowering inspection. For architectural
context and transfer/storage markers, see
[S60-TRITON-002](triton-compute-memory-paths.md).

| Outer dot M x N x K | Local dot M x N x K | Observed LLVM callee | Shared metadata |
|---|---|---|---:|
| 64 x 32 x 128 | 32 x 16 x 128 | `tops_matmul_bf16_fp32` | 32768 bytes |
| 64 x 64 x 128 | 32 x 32 x 128 | `tops_convgen_m32_n32_k128_bf16_fp32` | 49152 bytes |

```python
# Schematic only; layouts/load masks are part of the real implementation.
# a: [64, 128] BF16; b: [128, 64] BF16
scores = tl.dot(a, b, out_dtype=tl.float32)
```

Here K=128 is the multiplication reduction dimension (head dimension in the
routing dot), **not MoBA routing Top-K**. The original inspection includes two
narrow and three padded cached variants. GCU IR shows the local result and
operand shapes; LLVM IR shows the actual call. A declaration alone is not
proof that the kernel calls that entry.

The B2x4096 routing diagnostic decreased from approximately 362 ms to 17 ms
(about 21x) after padding; this is a routing-phase observation, not a 21x
end-to-end claim. More columns and higher shared metadata accompanied the
improvement. The supported inference is a compiler computation-path change,
not less arithmetic, reduced precision or reduced shared-memory demand.

## Candidate mechanism and correctness

For a similar measured dot bottleneck, test padding the outer N dimension to
reach a favorable *lowered* tile. Initialize invalid dot operands to zero and
exclude padded score columns from routing/softmax using validity masks.
Zero scores alone are unsafe: they can beat valid negative routing scores.
Keep causal masks, original chunk IDs and tie ordering intact. Test partial
tiles, negative scores, equal-score ties, short inputs and Top-K saturation.

## Reproduction and falsification

1. Preserve the narrow baseline and exact compiler/environment/source hash.
2. Compile both variants through the contracted runner. Retrieve the matching
   metadata, source/TTIR, GCU IR and LLVM IR through the controlled adapter.
   Associate artifacts with source hash, specialization and launch options;
   unrelated cached kernels are not evidence for the current candidate.
3. Check outer `tt.dot`, local `gcu.matmul`, and actual LLVM `call` sites:

   ```bash
   rg -n 'tt.dot|gcu.matmul|tops_convgen_|tops_matmul_' <local-artifacts>
   ```

4. Run separate correctness; measure the same routing diagnostic and full
   contracted operator, including padding overhead, on all required shapes.
5. Reject the hypothesis if the expected lowering does not occur, correctness
   fails, or net performance/regression rules fail. A callee change alone is
   insufficient to promote.

## Limits and evidence

N=64 is an observed configuration, not a necessary/sufficient rule. Changes
to M/N/K, dtype, strides/layout, launch parameters, compiler or device library
require new artifact inspection and timing. Backend splitting explains the
observed dimensions; its general policy and final hardware instructions were
not established. Do not equate `convgen` to a particular NVIDIA instruction or
assert that `tops_matmul` cannot use matrix hardware. Torch BMM is a different
compilation path; this recipe does not prove its library dispatch.

Sources: `convgen-analysis`, `convgen-probe`, `convgen-phases-before`, and
`convgen-phases-after` in [sources.json](sources.json). The analysis records
that the cached `libdevice.bc` path was absent during follow-up inspection;
entry definitions were not resolved.

Follow-up 2026-10-16: [S60-TRITON-003](triton-streaming-score-tiles.md)
records an independently validated streaming-branch case, plus observed FP16
K64/K128 callees. Check fallback branches separately: a direct branch already
using padding does not prove its longer-input streaming branch does so.

Independent follow-up: [S60-TRITON-005](triton-hardware-trigger-probes.md)
validates a masked-load N32-to-N64 standalone dot without changing B's
physical layout or the output shape, shows corresponding `m.vmm`
instructions, and records contrary high-variance matrix timing repeats.

## Operand orientation follow-up (2026-10-20)

On GCU300 with Triton3.6.0, BF16 operands and FP32 accumulation, an
independently executed paired score kernel confirms that algebraically
transposing the output can change the selected computation path. For small
query/head dimensions, `K[N,D] @ Q[D,HQ]` calls `tops_matmul_bf16_fp32`,
whereas `Q[HQ,D] @ K[D,N]` calls a native `tops_convgen_*` entry. Matching
linked final code has no `m.vmm` in the former specialization and contains
`m.vmm2.mode17.bf` in the latter. A wider-HQ control already uses convgen
in both orientations, so orientation is not a universal enable switch.

This transformation changes load indexing and the reduction axis, not the
underlying Q/K tensor storage or dtype; no host transpose/copy is required.
For per-query token maxima, reduce the first output along token axis0 or
the transposed output along token axis1. Check masks, tails, output indexing
and forced-score semantics independently. Paired numerical checks passed
within tolerance; do not infer general bitwise equivalence or hardware-unit
utilization from these artifacts. This is an isolated mechanism verification,
not a reconstruction of an earlier historical binary or a full-operator win.

Exact specialization boundaries, source substitutions, timings and artifacts
remain in the task workspace. Sources: `msa-score-orientation-validation`,
`msa-score-orientation-artifacts` and `msa-score-orientation-report` in
[sources.json](sources.json).

## Shared-RHS vector reductions can form matrix rows (2026-10-20)

On GCU300/Triton3.6.0, independent query heads sharing the same KV head
can be assembled as rows of a bounded query tile. This converts separate
vector score and weighted-value reductions into matrix products without
changing which KV elements each useful row consumes. Group only rows with
the same RHS and preserve per-row maxima, normalization, masks and output
ownership. Account for padded rows and reduced program parallelism.

An executed BF16 decoder probe used padded M16/N128/K128 score products
and FP32 weighted-value products. Actual calls selected BF16/FP32 convgen
and FP32/FP32 bias-convgen with linked `m.vmm`; the per-head vector
comparator had neither matrix calls nor observed `m.vmm`. Keeping softmax
weights and V in FP32 avoided introducing a BF16 probability cast merely
to select a matrix path. This is not a general FP32 accuracy or shape rule.

The integration retained the existing D64/D128 public interface and bounded
grouping. Independent BF16/FP16 CPU-reference checks include group and token
tails, windows and a larger group; frozen integration checks cover runtime
lengths and noncontiguous inputs. Code generation and numerical acceptance
do not establish full-workload promotion, useful overlap or physical-unit
utilization. Task-specific dispatch sizes and timings remain in the workspace.
Sources: `parallax-native-baseline-artifacts`,
`parallax-grouped-decoder-artifacts`, and `parallax-grouped-decoder-guards`
in [sources.json](sources.json).

## Packing and reduction-width follow-up (2026-10-20)

Executed paired probes on GCU300, Triton3.6.0 and
torch_gcu2.10.0+3.7.20260408 confirm three additional mechanisms:

- Two narrow products sharing a left operand can pack their independent
  right operands into wider output columns, then split the results. This
  selected native BF16/FP32 convgen with linked matrix instructions where
  separate products selected generic matmul. Transposing the separate
  products alone did not produce that switch.
- Independent small triangular systems can form a block-diagonal system.
  The tested FP16 polynomial inverse then selected native convgen and
  bias-convgen instead of generic matmul/addmm. It remains a sequence of
  matrix operations, not a special single-instruction inverse.
- Increasing the reduction width of a BF16/FP32 state-update product changed
  generic matmul/addmm into a native matrix path in both isolated code and
  a complete recurrent-kernel context.

Launch geometry is part of each result: a higher-group control reverted the
packed-output and block-diagonal cases to generic paths. This is not a
universal minimum-width rule, a recommendation to copy one group count, or
evidence of a measured utilization percentage. The complete recurrent kernel
already used native projections before the tested update-path switch; audit
each operation rather than calling the entire earlier operator non-native.

The complete before-variants were reconstructed in temporary source copies,
not recovered historical binaries. Independent references passed; no timing
or end-to-end speedup was measured in this audit. Exact source substitutions,
shapes, group controls, decoded-instruction limitations and hashes stay in
the task report. Sources: `kda-historical-path-probes`,
`kda-historical-path-context`, and `kda-historical-path-report` in
[sources.json](sources.json).

## Controlled local-alignment boundary follow-up (2026-10-21)

On GCU300, Triton3.6.0 and torch_gcu2.10.0+3.7.20260408, controlled
experiments support the following bounded empirical selector for positive,
statically shaped local `gcu.matmul` dimensions:

| Input / output types | Observed convgen predicate |
|---|---|
| BF16 / FP32 | `local_N % 32 == 0` and `local_K % 32 == 0` |
| FP16 / FP32 or FP16 | `local_N % 32 == 0` and `local_K % 32 == 0` |
| FP32 / FP32, `input_precision='ieee'` | `local_N % 16 == 0` and `local_K % 16 == 0` |

These are dimension alignments measured in elements, not pointer-address
alignment conditions. A minimum-M or M-alignment condition was unnecessary
in the sampled operations: even M1 selected convgen. Non-power-of-two local
boundaries distinguish divisibility from a size threshold; merely increasing
N or K beyond the alignment unit is insufficient.

The predicate agreed with all complete Triton compilation controls and
synthetic local-op boundary/joint-dimension controls. Separate representative
device tests passed CPU references and matched selected convgen calls to
linked matrix instructions. NN/NT/TN physical operand storage, dense output
addition and stage controls retained the classification at the sampled
geometries. Exact configurations, counts and artifacts remain in the task.

Apply the predicate to the actual GCU IR operand shapes. Warp splitting and
larger-kernel context can change those shapes; there is no general formula
substituting outer N/num_warps for local N. This explains why packing or
changing group count can alter selection without being independent enable
switches. Historical examples and older contrary observations remain valid
within their stated scope.

### Reuse procedure

1. Match the device, compiler/library versions, operand/output types and
   precision request to this scope. Revalidate after a version change.
2. Inspect each actual local `gcu.matmul` in the target specialization and
   branch. For `A[M,K] @ B[K,N]`, apply the table to local N and K in elements.
3. Confirm the prediction at actual LLVM `call @tops_convgen_*` sites,
   including bias variants, and inspect the matching linked binary for
   `m.vmm*`. Bind artifacts to source, specialization and launch configuration;
   a declaration alone is insufficient, and compilation alone is not execution.
4. If padding, packing, orientation or group count changes the local shape,
   inspect lowering again. Validate masks, tails, reductions and output
   semantics in separate numerical tests. Measure full-call performance with
   a performance-only benchmark, including added arithmetic and storage cost.

Aligned local dimensions predict selection within the observed scope; they
do not establish a speedup. A single group or a large outer N is not an
independent selection condition. Worked shape/group controls and experiment
counts remain in the external `convgen-dispatch-report` evidence.

The installed native selector implementation was unavailable as source.
This empirical rule is not a source-level proof for every shape, layout,
compiler/library version or architecture. Synthetic local-op lowering does
not establish linking, resource feasibility, execution, accuracy or speed for
every selected dimension. The original follow-up did not cover FP8, integers,
BF16 output, arbitrary dynamic or strided local memrefs or other precision
modes; the extension below separates newly tested interfaces and evidence levels.
Nor does the rule identify a physical execution unit or establish that all
generic matmul implementations avoid matrix instructions.

Sources: `convgen-dispatch-analysis`, `convgen-dispatch-shapes`,
`convgen-dispatch-reduction`, `convgen-dispatch-controls`,
`convgen-dispatch-local-boundaries`, `convgen-dispatch-local-cross`,
`convgen-dispatch-executed-tests`, `convgen-dispatch-environment` and
`convgen-dispatch-report` in [sources.json](sources.json).

## Output types, frontend gates and wider contexts (2026-10-21 extension)

Distinguish input storage type, requested `tl.dot` output, actual local
matmul result, and final tensor storage. Executed BF16-input/BF16-storage
kernels use an FP32 local result and convert on storage; their selection
still follows local BF16/FP32 N/K32 alignment. Changing final storage alone
does not establish a different dot-result type. With BF16 or FP32 inputs,
even a requested FP16 dot result remained FP32 in the tested complete kernels.
Inspect TTIR/GCU IR rather than inferring the local type from that argument.

The installed semantic implementation explicitly rejects BF16 `tl.dot`
output and recommends an FP32/FP16 result followed by conversion. Direct
local IR can nevertheless lower BF16/BF16, FP16/BF16, BF16/FP16 and
FP32/BF16 or FP16 result operations. Their tested static grids match the
input-type alignment units above. Those new direct-result library calls
were not linked or executed; this does not make BF16 `tl.dot` output a
supported frontend interface. See [S60-TRITON-011](triton-gcu-indexing-and-compiler-limits.md)
for precision, mixed-type and FP8 frontend gates.

Executed masked logical tails, reduction loops, explicit dot accumulators,
output addition, transposed/strided/offset global inputs, final storage
conversion and launch/stage controls retained the floating-point selector
when applied to actual local operations. Logical N/K can be unaligned while
the padded local tile is aligned. Total loop K can differ from the local
reduction block. Explicit local-stride and dynamic-memref probes provide
only lowering evidence, not arbitrary-stride runtime correctness.

The complete GCU300 frontend accepted only IEEE precision in these probes.
Synthetic local IR additionally accepted TF32/TF32x3 and matched local
N/K16 selection; this establishes neither normal Triton support for those
requests nor their numerical precision semantics. Invalid precision/type
combinations are rejections, not successful generic fallbacks.

Executed INT8/INT32 kernels call `tops_matmul_si8_si32` and contain linked
`m.vmm2.mode19.s.acc0`, with exact CPU-reference equality. This is direct
contrary evidence to equating every generic callee with absence of matrix
instructions. Keep integer dispatch outside the floating-point convgen
predicate. ISA presence does not identify physical-unit utilization or speed.

Sources: `convgen-extension-analysis`, `convgen-extension-local`,
`convgen-extension-compiler`, `convgen-extension-device`,
`convgen-extension-semantic`, `convgen-extension-preflight`,
`convgen-extension-integrity`, `convgen-extension-manifest` and
`convgen-extension-report` in [sources.json](sources.json). Exact cases,
counts, numerical policy and rejection logs remain in the task workspace.

## Recurrent fallback decomposition follow-up (2026-10-22)

Executed GCU300/Triton3.6.0 FP16 and BF16 recurrent-state probes confirm
that changing execution-group splitting, while preserving the outer tile,
arithmetic and precision, can widen an unaligned local output dimension into
an aligned one. The actual local operation changes from generic floating-point
addmm without observed linked matrix instructions to bias-convgen with linked
`m.vmm`. Forward, recomputed forward and backward state variants were checked.

Independent CPU FP64 output, final-state and all-input-gradient checks pass
for paired tail, rectangular-state, alternate-layout and variable-length
examples. Structured payload transfers remain present in both variants.
This supports the local-alignment mechanism; it is not a universal execution-
group recommendation, full-contract correctness result or latency improvement.
The parent operator's separate original numerical gate failed, so no full
performance comparison or deployment was performed. Exact launch choices,
cases, binaries and rejected promotion remain in the task workspace.

Sources: `gla-fallback-trigger-probe`, `gla-runtime-path-audit`,
`gla-path-analysis` and `gla-numerical-blocker` in [sources.json](sources.json).


## Full integration follow-up after the bounded activation probe

The same GCU300/Triton3.6.0 recurrent fallback activation was integrated into
the complete operator and passed the frozen numerical suite plus independent
CPU FP64 tail,rectangular-state,alternate-layout,variable-length and strong-
gate guards. Separate performance-only public-call sweeps confirm a bounded
benefit in the fallback workload class. Full formal per-shape promotion rules
remain independent; a large class-specific gain did not justify deploying a
candidate that failed those rules. Exact launch choices,shape/timing tables,
repeats and candidate decisions remain outside the knowledge base.

Sources:`gla-native-matrix-confirmation`,`gla-integration-correctness` and
`gla-optimization-final` in [sources.json](sources.json).
