# S60-TRITON-010: live tiles, mask layout and output ownership

## Scope and status

Reusable design mechanisms for reductions, selection, compaction and
partitioned normalization. Compiler observations were made on S60/GCU300,
Triton3.6.0 and torch_gcu2.10.0+3.7.20260408. They establish mechanisms and
tested semantic boundaries, not universal tile sizes or performance gains.

## Bounded live tiles

When independent rows repeat the same reduction or selection passes, a
bounded row tile can amortize small-program overhead and retain intermediates.
Select the tile using live elements/bytes, element type, padding and available
parallel work. Larger tiles can increase temporary storage or reduce useful
parallelism. Revalidate the complete call; do not import another operator's
row count, launch options or resource budget as a default.

### Native matrix selection does not guarantee feasible fusion

On the reviewed GCU300/Triton3.6.0 stack, padding narrow FP32 weighted
reductions into matrix rows/columns selects native convgen and linked matrix
instructions, but full-width expanded intermediates can exceed the compiler's
local-memory allocation budget. A local-memory OutOfResources error is not
HBM exhaustion, and the reported failed allocation does not establish the
total peak deficit. Bound every live padded operand and result, not only the
dot reduction block.

Splitting scoring, weighted mixing and optional output normalization can make
that computation executable while preserving FP32 products/probabilities and
independently validated outputs. It also introduces launches, score/output
workspace and input rereads. Actual native matrix instructions therefore do
not establish a complete-call advantage over an existing vector fusion.
Paired full-call measurements in the bounded follow-up reject both tested
split organizations; no universal matrix-versus-vector winner is implied.
Do not add low-precision intermediate casts solely to select a matrix path.
Sources: `attnres-matrix-trial-analysis`, `attnres-matrix-trial-checks`, and
`attnres-matrix-trial-report` in [sources.json](sources.json). Task-specific
padding, timings, failures and candidate history remain in that workspace.

Independent bounded chunks may also be processed sequentially inside one
program to amortize program entry and metadata work. Reset each chunk's
pointer bases and local state, validate a partial final group, and preserve
valid chunk-to-sequence mapping. This is distinct from sharing a recurrence
state between unrelated chunks. Fewer independent programs can reduce useful
parallelism, so validate full-call performance and generated loop/transfer
paths rather than assuming amortization wins.

## Prefix ranks for compaction

Given an already established Boolean keep/first-occurrence mask, the ordered
destination rank is inclusive_cumsum(mask_as_int)-1. This replaces an extra
pairwise rank construction while preserving the order of retained elements.
It does not itself deduplicate values or establish which elements to keep.
Applications include sparse-list packing and stream compaction. Check empty
and padded inputs, stable ordering, integer overflow and the installed scan
lowering. See S60-TRITON-009 for independent scan probes.

## Match validity masks to the compute layout

Express query/head or row/group validity directly in the matrix/reduction
layout when this is semantically equivalent to broadcasting and reshaping a
higher-rank mask. The reusable opportunity is avoiding expanded temporaries
and layout conversions. It does not remove padding or causal validity checks.
Verify compiled temporary resources and full-call performance; a rank change
alone is not proof of a faster kernel.

## A single owner can remove an identity merge

If one partition owns the complete normalized output and no consumer needs
its intermediate statistics, it can write the final destination directly.
This may remove temporary output/statistics and a merge that has become an
identity operation. The argument applies to split reductions and normalized
weighted sums, not only to a particular operator.

Before doing so, establish complete ownership, destination strides and alias
safety, accumulation/rounding equivalence, and whether gradients or downstream
consumers require the intermediate values. Multiple partitions still require
the original mathematically correct combination.

## One owner can accumulate all shared-destination contributions

When several logical producers contribute to the same output tile, a single
program may accumulate their contributions and write the final tile directly.
This can remove producer-sized temporary buffers and a separate fold while
retaining a shared input tile. It is distinct from bypassing an identity merge:
the owner must perform every original contribution, with valid masks/strides.

Accumulating FP32 contributions before the final cast is not bitwise equivalent
to summing already-rounded per-producer partials. Validate outputs and gradients
against an independent high-precision reference, including group counts, tails,
padded dimensions and windows. Dispatch must also match the actual launch
branch: a fixed short kernel that still writes per-producer outputs cannot use
a final-output-sized destination merely because another branch supports it.

More serial work and fewer independent programs can offset saved buffers and
launches. Preserve enough independent work, inspect the last actually selected
launches, verify destination tensor dimensions and the absence of a fold, and
measure the complete call across all required shapes. A native matrix path may
already exist on both sides; this transformation need not activate a new unit.
The GCU300 follow-up retained convgen and linked matrix instructions. Static
call evidence does not establish physical utilization or DRAM traffic. Sources:
`parallax-ownership-lowering`, `parallax-ownership-checks`, and
`parallax-ownership-path-checks` in [sources.json](sources.json).

## Degenerate axes require separate layout validation

A singleton axis with a nonunit input stride can take a different compiler
path from the corresponding multirow tile. A numerically equivalent 1D
expression can be a compatibility fallback only after it passes independent
checks of indexing, empty inputs, ties, exceptional values and output strides.
Do not infer that all singleton/strided expressions fail, or that constexpr
strides or a 1D rewrite automatically repair an unrelated expression.

## Defer a row reduction under uniform rescaling

For a streaming recurrence `L_next = alpha * L + sum(p)`, a per-lane
state can instead follow `v_next = alpha * v + p`, with `L = sum(v)`
after the loop. This removes the repeated sum when no intermediate consumer
needs L. Alpha must be identical across lanes of each row on each iteration,
and the lane extent and padding semantics must stay consistent. A preexisting
scalar state can seed one lane while all other lanes start at zero.

The extra FP32 tile changes live storage and accumulation order. Algebraic
equivalence is not bitwise floating-point equivalence. Verify precision,
exceptional values, masked rows, compiler lowering and the complete call;
additional local storage can outweigh fewer reductions. Maxima or other
statistics still needed inside the loop cannot be deferred by this argument.

## Runtime-verified common work with private continuation

When neighboring rows have the same ordered prefix of selected inputs, compare
the actual IDs and availability at runtime before executing that prefix as a
useful row tile. Stop at the first mismatch; do not infer equality from a
selection policy or reuse rows with different payloads/scales. Retain each
row's own validity and causal mask. Padded rows must not influence equality.

A normalized weighted sum can continue from a partial maximum, denominator
and unnormalized FP32 numerator. Keep these statistics in a compatible scale
and preserve selection order. Rounding an intermediate normalized output or
adding partial normalized outputs is not an equivalent continuation.

Bound live matrix rows, workspace bytes and independent program count. Extra
launches, allocation and state stores/reloads belong inside complete-call
timing, including the no-common-prefix case. Arbitrary suffixes still require
their original semantics. Reduced logical loads do not prove reduced physical
DRAM traffic without counters; cache reuse may already exist.

## Disjoint output regions can avoid repeated matrix products

When one left operand is multiplied separately by disjoint column-masked
versions of the same right operand, those products may be combined using
the original right operand. Each retained output column still has the same
reduction inputs. Preserve operand rounding and every safety mask: merging
another row block can reintroduce an unused overflowing region, even when a
later output mask would discard it. Algebraic equivalence alone does not
guarantee bitwise equivalence or faster lowering.

A GCU300/Triton3.6.0 BF16/FP32 follow-up confirmed fewer actual convgen
calls, independent tail/gate/state checks and a modest full-workload gain.
Pairing additional outputs into a wider matrix instead regressed in that
task. Wider matrices and fewer source dots are hypotheses, not promotion
criteria; inspect layout costs and measure the full call.

## Bounded exponential factors can be reused with precision checks

For a cumulative gate x and a shift c, a retained factor E=exp(x-c) can
supply exp(x) as E*exp(c), and a retained inverse factor exp(c-x) can supply
exp(y-x) as exp(y-c)*exp(c-x). Bound every factor's range before relying
on this transformation. Intermediate overflow or premature underflow may
break a mathematically valid identity. Preserve masks and downstream dtype
rounding; compare long recurrences, tails and extreme gate values separately.

The same follow-up verified fewer special-function call sites but only a
small incremental end-to-end gain. Replacing the inverse exponential with
a reciprocal also regressed in that task. Extra live tiles, broadcasts and
arithmetic may offset recomputation savings; those mechanisms require their
own artifact evidence, not an assumed explanation from wall time. Sources:
`chunk-kda-k1-mechanisms`, `chunk-kda-k1-score-artifacts`, and
`chunk-kda-k1-exp-artifacts` in [sources.json](sources.json).

## Evidence and reuse

### Midpoint centering does not guarantee finite exponential factors

A GCU300/Triton3.6.0 BF16 gated-recurrence follow-up reproduces nonfinite
score intermediates, outputs and value gradients from finite inputs even
though cumulative gates, recurrent states and a CPU FP64 recurrence remain
finite. Matched matrix/transfer paths are already active; native computation
does not imply numerical safety. Separate bounded smaller-region controls
restore finite, reference-accurate outputs, final states and all input gradients.
No full correctness or performance promotion followed this diagnostic.

For natural-exponent gates with range Delta, midpoint-centered factors can
reach exp(Delta/2). They can overflow FP32 when Delta/2 exceeds its exponential
range, even when the original causal factor exp(g_i-g_j) is bounded. Products
with Q/K and accumulation can impose stricter limits. BF16 input/final storage
does not protect a computation whose intermediate factors are FP32. Applying
a causal mask after a dot cannot repair intermediate Inf or zero-times-Inf.

Bound the actual factors and downstream rounding, or retain a stable direct-
difference/recurrence fallback. A smaller fixed tile only establishes safety
for the tested gate range, not arbitrary negative gates. Do not silently clip
exponents, loosen tolerance or discard the extreme class to enable matrix
performance work. Keep failed original gates and any bounded diagnostic
controls separate from a correctness-approved candidate.

Sources: `gla-centered-exp-diagnosis`, `gla-numerical-blocker` and
`gla-path-analysis` in [sources.json](sources.json).

Catalog source IDs point to external compiler artifacts and independent
numerical guards. Task-specific tuning, workload tables, candidate history
and end-to-end speedups remain in the originating task workspace. Revalidate
each mechanism against the new workload's semantics and complete cost.


## Integrated exponential-range guard follow-up

A complete GCU300/Triton3.6.0 integration resolves the previously observed
strong-negative-gate failure by selecting a stable direct causal-difference
score calculation when centered factors exceed a device-side range guard.
The bounded case retains its original FP32 matrix computation; direct and
split-key paths use the same safety mechanism. Frozen and independent output,
state and all-input-gradient tests pass across the recorded dtype/layout scope.
No exponent clipping or relaxed numerical threshold is used.

This supports switching between equivalent algorithms on actual factor range,
rather than assuming that midpoint centering or a fixed small tile is always
safe. Keep headroom for operand products and accumulation. A guard for one
factorization does not prove that every later recurrence/gradient path handles
arbitrarily extreme inputs; each still needs its own numerical evidence.
The task retains this correctness repair even when later performance variants
fail full-workload regression rules. Exact guard choices and test classes stay
in the task evidence,not as universal constants in this card.

Sources:`gla-range-guard-correctness`,`gla-range-guard-extra`,
`gla-retained-runtime-audit` and `gla-optimization-final` in
[sources.json](sources.json).


## Normalizing a weighted mean and choosing a probability-output boundary

For a weighted mean o=u/s followed by output RMSNorm, a positive finite
scalar s permits the equivalent expression
u*rsqrt(mean(u*u)+eps*s*s)*weight. The epsilon must scale with s squared.
This can remove a full-width division before the row reduction. It applies
when no other consumer needs the normalized intermediate. It does not imply
bitwise equivalence: changed rounding and the larger unnormalized square
can alter range safety. Bound the denominator and all intermediate values,
and independently check zero/tiny residuals, larger eps, padded dimensions,
output dtype and probability semantics. A max-shifted softmax denominator
is bounded by the source count for finite scores, but this does not by
itself prove the numerator square cannot overflow.

When a streaming weighted reduction optionally returns all source
probabilities, retaining source scores inside the main program can remove
framework postprocessing calls. It also adds probability work per program
and may change live storage or scheduling. An alternative is to preserve
the main program's logits/LSE and combine scale, subtract and exponential
into a single bulk postprocessing kernel. In-place conversion is safe only
when each logit element has one owner and no consumer still needs logits.
Measure both boundaries across small and large row counts; fewer launches
do not establish a complete-call win. Timing alone does not identify local
resource pressure or physical traffic as the cause of a regression.

Independent numerical checks and matched full-call comparisons exercise
these mechanisms on the reviewed GCU300/Triton3.6.0 stack. The evidence
includes a rejected all-probabilities-in-main candidate; no universal
fusion policy or row tile follows from this single operator. Exact timings,
decisions and constants remain outside this knowledge card.

Sources: `attnres-followup-formal`, `attnres-followup-confirm`,
`attnres-followup-probability-main`, `attnres-followup-probability-post`,
`attnres-followup-numerics`, `attnres-followup-post-numerics` and
`attnres-followup-post-source` in [sources.json](sources.json).


## Program-capacity predicates still need adjacent tail checks

A device-capacity predicate that retains several row-program waves is a
scheduling hypothesis, not a sufficient performance guard. A GCU300 follow-up
passes its formal cases with such a predicate yet regresses at an adjacent
held-out row count after a partial row tile is introduced. Both branches
pass independent numerical tests. The extra program and partial final tile
are source-level differences; timing alone does not isolate their hardware
cost or exclude compiler/runtime effects.

Validate adjacent counts on both sides of a dispatch condition, partial
tiles and changes in source count before recommending a capacity rule.
Do not move a cutoff solely to pass the public cases or infer that every
unaligned row count regresses. The failed predicate and full measurements
remain in the task; no universal wave count or divisibility policy is added.
Sources: `attnres-followup-capacity-formal`,
`attnres-followup-capacity-guards` and `attnres-followup-capacity-source`
in [sources.json](sources.json).
