# Shared-memory and resource policy

This is the user's coding convention for KDA optimization work. It constrains
candidate implementations; it is not a claim that hardware shared-memory
reservation is inherently invalid.

## Prohibited patterns

- Do not explicitly allocate an empty shared-memory object from a shape or byte
  count and subsequently populate it with stores or copies. Initializing every
  location before reading does not make this source-level pattern acceptable.
  This applies to row caches, reduction scratch, and communication buffers.
- Do not disguise the same reservation with renamed helpers, dummy data, zeros,
  or sentinels whose purpose is only to obtain storage for later replacement.
- Do not introduce or retain an independent resource-budget autotuning scheme:
  a register-prefix element budget such as `REG`, an explicit shared-memory byte
  budget such as `CACHE_BYTES`, or a physical register-count cap used as a
  standalone optimization knob. Do not search these budgets to trade occupancy
  against manually reserved storage.

## Permitted data flow

- An interface may take actual algorithm data and construct or copy its shared
  representation as part of that operation. Its extent must follow the data
  and documented layout requirements. Use only an interface documented and
  available in the contracted backend/version.
- Compiler-managed shared staging and registers holding intermediate values
  remain permitted. A real algorithmic zero-initialized accumulator is distinct
  from dummy initialization used only to reserve a later cache.
- Algorithmic tiles, thread/warp counts, and pipeline stages may still be tuned
  for a measured computation or data movement. Their indirect effects on
  physical register/shared-memory use do not make them forbidden resource
  quotas. Replacing `CACHE_BYTES` with a tile name while retaining the same
  empty-allocation scheme does not comply.

No concrete allocation/copy API is prescribed here. Check its real semantics;
do not invent a data-initialization interface. If the backend cannot express
the permitted flow, retain an allowed ordinary/streaming implementation or
report the missing capability under the task's extension policy.

## Interpreting configuration code

Review the complete kernel and its callers rather than matching parameter
names alone:

| Construct | Interpretation |
|---|---|
| `_prune_cache_configs` deriving packing metadata | Does not itself allocate memory; inspect the consumer. |
| `WORD_BLOCK` derived from tile size and dtype | May describe real packed data; it cannot legitimize an empty allocation. |
| `REG` selecting a live logit prefix | Counts elements, not physical registers; a separately tuned retention budget is prohibited here. |
| `CACHE_BYTES` selecting an empty shared buffer | The resource-budget and allocate-then-fill scheme is prohibited. |
| Tile/warp/stage configuration | Permitted when tied to actual work and evaluated through the normal task gates. |

## Candidate review

Before implementing a candidate, identify shared objects, their actual input
data, initialization interfaces, consumers, lifetimes, and the meaning of each
tunable parameter. Reject an empty-allocation or independent resource-budget
proposal before implementation. A documented API, passing correctness, or a
speedup does not override this convention.

For an existing noncompliant candidate, record the policy reason and restore
the best compliant correct checkpoint. If none exists, report that fact and
keep the original evidence intact; do not relabel an incompatible checkpoint
as compliant. Change operator source only within the task's authorized paths.
Keep frozen references, tests, benchmark definitions, and historical evidence
unchanged. This document supplies no permission to expand that scope.
