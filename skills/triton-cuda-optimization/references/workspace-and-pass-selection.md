# Workspace and pass selection

## Decide from total cost

A partitioned kernel can improve parallelism but adds intermediate writes,
reads, reduction work, launches, and memory pressure. Compare the whole path,
not only the first kernel.

Before raising a workspace threshold, calculate the peak bytes of all live
temporary tensors. Check the largest required shape on the actual device. An
allocation that succeeds at a medium shape can still fail asynchronously at a
larger shape; synchronize during validation so the failure is attributed to
the correct candidate.

## Shape-dependent routing

When one implementation wins only above or below a shape boundary:

- keep both paths only if the dispatch predicate uses stable input metadata;
- choose the boundary from repeated measurements around the crossover;
- validate boundary-adjacent shapes for correctness and performance;
- include dispatch and allocation overhead in timing;
- reject a larger threshold if any required shape exceeds the memory budget.

## Evidence expected

Record the formula used for temporary bytes, measured crossover shapes,
required-shape ratios, failure mode, and final threshold. A threshold without
this evidence is task folklore and should not be promoted into a reusable
backend skill.
