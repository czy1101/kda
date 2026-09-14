# Kernel Analysis Task

Read `.kernelpilot/task.yaml`.

Inspect the implementation and the reference named in the contract. Do not edit code.

Produce `docs/analysis.md` covering:

- the operator's mathematical definition and exact semantics
- data layout and indexing conventions
- the hot path and the work each program instance performs
- the shapes that matter, and which ones the contract requires
- how the baseline is produced and whether it is comparable
- the unknowns that block a confident optimization decision

Keep every claim tied to the code you actually read. Where the code is
ambiguous, say so instead of guessing. Finish with the ranked list of unknowns
that need profiling evidence.
