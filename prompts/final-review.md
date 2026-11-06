# Final Review Task

Read `.kernelpilot/task.yaml`, `runs/candidates.jsonl`, `runs/benchmark.csv`,
and `profile/`.

Do not modify the implementation. Produce the final report:

- metric, direction, baseline value, best candidate value, unit, and measured improvement
- correctness status of the best candidate, and the command that proved it
- the main optimizations, each tied to the evidence that justified it
- for an experiment-audit deliverable, the candidates that were rejected or revised, and why
- remaining bottlenecks, and the profiling evidence behind them
- whether the target was reached; if it was not, the observed shortfall and the
  stop condition that ended the run
- final checkpoint restoration and TLE assessment, including missing primitive
  requirements when TLE was enabled
- for an execution-audit deliverable, workflow errors and recoveries

Separate measured facts from hypotheses and unknowns. Do not claim an
improvement unsupported by the recorded full benchmark evidence. Keep failed
and regressed candidates in the internal ledger; include them in an audit
deliverable. Always show per-shape regressions of the reported methods.

## Report scope

Read `skills/operator-optimization-report/SKILL.md` before choosing the report
structure. Follow the requested delivery format and language. A successful-method
summary expands retained mechanisms with complete comparisons and inline
evidence; it does not copy the internal ledger of rejected candidates or workflow
errors into the deliverable. Preserve those records internally. An execution-audit
report includes them. Do not hide per-shape regressions of retained mechanisms.
If cumulative benchmark evidence is missing, identify that gap; this review task
does not itself authorize implementation edits or new GPU measurements.
