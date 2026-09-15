# Final Review Task

Read `.kernelpilot/task.yaml`, `runs/candidates.jsonl`, `runs/benchmark.csv`,
and `profile/`.

Do not modify the implementation. Produce the final report:

- metric, direction, baseline value, best candidate value, unit, and measured improvement
- correctness status of the best candidate, and the command that proved it
- the main optimizations, each tied to the evidence that justified it
- the candidates that were rejected or revised, and why
- remaining bottlenecks, and the profiling evidence behind them
- whether the target was reached; if it was not, the observed shortfall and the
  stop condition that ended the run
- workflow errors, recoveries, final checkpoint restoration, and TLE assessment
  or missing primitive requirements when TLE was enabled

Separate measured facts from hypotheses and unknowns. Do not claim an
improvement that `runs/benchmark.csv` does not show, and do not hide failed or
regressed candidates.
