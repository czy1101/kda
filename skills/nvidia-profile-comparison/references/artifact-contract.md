# Profiler artifact contract

Keep artifacts separated by role:

```text
profile/
  candidate/
    manifest.json
    report.ncu-rep
    metrics.csv
  reference/
    manifest.json
    report.ncu-rep
    metrics.csv
  comparison.md
  analysis.md
  low-level-analysis.md
```

Each manifest should identify the task, candidate or reference revision,
device, relevant tool versions, command, inputs, shapes, capture time, and
artifact hashes.

Direct-AI fallback is valid only when a structured export contains enough
metric names, values, units, kernel identity, and launch context to support the
analysis. Record why the routed skill was unavailable or inconclusive. Preserve
the binary report for later inspection, but never invent values that were not
exported.

Low-level artifacts must be attributable to the profiled implementation and
kernel. When attribution or compiler settings are unclear, mark the evidence as
inconclusive rather than comparing unrelated instruction streams.
