# Backend optimization knowledge

Backend profiles describe how KernelPilot matches a device and discovers
analysis tools. Vendor-specific optimization knowledge lives alongside those
profiles under:

```text
backends/<vendor>/note/INDEX.md
```

Before optimizing, confirm the backend and device with the task contract and
live environment preflight, then read the matching note index. Check each note's
validated device, architecture, language, operator, workload, and limitations.
Treat notes as starting hypotheses; task-local correctness, profiling, and
benchmarks remain authoritative.

Current vendor note collections:

- [Hygon](hygon/note/INDEX.md)

When a new domestic accelerator backend is added, create its vendor note
directory and index. If there are no matching notes yet, state that and use
task-local evidence rather than transferring another vendor's claims.
