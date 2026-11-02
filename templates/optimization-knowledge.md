# {{title}}

> Status: confirmed only after correctness, benchmark, and promotion gates.

## Target

- Backend/device/architecture: `{{backend}}` / `{{device}}` / `{{arch}}`
- Operator and implementation: `{{operator}}` / `{{implementation}}`
- Dtype/layout/workload: {{applicability}}

## Result

{{measured per-shape candidate and baseline table}}

## Mechanism

{{mechanism backed by profile/compiler/benchmark evidence}}

## Transferable lesson

{{what future kernel work can reuse, and what must be re-measured}}

## Limitations

{{untested shapes, dispatch guards, and known regressions}}

## Evidence

{{task-local benchmark, profile, correctness, and generalization artifacts}}
