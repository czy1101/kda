# S60-COMPILER-001: what identifies a compiled artifact, and what each field proves

## Generalization gate

- Reusable mechanism, interface or bounded backend limitation: on S60/GCU300 the
  device artifact is produced by a five-stage chain whose stages are owned by
  **three different compilers**: `ttir`/`ttgir`/`gcuir` run in the Triton process,
  `llir` is an external vendor optimizer process, and `fatbin` is a second
  external driver that generates the object, links it and packages the bundle.
  Identity of a compiled artifact is therefore a tuple of (kernel name, source
  identity, `constexpr`/grid/warps/stages, selected configuration, compiler and
  toolchain file hashes, per-stage artifact hashes). The Triton cache key is
  **not** part of that tuple: it changes when a source path, a line number or a
  specialization attribute changes, and it can stay equal while the artifact
  changes. Conversely, registered pass, executed pass and a pass that produced a
  visible change are three separate facts.
- Another operator/workload class to which it applies: any operator whose
  candidate explanation is "the compiler did something unexpected". The identity
  tuple and the three-fact separation are language-level, not GLA-specific; they
  apply to any Triton kernel on this backend and to CUDA targets once the stage
  names are replaced.
- Independent probe, mathematical argument or cross-workload evidence: one
  reviewed installation, with process parentage taken from `SIGCHLD` rather than
  from log order; six kernels censused for symbol declaration versus call site;
  two cache entries for one kernel distinguished by a specialization attribute;
  four autotune configurations of one kernel where the fourth fatbin was
  byte-identical to the first under a different cache key; one `ttgir` that was
  byte-identical to its `ttir` while four passes recorded non-zero time.
- Conditions that invalidate the claim: a different compiler version or backend
  that moves a stage across the process boundary; a build whose pass pipeline
  differs (the two conditionally skipped `gcuir` passes depend on target
  capabilities); a toolchain that stops bridging LLVM versions through SPIR-V;
  any environment where the cache key is derived from the artifact rather than
  from the source and options.

## Scope and status

- Observation date and review date: reviewed 2026-11-24, card written
  2026-11-26.
- Device/backend/architecture: Enflame S60 / GCU300, `enflame` backend, one
  visible card (physical card 7).
- Framework, compiler, SDK/driver and device-library versions: FlagTree Triton
  3.6.0, `torch` 2.10.0+cpu, TOPS SDK LLVM 11.0.0git, toolkit datadir
  `/opt/triton_gcu`. The package `triton_gcu` is not importable and not listed by
  `pip`; the backend is reached through the registered `enflame` entry point.
  Driver version not established by this card.
- Language, dtype/accumulation, M/N/K, strides/layout, launch options: Triton;
  chain facts are dtype-independent. The illustrative kernel was BF16/FP32 with
  `BT=64`, `BK=BV=128`, `num_warps=1`, `num_stages=1`.
- Shape class, inference/training mode and numerical semantics: dense chunked GLA
  forward used only as the artifact carrier; the card makes no performance claim
  about it.
- Status: **compiler-observed**. Not performance-observed; nothing here is a
  speedup claim.

## Observed facts

Stage ownership, as established from the installed backend entry point and the
external process tree:

| Stage | Owner | Boundary | Passes |
|---|---|---|---|
| `ttir` | Triton frontend | in-process | 8 |
| `ttgir` | Triton frontend | in-process | 4; output may be byte-identical to `ttir` |
| `gcuir` | Enflame backend | in-process | 21 registered, 19 executed |
| `llir` | vendor optimizer | **external process** | 28 |
| `fatbin` | vendor driver | **external process** | object from `llc`, link from `clang`/`ld.lld`, packaging from `clang-offload-bundler` |

Identity facts that changed a conclusion:

- Two cache entries existed for one kernel name. They differ by one
  `tt.divisibility` attribute in the `ttir` signature and correspond to different
  runs. Taking "the" cache entry for a kernel name is a coin flip.
- Four autotune configurations of one kernel produced a fatbin where
  configuration 4 was byte-identical to configuration 1 (same size, same hash)
  under a different cache key. A cache-key change is not device code.
- `ttir` and `ttgir` were byte-identical for two kernels while the four `ttgir`
  passes each recorded non-zero time. An executed pass is not a changed output.
- `MLIR_ENABLE_DUMP=1` printed 6 of 19 `gcuir` passes; passes inside a nested
  pipeline printed nothing. Timing dumps were complete. A missing "After" line is
  a dump limitation.
- A symbol census that reported `tops_exp2_v4_fp32` "17 times" was counting one
  declaration plus sixteen call sites. Declaration, static call site, in-loop
  call site, in-branch call site, dynamic execution count and final instruction
  count are six quantities; the last two were not measured.
- Absence of `triton_gcu.matmul` in the `gcuir` of a kernel that contained
  `tl.dot` was once read as "no matrix multiply". The same file has one
  `gcu.matmul`, the `llir` has one convgen call site, and the device image has 32
  matrix instructions plus one MAC; the two `triton_gcu.matmul` strings were
  region border markers.

## Mechanism and candidate hypothesis

The mechanism is a three-compiler pipeline with two process boundaries and one
LLVM version bridge. Because each boundary hides its interior, an artifact-level
observation can localize a difference to an interval but not to a pass, unless a
before/after pair exists for that pass in the same compilation.

Candidate hypothesis for a future run: when the source is unchanged and the
artifact differs, the difference will be attributable to a specialization
attribute or to a configuration change rather than to a source-level construct.
Falsified by two captures with identical specialization, configuration and
source revision that still differ at the same stage.

Unverified: the definition source of `tops_convgen_*`; whether
`gcu-compiler-kernel` writes the same object that `clang` produced; the mapping
from compiler-reported `shared` to the shared region in the device ELF.

## Reproduction and validation

Use `scripts/compile_artifacts.py` for identity, offline and without importing
the framework:

```bash
python3 scripts/compile_artifacts.py index --root <capture-dir> \
  --expect ttir,ttgir,gcuir,llir,fatbin --out runs/compile-artifacts.json
python3 scripts/compile_artifacts.py compare --left <a> --right <b> \
  --expect ttir,ttgir,gcuir,llir,fatbin --locations \
  --out runs/compile-artifact-diff.json
```

The tool reports identity only. Reproduction of the chain facts needs the device
and is out of scope for the tool. The procedure is
[../../docs/enflame-compiler-diagnostics.md](../../docs/enflame-compiler-diagnostics.md).

## Limits, failures and revalidation

- One installation, one compiler version. Stage names, pass counts and tool
  layout must be re-established on any other version.
- Physical placement of shared memory is unresolved: the compiler-reported
  `shared` for the illustrative kernel was 65536 while the device ELF's block
  shared-segment field was 0. Do not treat either number as the other.
- Dynamic execution counts and a 1:1 symbol-to-instruction mapping were not
  obtained. Family-level correspondence is not a mapping.
- Several inspection tools were unavailable or limited (`gcuobjdump --list-elf`,
  `llvm-gcu-resusage` on ELF, `llvm-nm`, `llvm-objdump -d` without `--mcpu`).
  These are capability limits, not findings.
- The vendor driver's internals after `llir` are closed. Only the processes it
  starts, the arguments it passes and the files it writes back are observable.

## Sources

- `s60-compile-chain-v2` in [sources.json](sources.json): the reviewed chain
  document (stage ownership, process tree, artifact hashes, `builtins` census,
  unknown items).
- `s60-compile-chain-errata` in [sources.json](sources.json): the corrected
  claims that produced the rules above.
- Related cards: [S60-EVIDENCE-001](evidence-and-measurement.md) for measurement
  boundaries, [S60-TRITON-014](triton-shared-transpose-materialization.md) for the
  shared-allocation accounting that this card keeps separate from the metadata
  field.
