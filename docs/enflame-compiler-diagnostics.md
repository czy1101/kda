# S60 compiler-artifact diagnostics

Scope: how to establish what the Enflame S60 (GCU300) toolchain did to one
kernel, and where to stop. This is a diagnostic procedure for the `compiler`
direction in [bottleneck-localization.md](bottleneck-localization.md). It does
not authorize changing the compiler.

Everything below is scoped to one reviewed installation (FlagTree Triton 3.6.0
with the `enflame` backend, `torch_gcu`, TOPS SDK LLVM 11). Stage names, pass
counts and tool paths are observations from that environment, not a
specification. Re-check the installed chain before relying on any of them.

## 1. Confirm the real chain before reading any artifact

Do not assume NVIDIA's stage semantics. On this backend `ttir` and `ttgir` are
frontend, `gcuir` is still in-process, and everything after `llir` belongs to an
external vendor toolchain.

Establish, from the installed package rather than from documentation:

1. Which backend is registered and which class implements it. On this install
   `import triton_gcu` fails and `pip show` finds nothing; the entry point is a
   registered backend named `enflame`.
2. The `add_stages` sequence, its stage names, and which producer function
   creates each artifact.
3. For each stage, whether it runs **in this process** or as an **external
   process**. The boundary decides what a Python traceback can tell you.
4. Where the device library is linked, where the target object is generated, and
   where the fatbin is packaged. These are three different tools.

What the reviewed chain looked like:

| Stage | Producer | Runs | Note |
|---|---|---|---|
| `ttir` | frontend | in-process | 8 passes |
| `ttgir` | frontend | in-process | 4 passes; output can be byte-identical to `ttir` |
| `gcuir` | backend | in-process | 21 passes registered, 19 executed; two skipped conditionally |
| `llir` | external `gcu-compiler-opt` | **external process** | 28 passes; `toolkit` prints child stderr into the parent |
| `fatbin` | external `gcu-compiler-compile --device-only` | **external process** | machine code from `llc`, link from `clang`/`ld.lld`, packaging from `clang-offload-bundler` |

Pass counts are not a pipeline specification: each number above is the count
observed while compiling **one** kernel sample on the reviewed installation.
Another kernel, another shape class or another build can register a different
set, so re-count before concluding that a pass "must have run".

Three consequences that change how artifacts are read:

- Child-process timing printed into the Python stdout does not mean the pass ran
  in-process. Attribution needs the process tree, not output interleaving.
- `gcu-compiler-opt` and `gcu-compiler-kernel` are siblings under
  `gcu-compiler-compile`, not nested. Asserting nesting from command order is
  how a previous report got the code generator wrong.
- Between `llir` and machine code the vendor bridges LLVM versions through
  SPIR-V and then rewrites the triple, data layout and kernel attributes with
  text edits. A surprising identifier in an intermediate file may come from that
  bridge rather than from the kernel.

**Do not infer IR type from the file suffix.** `.ttir`, `.mlir` and `.ll` are
filename conventions; the same toolchain has emitted a `.ttgir` that was
byte-identical to its `.ttir`. Check the bytes: an offload bundle starts with
`__CLANG_OFFLOAD_BUNDLE__`, an ELF starts with `\x7fELF`, MLIR text contains
`module {`. `scripts/compile_artifacts.py` reports the suffix as a `stage_hint`
and the container separately, and flags a conflict between them.

## 2. Save the identity of the specialization that actually ran

A cache directory proves that a compilation happened. It does not prove that the
timed call used that compilation. Record all of:

| Item | Why it is needed |
|---|---|
| Kernel name and source identity | Two kernels can share a name across revisions |
| Shape, dtype, stride, input identity | Specialization, not just the kernel, decides the artifact |
| `constexpr`, grid, warps, stages | Same source, different artifact |
| Selected autotune configuration | The deployed config is often the real change |
| Compiler, backend and SDK identity, with file hashes | A stage name means different things across versions |
| Available stage artifacts, their metadata and hashes | The only comparable object |
| Evidence that the timed region used this specialization | Closes the chain |

Seen on this backend, both directions of the trap:

- One kernel name produced **two** cache entries whose `ttir` signatures differ
  only by a `tt.divisibility` attribute. They correspond to different runs
  (different batch-head counts) and their artifacts are not interchangeable.
- Four autotune configurations of one kernel produced a fatbin where
  configuration 4 was **byte-identical** to configuration 1, under a different
  cache key.

So: a cache key is not device code. A changed key with identical artifacts means
the source path, a line number or a hash input changed, not the generated code.
A present cache entry is not a launch record.

## 3. Compare artifacts without over-claiming

Order of operations:

1. **Index, then compare.** Hash every artifact on both sides first. A comparison
   that starts from a diff cannot tell you whether a stage is missing or merely
   unchanged.
2. **Keep the raw hashes.** Never edit an artifact in place to make a comparison
   easier.
3. **Separate the classes of difference.** A line-number or `loc(...)` shift, an
   SSA renumbering, and a different computation are three different findings.
   Only the third supports a mechanism claim.
4. **If you normalize, save the normalized output separately** and keep the
   original hash in the record. Never let a normalized file replace the artifact.
5. **Do not declare semantic equivalence from a text transform.** Stripping
   `loc(...)` and observing that lines match is a byte-level statistic. It is not
   a proof that two IR modules compute the same thing, and it is not a proof that
   the machine instructions match.
6. **Identical bytes do not imply an unchanged timed region; differing bytes do
   not imply differing machine instructions.** For the second case, continue to
   the device image or the disassembly before saying "instruction".
7. **Check what was paired.** One kernel name can hold several specializations
   in one capture. The tool pairs exact relative paths first and falls back to a
   unique filename, recording `match_quality: name-only` when it does; a name
   that stays repeated on either side is not paired at all and is listed under
   `summary.ambiguous_names`. A name-only pair may be two different
   specializations, so confirm shape, dtype and launch options on both sides
   before reading a byte result as a comparison of the same artifact.

Offline, with no device and no framework import:

```bash
# identity of one capture, including missing stages and unsupported files
python3 scripts/compile_artifacts.py index \
  --root <artifact-dir> --expect ttir,ttgir,gcuir,llir,fatbin \
  --out runs/compile-artifacts.json

# layered comparison of two captures; --locations adds a line-level statistic
python3 scripts/compile_artifacts.py compare \
  --left <baseline-dir> --right <variant-dir> \
  --expect ttir,ttgir,gcuir,llir,fatbin --locations \
  --out runs/compile-artifact-diff.json
```

The tool reports identity, sizes, stage hints, container kinds, cache keys,
bundle targets, first differing offset and differing-byte counts. It reports no
operation counts and makes no semantic claim. Read its `unresolved` list before
quoting a result.

`index` and `compare` accept a directory or a previously written manifest.
Fixtures used in this repository's tests are test data, not S60 evidence.

## 4. Localize a pass before blaming it

1. **Locate the stage first.** Find the earliest stage where the predicted
   difference appears; earlier stages are then exonerated for that difference.
2. **Compare across the pass, not across the pipeline.** A pass that ran between
   two stages is only implicated if the difference appears between those stages.
3. **Registered, executed and effective are three different facts.**
   - Registered: the pass is in the pipeline.
   - Executed: it ran in this compilation.
   - Effective: its output differs visibly.
   On this backend all three diverge: 2 of 21 `gcuir` passes were skipped
   conditionally, and 4 `ttgir` passes ran with non-zero time while producing
   output byte-identical to their input.
4. **Know which dump is complete.** `MLIR_ENABLE_DUMP=1` printed only the passes
   attached directly to the top-level pass manager — 6 of 19 here. Passes inside
   a nested pipeline printed nothing. Timing dumps were complete. An absent
   "After" line is a dump limitation, not a pass that did not run.
5. **When the dump is partial, report the interval you can bound.** "Cannot
   locate between `gcuir` and `llir`" is a result. "Must be the layout pass"
   is not, unless the interval narrows to it.
6. **Public source of a pass is not the installed pass.** A matching file in a
   public repository does not establish that the installed binary was built from
   it, or with the same flags. Keep the installed file's hash in the record.

Resource fields deserve the same care. On this backend the same word carries
four different numbers:

| Value | Source | Meaning |
|---|---|---|
| `metadata.shared` | computed at compile time, stored in cache metadata | consumed as the launch's dynamic shared-memory argument (`sharedMemBytes`), passed through the host launcher |
| sum of shared allocation sizes in an IR stage | text in that stage | an allocation ledger, not a peak; buffers may not be live together |
| `metadata.max_shared` | compile-time formula (`8 MiB × num_warps`) | an upper bound used to raise `OutOfResources` |
| `max_shared_mem` | a hard-coded per-arch dict in the backend driver | not a device query |
| shared region in the device ELF | vendor struct | physical reservation; mapping from `metadata.shared` is unresolved here |

These are four different quantities and only one of them is measured:

- the **allocation ledger** is text in an IR stage: what was allocated, not what
  was live;
- the **metadata field** is a compile-time number handed to the launch;
- the **live set** — which buffers overlap in time — was not measured at all;
- the **physical reservation** is the device image's own record, and its
  relationship to the metadata number is unresolved on this backend.

A previous revision of this page said `metadata.shared` is "what gets passed to
the module loader". The reviewed path is the launch argument, not the module
loader. This revision did not re-open the installed backend — the S60 host was
not reachable from the machine that edited it — so the claim rests on the
earlier installed-source review and still needs its exact file and line pinned
on the host. Treat "which argument carries it" as established and "where
exactly, in this version" as outstanding.

For the local-state kernel reviewed, `metadata.shared` was 65536 while the
device ELF's block shared-segment field was 0. Treat the relationship as
unestablished rather than assuming the smaller number wins.

Similarly, `warp_size` appears as a backend constant (12), as a compile option
(1), as `ttg.threads-per-warp` (1), and as the ELF's `wavefront_size` (32). They
are not interchangeable and none of them alone is "the hardware width".

## 5. Decide where to change, from what the artifacts showed

| Artifact finding | Where to change | What to keep fixed |
|---|---|---|
| The kernel expresses redundant work (a mask that repeats the producer's zeros, a narrow strided read, a duplicated staging copy) | Kernel source | Config, so the measurement isolates the source change |
| The expression is reasonable but the layout conversion is materialized as extra transfers | Kernel source, unless the rewrite is impossible or the rule would serve several kernels worth the maintenance | Config; then compare artifacts to confirm the conversion disappeared |
| The deployed configuration is what moved the number | Configuration, as a controlled experiment | Source |
| The generated device code is identical while the full call differs | Runtime boundary: submission, allocation, synchronization, module load, host launcher, or the timed region | Kernel source and config |
| The first call is slow and the warm call is normal | Compilation, autotune trials, cache population, module load, host-launcher build — not the kernel | Warm-up policy, and say whether the target metric includes these |

Record which of the four change classes the candidate belongs to
(`kernel`, `config`, `compiler`, `runtime`). A source rewrite whose gain partly
comes from the autotuner switching configuration is a source change plus a config
change: report the delta at fixed configuration, the delta at fixed source, and
the deployed combination, then say whether they add up. The fixed-config delta is
the semantic one, but it is not a licence to subtract: the rewrite can make a
configuration viable that was not viable before, so the two deltas need not sum
to the deployed result.

Two examples from the S60 GLA work, both of which were initially mis-attributed:

- A key-major FP32 gate tile consumed by a broadcast elementwise op materialized
  a shared transpose pair. The finding was artifact-level; the gain was
  aggregate over fewer transposes, fewer waits, fewer barriers and a smaller
  footprint, and those were never ablated.
- A "no `triton_gcu.matmul` in the IR" observation was once used to conclude
  that a kernel had no matrix multiply. The kernel had one `tl.dot`, one
  `gcu.matmul`, one convgen call site and 32 matrix instructions; the two
  `triton_gcu.matmul` strings were region border markers. Absence of a token is
  not absence of an operation.

## 6. Gate for actually modifying the compiler

This repository does not authorize compiler changes. Source and configuration
are the cheap places to change; a rule change is a different kind of decision,
and "the source cannot express it" is only one of the reasons to take it. The
others are how many kernels the rule would serve, who maintains it, how narrow
its applicability condition is, and what it might break in kernels that already
compile. A rule that helps one kernel and silently changes another is a
regression with a speedup attached.

If a future task does authorize one, it needs its own contract and must satisfy
all of the following before any claim:

1. A separate task in a separate workspace, with an explicit allowed scope for
   the compiler build inputs. Never edit the installed toolchain in place.
2. A minimal reproduction: one kernel, one shape, one configuration, one
   artifact pair.
3. The rewrite's applicability condition and its legality argument, in terms a
   reviewer can falsify.
4. An independent build that leaves the installed environment untouched and
   records both versions' file hashes.
5. Positive and negative IR tests: a case the rule must transform, and a case it
   must leave alone.
6. Numerical, boundary and fallback validation, separated from performance
   measurement.
7. A compiler A/B at fixed kernel and fixed configuration, so the rule is the
   only changed variable.
8. An impact statement for other shapes, other operators and other backends,
   including which ones were not tested.
9. The reuse case: which other kernels or operators this rule is expected to
   serve, and what happens to them if it is wrong.
10. A maintenance statement: who owns the rule, which upstream version it is
    pinned to, and what happens on the next compiler upgrade.

Absent any of these, the correct outcome is a documented proposal, not a
modified toolchain.

## Tool availability limits observed here

These are recorded so a later run does not rediscover them as failures:

| Attempt | Outcome |
|---|---|
| `MLIR_ENABLE_DUMP` | prints only top-level manager passes (6 of 19 here) |
| `gcuobjdump --list-elf` | unsupported by the installed version |
| `llvm-gcu-resusage` | consumes LLVM IR text, not an ELF |
| `llvm-nm` | not shipped with this SDK |
| `llvm-objdump -d` without `--mcpu=gcu300` | every instruction `<unknown>` |
| `tops_convgen_*` definition | not in the builtins bitcode; source and inline point unlocated |

Unavailable tooling is a capability limit. It is not evidence about the kernel.
