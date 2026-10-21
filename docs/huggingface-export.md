# Export a Hugging Face kernel

Export a selected CuTe-DSL candidate after it passes the task's validation command.
CuTe-DSL is NVIDIA's Python language for CUDA kernels. This optional step runs
from the KDA checkout and packages Python sources and JSON data from the task
workspace. The task continues to own its validation command and promotion criteria.

Use Python 3.11 or newer. Install Hugging Face's `kernel-builder` and Python
`setuptools` in the export environment. The builder must support `create-pyproject`,
the generated `setup.py build_kernel` command, and `hash`. Install the `kernels`
Python package in the environment used to validate and load the exported kernel.

Create a release configuration, with file paths relative to the configuration file:

```toml
repo_id = "your-org/your-kernel"
version = 1
license = "mit"
license_file = "LICENSE.txt"
cuda_capabilities = ["10.0"]
python_depends = ["nvidia-cutlass-dsl"]
readme_file = "CARD.md"
```

Supply the actual source license and supported architectures. `python_depends`
defaults to `["nvidia-cutlass-dsl"]`. `readme_file` is optional; without it, the
exporter generates a short kernel card. Use a custom card to document the public
interface, supported inputs, tested dependency versions, and measurements. Python
dependencies must be supported by the chosen Hugging Face builder and loader.

From the KDA checkout, export a source directory from a separate task workspace:

```sh
python scripts/export_huggingface.py \
  --solution-dir ../task-workspace/solution \
  --entrypoint kernel.py \
  --config ../task-workspace/release.toml \
  --output ../task-workspace/outputs/huggingface
```

`--entrypoint` is relative to `--solution-dir`; nested paths such as
`package/kernel.py` are supported. Keep only Python and JSON source files in that
directory. Python bytecode caches are ignored; symlinks and other source types
are rejected. An optional `--kernel-builder` argument selects the builder executable.
The output directory must be outside the source directory.

The exporter takes a source snapshot and adapts imports only in the export copy.
The original candidate is unchanged. Existing output directories are never
overwritten; an unsuccessful build leaves no partial output directory. Each
package's module identifier includes a digest of the snapshot and release settings.

The output contains:

- `source/`: the generated Python-only builder project (`torch-noarch`);
- `hub/`: `README.md`, `build/CARD.md`, and `build/torch-cuda/`, ready for validation.

The builder writes architecture, dependency, and file-digest metadata to
`hub/build/torch-cuda/metadata.json`.

The package exposes `run` and, when present, `prepare`. Ordinary imports of
candidate helper modules become relative imports in the export copy. Existing
relative imports and third-party imports are preserved. Dynamic imports and
unaliased dotted local imports need manual packaging and produce an explicit error.
Native CUDA/C++ sources require a different build and are not supported here.

Validate the generated package in the intended GPU environment before publishing:

```python
from pathlib import Path
from kernels import get_local_kernel

kernel = get_local_kernel(
    Path("../task-workspace/outputs/huggingface/hub"), trust_remote_code=True
)
```

Run the task's correctness cases through `kernel.run`, including any required
stream, repeated-call, and graph-capture behavior. A `torch-cuda` package still
compiles its CuTe kernels on first use; it does not promise compatibility with
unlisted GPU architectures. Measurements of `prepare`'s returned launch function
are not measurements of the entire `run` call.

Publishing is separate from export. The official uploader publishes the card
to `main` and the build to the version branch. For example, after validation:

```sh
kernel-builder upload ../task-workspace/outputs/huggingface/hub \
  --repo-id your-org/your-kernel --repo-type kernel --branch v1
```

Consumers can then load the published version:

```python
from kernels import get_kernel

kernel = get_kernel("your-org/your-kernel", version=1, trust_remote_code=True)
```

The opt-in is required for publishers that the loader does not trust by default.
