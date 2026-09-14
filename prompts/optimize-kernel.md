# Kernel Optimization Task

Read `.kernelpilot/task.yaml`.

If `target.profile` is set, read `backends/<target.profile>.yaml` and load only
the skills routed for the current stage. If `execution.transport` is `ssh`, use
the KernelPilot remote pull/push commands; never open an unrestricted remote
shell from the optimization session.

Follow the repository `AGENTS.md` and the KDA optimization workflow
(`workflows/kernel-optimization.yaml`).

Requirements:

- Establish correctness first.
- Measure the existing implementation before editing.
- Use profiling evidence when the next optimization is not obvious.
- Make one coherent optimization hypothesis per candidate.
- Run correctness after each implementation change.
- Benchmark only correct candidates.
- Record all candidate results, including rejected and regressed ones.
- Continue until the configured target or stop condition is reached.
- Preserve the best correct candidate locally so a failed remote candidate can
  be rolled back through the same controlled adapter.

At completion, report:

- best candidate
- baseline latency
- optimized latency
- speedup
- correctness status
- main optimizations
- remaining bottlenecks
