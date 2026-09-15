# Kernel Profiling Task

Read `.kernelpilot/task.yaml` and the current best candidate's measurements.

Run KernelPilot tool discovery first and select only a backend-declared tool
that is available in the contracted task environment. Then run the configured
profiling command for the shape that matters. Profiling is
expensive: only run it when the next optimization is not already justified by
the evidence at hand, and stay within the contract's `stop.max_profile_runs`.
For SSH tasks, use only KernelPilot's contracted profile command; do not invoke
an unrestricted remote shell.

Write `profile/<candidate-id>.md` with:

- the command that produced the data and the shape it profiled
- what the kernel is bound by
- the top stalls and the utilization figures you can actually read from the tool
- the specific source locations the numbers point at

Then state one concrete, falsifiable hypothesis: which change should move which
metric, and what would falsify it. Do not propose a change the data does not
support.
