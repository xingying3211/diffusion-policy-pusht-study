# Provenance and curation changes

The source experiment directory was treated as read-only. Its name is
`lerobot-first-run`; all source paths in the manifest are relative to that
directory. No virtual environments, caches, downloaded wheels, model weights,
or upstream source trees were copied. No new rollout or training was run.

## Project-local code and authorship

`eval_cuda.py`, `eval_cuda_deterministic.py`, the shell launchers, and the two
original analysis scripts were present at the source project's top level,
outside the installed LeRobot package. They are classified as project-local
evaluation/instrumentation code, not as a new policy implementation. Without a
source Git history, individual authorship or use of coding assistance cannot be
independently established. The README describes the experiment work without
claiming sole authorship of every line.

The two evaluation wrappers in `scripts/` come from those project-local files.
For future runs, the reward logger now also saves `reward_trace_episode_000.json`
and subsequent episode files. The legacy `reward_trace.json` remains the last
episode only. Historical files have not been retroactively repaired. Historical
action logs use global step counters and lack episode boundaries; the included
trace analysis therefore requires single-episode runs.

The launchers come from `run_eval_action_steps.sh` and
`run_eval_deterministic.sh`. Curation updates their root/script paths, permits
`PYTHON` and `MODEL_DIR` overrides, writes new outputs under ignored `runs/`,
and saves invocation metadata. They still use the upstream evaluator and refuse
to overwrite existing run directories. The deterministic launcher sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; the inherited deterministic wrapper disables
cuDNN benchmarking, enables cuDNN deterministic mode, and calls
`torch.use_deterministic_algorithms(True)` after LeRobot's seed setup.

`plot_trace.py` replaces the original plotting scripts. The original
`analyze_trace.py` selected index 1 (Y) but labeled it X; both original analysis
scripts included steps 200–300 (101 steps) despite a "last 100" title.
The new plot uses the last 100 records, correct axis labels, explicit input/output
arguments, and refuses output overwrite. Archived SVGs were omitted because
their exact generation history was not preserved. The new shared README figure
is derived from the unchanged failed-run trace (steps 201–300).

`summarize_results.py`, `compare_traces.py`, `verify_model.py`,
`check_repository.py`, both READMEs, and these curated notes are new packaging
and verification work. They are not presented as historical experiment code.

## Records and selection

All 13 existing result directories with `eval_info.json` are retained as compact
JSON/GPU/timestamp evidence, so the main comparison does not hide other runs.
Recorded logs are copied after removing terminal progress-control sequences and
replacing the original absolute project path with `<PROJECT_ROOT>` (and home
prefixes with `<HOME>`). JSON video paths become repository-relative paths.
Metrics, seeds, traces, timing values, and success flags are unchanged.
All recorded small rollout videos are shared by both README languages.
The existing October 13 preview image is copied unchanged. Large/generated
verification summaries duplicating metrics and private paths are omitted.

The original lockfile is copied unchanged as `requirements-lock.txt`: it is a
historical environment snapshot, not a newly solved or hash-locked environment.
`results/environment.json` is the historical preflight record with its private
executable path replaced by `<PROJECT_ROOT>/.venv/bin/python`.
`results/model_reference.json` retains only model identity, revision, size and
weight hash from the original metadata. The original model config is summarized
in the README/source notes instead of redistributed.

`results/summary.csv` and `results/reproducibility.json` are newly computed from
the archived records. `results/curation_manifest.json` records each copied or
adapted file's source path, source SHA256, destination SHA256, and transformation.
New documentation/analysis files are identified by this note rather than given
invented historical provenance.

## Exclusions and unresolved evidence

Original README/notes were used as evidence but rewritten: they contain private
paths, machine/proxy details, and a planned `n_action_steps=1` comparison with no
matching result directory. That planned comparison is not reported as performed.
`collect_results.py` and `preflight.py` are not packaged as runnable scripts:
they hard-code the old path and would overwrite historical report files.
The source `source/` and `source-inspection/` contain third-party code; snapshots
matched the installed package, and they are excluded from this independent repo.

Per-run copies of the exact wrapper and dependency freeze were not saved.
Current scripts may include later instrumentation, and the historical lockfile
predates some runs. Logs verify the displayed configs, CUDA assertions and
deterministic-mode markers, but do not prove every run used identical binaries.
Claims of strict cross-machine or end-to-end bitwise reproducibility are therefore
not supported.

Before the first publication commit, trailing whitespace was removed from copied
logs and wrappers. The derived summary CSV was normalized to LF. These changes
do not alter experiment metrics or execution logic; affected manifest hashes
were updated.
