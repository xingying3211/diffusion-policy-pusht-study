[English](README.md) | [简体中文](README.zh-CN.md)

# Diffusion Policy + PushT Study

## 1. Project Overview

A small reproduction and analysis study of a **pretrained Diffusion Policy in
PushT**, using Hugging Face LeRobot. I used this project to study the inference
pipeline, run simulation evaluations, compare action execution chunks, and
inspect GPU execution and repeatability.

**I did not propose Diffusion Policy or train the original pretrained model.**
This repository documents existing experiments and their limitations. It is
intended as a transparent undergraduate robot-learning portfolio project.

![Recorded PushT rollouts: start, middle, and end frames](results/pusht-first-experiment-20261013/rollout_preview.png)

The image comes from three recorded simulation episodes. Shared videos:
[baseline success](results/pusht-first-experiment-20261013/videos/eval_episode_0.mp4),
[4-step failure](results/dp-steps4-5ep/videos/eval_episode_0.mp4),
[8-step failure](results/dp-steps8-5ep/videos/eval_episode_3.mp4).
These are local experiment recordings, not upstream demo videos.

## 2. Upstream Model and Dependencies

- Framework: [LeRobot 0.3.3](https://github.com/huggingface/lerobot/tree/v0.3.3).
- Environment: [gym-pusht 0.1.5](https://github.com/huggingface/gym-pusht), Gymnasium 0.29.1.
- Checkpoint: [lerobot/diffusion_pusht](https://huggingface.co/lerobot/diffusion_pusht),
  revision `84a7c23178445c6bbf7e1a884ff497017910f653`.
- Recorded runtime: WSL Ubuntu, Python 3.12.13, PyTorch 2.7.1+cu128,
  torchvision 0.22.1+cu128, diffusers 0.35.1, torchcodec 0.5;
  NVIDIA GeForce RTX 5060 Laptop GPU.

The historical [dependency snapshot](requirements-lock.txt) and
[preflight record](results/environment.json) preserve the recorded environment.
LeRobot 0.3.3 was chosen for this older checkpoint's policy-contained
normalization structure; the wrappers require `strict=True` loading and reject
CPU fallback. This does not establish compatibility with newer LeRobot releases.
Model weights (~1.05 GB), environments, and upstream source trees are excluded.

## 3. My Work

- Set up and checked a GPU simulation evaluation workflow for the pretrained model.
- Ran existing small evaluations and compared `n_action_steps=4` and `8`.
- Used project-local wrappers to check strict checkpoint loading, CUDA placement,
  first-step observations/actions, action queue behavior, rewards, and memory use.
- Read the observation preprocessing, DDPM sampling, action generation/queue,
  evaluator, and environment step code; documented the data flow.
- Compared fixed-seed runs with and without deterministic PyTorch/cuDNN settings.

The curated repository adds portable launchers, corrected trace plotting,
offline result checks, bilingual documentation, and file provenance. Project-local
scripts were present in the original workspace; no Git history establishes sole
authorship of every line. See [provenance and changes](notes/provenance.md) and
[upstream attribution](THIRD_PARTY_NOTICES.md).

## 4. Setup and Reproduction

### 4.1 Inspect archived results without a model

From the repository root, Python 3.12+ is sufficient; these commands use only
the standard library and do not start simulation or download anything:

```bash
python3 scripts/summarize_results.py
python3 scripts/compare_traces.py results/dp-repro-a results/dp-repro-b
python3 scripts/compare_traces.py results/dp-det-a results/dp-det-b
python3 scripts/check_repository.py
```

To regenerate an axis-correct plot from an existing trace, choose a new output
path (the plotter refuses overwrite):

```bash
mkdir -p runs
python3 scripts/plot_trace.py \
  results/dp-trace-steps4-seed1000/action_step_trace.jsonl \
  runs/pusher_x.svg --axis x
```

### 4.2 Recreate an evaluation environment

Use WSL/Linux with an NVIDIA driver and CUDA-capable PyTorch. With `uv` available,
the following commands create a separate environment and install the historical
versions. They download several GB and were **not executed during curation**:

```bash
uv python install 3.12.13
uv venv --python 3.12.13 .venv
uv pip install --python .venv/bin/python \
  torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements-lock.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128
uv pip check --python .venv/bin/python
```

The snapshot is not a fresh resolver lock or a guarantee of future wheel
availability. Curation checked the existing environment but did not test a clean
installation. No PushT training dataset is required for evaluation.

### 4.3 Obtain the checkpoint and run evaluation

If the model already exists, set `MODEL_DIR` to that directory. Otherwise, this
optional command downloads only the pinned model configuration and weights:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="lerobot/diffusion_pusht",
    revision="84a7c23178445c6bbf7e1a884ff497017910f653",
    allow_patterns=["config.json", "model.safetensors"],
    local_dir="model",
)
PY
export MODEL_DIR="$PWD/model"
.venv/bin/python scripts/verify_model.py "$MODEL_DIR"
```

Expected weight SHA256:
`995d14d35db57d95c35ad9704c3d79c8612b7bc45f3877e5c46c2cdc516856a8`.
The verifier checks weight size/hash and config presence. The original local
config also explicitly set `device=cuda` and `use_amp=false`; the launchers pass
those settings explicitly.

These are manual reproduction commands, **not new results**. Run names must be
new. Outputs go to ignored `runs/`; archived `results/` are preserved:

```bash
bash scripts/run_eval_action_steps.sh repeat-steps8 5 8
bash scripts/run_eval_action_steps.sh repeat-steps4 5 4
bash scripts/run_eval_deterministic.sh repeat-det-a 1 4
bash scripts/run_eval_deterministic.sh repeat-det-b 1 4
```

All launchers use CUDA, AMP=false, batch size 1, base seed 1000, offline model
loading, and refuse existing output directories. `PYTHON` optionally overrides
the default `.venv/bin/python`. Historical runtime depends on hardware; changing
the action chunk changes diffusion RNG consumption as well as replanning frequency.

## 5. Experiments and Results

### 5.1 Recorded evaluations

Values below are calculated from the existing `eval_info.json` files, not from
new evaluation. Both comparison groups use episode seeds 1000–1004, horizon 16,
two observations, and 100 DDPM denoising steps. The baseline uses seeds 1000–1002.

| Recorded run | n_action_steps | Episodes | Successes | Success (%) | Avg. max reward | Eval time (s) |
| --- | --- | --- | --- | --- | --- | --- |
| pusht-first-experiment-20261013 | 8 | 3 | 3 | 100.0 | 1.000000 | 101.505 |
| dp-steps8-5ep | 8 | 5 | 3 | 60.0 | 0.996139 | 183.962 |
| dp-steps4-5ep | 4 | 5 | 1 | 20.0 | 0.962637 | 459.651 |

Sources: [baseline](results/pusht-first-experiment-20261013/eval_info.json),
[8-step group](results/dp-steps8-5ep/eval_info.json),
[4-step group](results/dp-steps4-5ep/eval_info.json).
All 13 recorded runs, including additional checks and failures, are listed in
[summary.csv](results/summary.csv); the table is not a pooled success estimate.

These are exploratory results with only five episodes per comparison group and
one recorded batch per setting. They do **not** demonstrate that eight steps are
generally better. The longer four-step runtime includes more frequent sampling
and different episode lengths; it is not a controlled latency benchmark.
Total reward is also affected by episode length and should not be treated as a
standalone measure of task completion.

### 5.2 Repeatability and GPU evidence

Two ordinary fixed-seed, four-step, single-episode runs (`dp-repro-a/b`) both
succeeded, but action/state traces differ: 153 versus 154 calls. With deterministic
settings (`dp-det-a/b`), both succeeded with 165 calls and byte-identical
action/state JSONL files. Rewards are **not exactly identical**: their maximum
absolute difference is `3.3306690738754696e-16`. This supports a limited same-machine
repeatability observation, not bitwise determinism of the entire pipeline.
See [computed comparison](results/reproducibility.json) and the archived logs.

The baseline [GPU record](results/pusht-first-experiment-20261013/gpu_evidence.json)
reports strict loading, 561 CUDA action calls, peak allocated memory 3327.67 MiB,
and peak reserved memory 3344.00 MiB. PyTorch memory and whole-GPU `nvidia-smi`
samples measure different things; neither is a universal resource requirement.

## 6. Key Observations

- Observation images become `[1,3,96,96]` float32 tensors, with pusher state
  `[1,2]`, before entering the CUDA policy.
- DDPM generates a horizon of 16 actions; the policy queues 4 or 8 actions and
  returns one `[1,2]` target position per environment step. Observations update
  every step, while diffusion sampling waits for the action queue to empty.
- A shorter execution chunk replans more frequently but did not improve success
  in this small recorded comparison. Fixed seeds alone did not yield identical
  ordinary CUDA rollouts.
- Trace plots show pusher position and action targets, not object pose or contact
  forces. They support inspection of execution behavior, not a causal diagnosis.

![Corrected X-axis trace from the failed four-step run, steps 201–300](results/figures/pusher_x_last100.svg)

This figure was regenerated from existing records during curation; it is not a
new rollout. Further implementation details are in the shared
[source-reading notes](notes/source-reading.md).

## 7. Limitations and Acknowledgments

There is no training, fine-tuning, new algorithm, real-robot deployment, or
statistically robust benchmark in this study. The planned one-step action
comparison has no corresponding result and is not reported as completed.
Per-run script snapshots and dependency freezes are incomplete; current wrappers
may include later instrumentation. Historical `reward_trace.json` stores only
the final episode of a multi-episode run, and action traces lack episode IDs.
The curated wrapper additionally saves per-episode reward files for future runs.

Basic syntax, link, metric, provenance, ignore-rule and artifact checks were run;
no clean installation or fresh rollout was performed during curation. See the
[validation report](notes/validation.md). Both README languages use the same
code blocks, tables, figures, videos, and result files.

Credit for the algorithm belongs to the [Diffusion Policy authors](https://diffusion-policy.cs.columbia.edu/),
and for the pretrained model, framework and environment to their upstream
maintainers. See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution
and license scope. This repository was prepared locally for review; nothing was
published during curation.
