# Local validation — 2026-10-21 (Asia/Shanghai)

The independent WSL project was created without editing, moving, or deleting
the original experiment files. No training, evaluation rollout, installation,
large download, GitHub repository creation, remote configuration, or push was
performed. At completion of curation, the local WSL repository had an initial `main` branch
and no commits or remotes; publication preparation is a later step.

## Completed checks

- Parsed all seven selected Python scripts and checked both Bash launchers with
  `bash -n`. The dependency-free checker was also run with the original Python
  3.12 interpreter. No policy imports or CUDA inference are needed for these checks.
- Checked all 40 relative Markdown links across the READMEs and notes, including
  language navigation, images, videos, and result files. English and Chinese
  have seven matching main sections and identical code blocks, tables, and
  link/media targets. Both versions were reviewed for consistent caveats.
- Recomputed all 13 run summaries from per-episode JSON, checking aggregate
  rewards and success percentages. All archived exit codes are zero.
  Checked `summary.csv` and `reproducibility.json` against their source records.
- Confirmed logged settings for the 4/8-step comparison: base seed 1000,
  five episodes, CUDA, AMP=false, horizon 16, `num_inference_steps=None` (100
  configured timesteps), and batch size 1. Deterministic logs contain the
  expected deterministic-mode marker.
- Rechecked deterministic action/state trace equality at byte level. Reward
  sequences differ by at most `3.3306690738754696e-16`; ordinary fixed-seed
  traces differ in both content and length. No stronger determinism claim is made.
- Fully decoded all 29 copied MP4s (6,331 frames) using the existing PyAV
  installation, without running simulation. Results are recorded in
  `results/video_validation.json`. Visually inspected the existing rollout
  preview; it contains simulation frames and episode labels.
- Verified the original checkpoint size (1,050,862,408 bytes) and SHA256 against
  the archived upstream model metadata, without copying or loading the weights.
- Ran `uv pip check` against the existing experiment environment: all 124
  installed packages were compatible. This is not a clean-install test and does
  not prove the historical freeze captures each run's environment.
- Checked `.gitignore` with `git check-ignore`: virtual environments, caches,
  model weights, checkpoints, local secrets and future `runs/` outputs are
  ignored; archived `results/summary.csv` is included.
- Scanned selected text/code/log artifacts for common Hugging Face/GitHub/OpenAI
  token patterns, private keys, credential-bearing URLs, and private home paths.
  No matching secrets remain. Public logs use `<PROJECT_ROOT>`/`<HOME>` markers,
  and JSON video paths are repository-relative. Pattern scanning is not a proof
  that all possible sensitive formats have been ruled out.
- Confirmed no weights, environments, downloaded wheels, or upstream source
  trees are among the public candidate files. The project is about 3.2 MB;
  the largest file is the 182,627-byte `test-01` action trace.
- Source integrity audit compared 415 original project files before and after
  curation (including the model, results, scripts, notes and source snapshots).
  All SHA256 hashes matched. Installed environments, tools and caches were
  excluded from this content audit; they were not modified by this workflow.
  The detailed original integrity audit stays outside the public project.
- Checked the four unique external README reference targets. LeRobot v0.3.3,
  gym-pusht and the Hugging Face model page were accessible. The Diffusion Policy
  project page timed out through the browsing tool, so its current reachability
  is unverified. The fixed model revision is supported by archived metadata;
  direct access to its revision-tree page was unavailable through the tool.

## Practical limits

The newly packaged launchers and reward-file fix have syntax/static checks but
have not been exercised in a new policy rollout. Rebuilding the environment from
scratch, network/package availability, and behavior on another GPU remain
unverified. Historical per-run code snapshots and environment freezes are missing.
The available results support an undergraduate reproduction/analysis showcase,
not a new algorithm or a robust comparative performance claim.

The project is ready for a local content review and public display with these
limitations stated. No blanket license was selected for project-local work;
this does not prevent display, but open-source reuse terms remain a separate
choice. Upstream attribution and installed package license texts are included.
