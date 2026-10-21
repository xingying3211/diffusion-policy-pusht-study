#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MODEL_DIR="${MODEL_DIR:-$ROOT/model}"
export SDL_VIDEODRIVER=dummy
export SDL_AUDIODRIVER=dummy
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export CUBLAS_WORKSPACE_CONFIG=:4096:8
RUN_NAME="${1:-pusht-$(date +%Y%m%d-%H%M%S)}"
EPISODES="${2:-3}"
ACTION_STEPS="${3:-8}"
[[ "$ACTION_STEPS" =~ ^(2|4|8)$ ]] || {
    echo "ACTION_STEPS must be 2, 4, or 8"
    exit 2
}
[[ "$RUN_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo 'Invalid run name'; exit 2; }
[[ "$EPISODES" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid episode count'; exit 2; }
export RUN_DIR="$ROOT/runs/$RUN_NAME"
PYTHON="$(command -v "$PYTHON")"
test -x "$PYTHON" || { echo 'Missing independent virtual environment'; exit 2; }
test -f "$MODEL_DIR/config.json" || { echo "Missing model config"; exit 2; }
test -f "$MODEL_DIR/model.safetensors" || { echo 'Missing pretrained model'; exit 2; }
test ! -e "$RUN_DIR" || { echo "Result directory already exists: $RUN_DIR"; exit 2; }
mkdir -p "$RUN_DIR"
"$PYTHON" - "$EPISODES" "$ACTION_STEPS" <<'PY'
import json, os, sys
from pathlib import Path
record = {"n_episodes": int(sys.argv[1]), "n_action_steps": int(sys.argv[2]),
          "seed": 1000, "batch_size": 1, "device": "cuda", "use_amp": False,
          "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
          "checkpoint": "User-supplied local MODEL_DIR; verify with scripts/verify_model.py"}
(Path(os.environ["RUN_DIR"]) / "run_config.json").write_text(json.dumps(record, indent=2))
PY
date -Iseconds > "$RUN_DIR/start_time.txt"
nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu --format=csv -l 1 > "$RUN_DIR/gpu_samples.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true' EXIT
set +e
"$PYTHON" "$ROOT/scripts/eval_cuda_deterministic.py" \
  --policy.path="$MODEL_DIR" \
  --policy.n_action_steps="$ACTION_STEPS" \
  --env.type=pusht \
  --eval.batch_size=1 \
  --eval.n_episodes="$EPISODES" \
  --policy.use_amp=false \
  --policy.device=cuda \
  --seed=1000 \
  --output_dir="$RUN_DIR" 2>&1 | tee "$RUN_DIR/eval.log"
STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$STATUS" > "$RUN_DIR/exit_code.txt"
date -Iseconds > "$RUN_DIR/end_time.txt"
exit "$STATUS"
