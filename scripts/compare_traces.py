"""Compare two archived single-episode traces; no simulation or model load."""
import argparse
import json
from pathlib import Path


def compare(a, b):
    for folder in [a, b]:
        info = json.loads((folder / "eval_info.json").read_text())
        if len(info["per_episode"]) != 1:
            raise ValueError("Only single-episode runs are supported")
    trace_a = (a / "action_step_trace.jsonl").read_bytes()
    trace_b = (b / "action_step_trace.jsonl").read_bytes()
    reward_a = json.loads((a / "reward_trace.json").read_text())
    reward_b = json.loads((b / "reward_trace.json").read_text())
    return {
        "action_state_trace_bytes_equal": trace_a == trace_b,
        "action_state_trace_lengths": [len(trace_a.splitlines()), len(trace_b.splitlines())],
        "reward_lengths": [len(reward_a), len(reward_b)],
        "reward_exact_equal": reward_a == reward_b,
        "reward_max_abs_diff": (max((abs(x - y) for x, y in zip(reward_a, reward_b)), default=0)
                                if len(reward_a) == len(reward_b) else None),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.run_a, args.run_b), indent=2))
