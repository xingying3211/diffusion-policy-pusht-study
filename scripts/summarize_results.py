"""Summarize archived eval_info.json files without importing LeRobot or torch."""
import argparse
import csv
import json
import math
from pathlib import Path


def rows(root):
    for path in sorted((root / "results").glob("*/eval_info.json")):
        info = json.loads(path.read_text())
        episodes = info["per_episode"]
        aggregate = info["aggregated"]
        successes = sum(ep["success"] for ep in episodes)
        assert math.isclose(100 * successes / len(episodes), aggregate["pc_success"])
        for name in ["sum_reward", "max_reward"]:
            assert math.isclose(sum(ep[name] for ep in episodes) / len(episodes),
                                aggregate["avg_" + name], rel_tol=1e-12)
        trace = json.loads((path.parent / "observation_action_trace.json").read_text())
        yield {
            "run": path.parent.name,
            "n_action_steps": trace["policy"]["n_action_steps"],
            "episodes": len(episodes),
            "successes": successes,
            "success_percent": aggregate["pc_success"],
            "avg_sum_reward": aggregate["avg_sum_reward"],
            "avg_max_reward": aggregate["avg_max_reward"],
            "eval_seconds": aggregate["eval_s"],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new CSV path; refuses overwrite")
    args = parser.parse_args()
    data = list(rows(Path(__file__).resolve().parents[1]))
    if args.output:
        with args.output.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    else:
        for row in data:
            print(f'{row["run"]}: {row["successes"]}/{row["episodes"]}, '
                  f'n_action_steps={row["n_action_steps"]}, '
                  f'eval_s={row["eval_seconds"]:.3f}')


if __name__ == "__main__":
    main()
