"""Plot the last 100 logged steps of an archived single episode using stdlib SVG."""
import argparse
import json
from pathlib import Path


def plot(trace, output, axis):
    records = [json.loads(line) for line in trace.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("Empty trace")
    if [r["step"] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError("Expected consecutive step numbers starting at 1")
    info_path = trace.parent / "eval_info.json"
    if info_path.exists() and len(json.loads(info_path.read_text())["per_episode"]) != 1:
        raise ValueError("Only single-episode traces are supported")
    records = records[-100:]
    steps = [r["step"] for r in records]
    series = [[r[key][axis] for r in records] for key in ["state", "action"]]
    low, high = min(map(min, series)) - 5, max(map(max, series)) + 5
    px = lambda step: 75 + (step - steps[0]) / max(steps[-1] - steps[0], 1) * 940
    py = lambda value: 55 + (high - value) / max(high - low, 1) * 370
    label = "XY"[axis]
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1050" height="500" viewBox="0 0 1050 500">',
           '<rect width="1050" height="500" fill="white"/>',
           f'<text x="75" y="30" font-size="20">PushT: {label} position, logged steps {steps[0]}-{steps[-1]}</text>']
    for i in range(6):
        value = low + (high - low) * i / 5
        y = py(value)
        svg += [f'<line x1="75" y1="{y:.2f}" x2="1015" y2="{y:.2f}" stroke="#ddd"/>',
                f'<text x="65" y="{y+4:.2f}" text-anchor="end" font-size="12">{value:.1f}</text>']
    for r in records:
        if r["replanned"]:
            x = px(r["step"])
            svg.append(f'<line x1="{x:.2f}" y1="55" x2="{x:.2f}" y2="425" stroke="#aaa" opacity="0.4"/>')
    for values, color in zip(series, ["#1677bb", "#e07824"]):
        points = " ".join(f"{px(s):.2f},{py(v):.2f}" for s, v in zip(steps, values))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    for s in [steps[0], *[s for s in steps[1:-1] if s % 10 == 0], steps[-1]]:
        svg.append(f'<text x="{px(s):.2f}" y="447" text-anchor="middle" font-size="12">{s}</text>')
    svg += ['<text x="75" y="480" fill="#1677bb" font-size="15">Blue: observed pusher</text>',
            '<text x="350" y="480" fill="#e07824" font-size="15">Orange: action target</text>',
            '<text x="650" y="480" font-size="15">Gray: replanning</text>', '</svg>']
    with output.open("x", encoding="utf-8") as handle:
        handle.write("\n".join(svg) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--axis", choices=["x", "y"], default="x")
    args = parser.parse_args()
    plot(args.trace, args.output, "xy".index(args.axis))
