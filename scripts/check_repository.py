"""Lightweight offline checks. Does not import the policy or start evaluation."""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

from summarize_results import rows


def main():
    root = Path(__file__).resolve().parents[1]
    files = [p for p in root.rglob("*") if p.is_file()
             and not any(part in {".git", "__pycache__", "runs", "model", ".venv"}
                         for part in p.relative_to(root).parts)]
    for path in files:
        assert path.stat().st_size < 5 * 1024 * 1024, path
        assert path.suffix not in {".safetensors", ".pt", ".pth", ".ckpt", ".whl"}, path
        if path.suffix == ".py":
            ast.parse(path.read_text(), filename=str(path))
        if path.suffix in {".md", ".py", ".sh", ".txt", ".json", ".jsonl", ".csv", ".log", ".svg"}:
            content = path.read_text()
            # Common token/private-key patterns; this is not an exhaustive secret scanner.
            patterns = [r"hf_[A-Za-z0-9]{20,}", r"gh[pousr]_[A-Za-z0-9]{20,}",
                        r"sk-[A-Za-z0-9_-]{20,}", r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
                        "/" + r"home/[^/\s]+/", r"[A-Za-z]:\\Users\\",
                        r"https?://(?:[^/\s@]+:[^/\s@]+@)"]
            assert not any(re.search(pattern, content) for pattern in patterns), path
    link_count = 0
    for path in [p for p in files if p.suffix == ".md"]:
        for href in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", path.read_text()):
            if urlsplit(href).scheme:
                continue
            target = unquote(href.split("#")[0])
            assert (path.parent / target).exists(), (path, href)
            link_count += 1
    en, zh = [(root / name).read_text() for name in ["README.md", "README.zh-CN.md"]]
    navigation = "[English](README.md) | [简体中文](README.zh-CN.md)"
    assert en.startswith(navigation) and zh.startswith(navigation)
    assert re.findall(r"```[\s\S]*?```", en) == re.findall(r"```[\s\S]*?```", zh)
    assert [line for line in en.splitlines() if line.startswith("|")] == [
        line for line in zh.splitlines() if line.startswith("|")]
    assert len(re.findall(r"^## ", en, re.M)) == len(re.findall(r"^## ", zh, re.M)) == 7
    links = lambda text: re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    assert links(en) == links(zh), "README link/media targets must match in order"
    shell_files = list((root / "scripts").glob("*.sh"))
    for path in shell_files:
        subprocess.run(["bash", "-n", str(path)], check=True)
    manifest = json.loads((root / "results/curation_manifest.json").read_text())
    for entry in manifest:
        path = root / entry["destination"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["destination_sha256"], path
    from compare_traces import compare
    comparisons = json.loads((root / "results/reproducibility.json").read_text())
    assert comparisons["ordinary"] == compare(root / "results/dp-repro-a", root / "results/dp-repro-b")
    assert comparisons["deterministic"] == compare(root / "results/dp-det-a", root / "results/dp-det-b")
    import csv
    run_rows = list(rows(root))
    assert len(run_rows) == 13
    with (root / "results/summary.csv").open(newline="") as handle:
        archived_rows = list(csv.DictReader(handle))
    assert archived_rows == [{key: str(value) for key, value in row.items()} for row in run_rows]
    for path in (root / "results").glob("*/eval_info.json"):
        info = json.loads(path.read_text())
        assert (path.parent / "exit_code.txt").read_text().strip() == "0"
        for video in info.get("video_paths", []):
            assert (root / video).is_file(), video
    if (root / ".git").exists():
        ignored = [".venv/bin/python", ".venv-1/bin/python", "model/model.safetensors",
                   "foo.pth", "runs/trial/eval_info.json", ".env", ".uv-cache/test", "cache.whl"]
        for item in ignored:
            result = subprocess.run(["git", "check-ignore", "--quiet", item], cwd=root)
            assert result.returncode == 0, item
        result = subprocess.run(["git", "check-ignore", "--quiet", "results/summary.csv"], cwd=root)
        assert result.returncode == 1
    print(f"PASS: {len(files)} files, {link_count} local Markdown links, "
          f"13 runs, source hashes, syntax, bilingual parity, artifact/secret patterns.")
    print("External URLs and a fresh environment/rollout require separate validation.")


if __name__ == "__main__":
    main()
