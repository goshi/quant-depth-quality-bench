#!/usr/bin/env python3
"""Aggregate one or several series into a per-depth comparison table.

Usage:
  python -m quantdepth.compare fp8 q4 vllm23 [--config config.toml]
  python -m quantdepth.compare --all          (every series found in the results dir)

Reads <out_dir>/results_<label>.jsonl (+ retry rows if present) and prints
code/fact/needle pass rates, length-cuts and loops per depth, plus totals.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from .config import load


def load_series(label: str, out_dir: Path, retry_max: int) -> dict:
    """Rows per depth; retry rows (stop) replace the original cut outcome."""
    rows: dict[int, dict] = {}

    def add(r: dict) -> None:
        if "error" in r:
            return
        key = (r["depth_k"], r["kind"], r["lang"], r["idx"], r["seed"])
        d = rows.setdefault(r["depth_k"], {})
        if key in d and r.get("finish") == CUT_SENTINEL:
            return  # do not let an original cut overwrite a retried row
        d[key] = r

    CUT_SENTINEL = "length"
    main = out_dir / f"results_{label}.jsonl"
    if not main.exists():
        return {}
    for line in open(main, encoding="utf-8"):
        if line.strip():
            add(json.loads(line))
    retry = out_dir / f"retry{retry_max}_{label}.jsonl"
    if retry.exists():
        for line in open(retry, encoding="utf-8"):
            if line.strip():
                add(json.loads(line))
    return {d: list(v.values()) for d, v in sorted(rows.items())}


def pass_of(r: dict) -> bool | None:
    if r["kind"] == "code":
        return r.get("code_ok")
    return r.get("exact_ok")


def agg(rows: list[dict]) -> dict:
    code = [r for r in rows if r["kind"] == "code"]
    fact = [r for r in rows if r["kind"] == "fact"]
    needle = [r for r in rows if r["kind"] == "needle"]
    return {
        "n": len(rows),
        "code": f"{sum(bool(pass_of(r)) for r in code)}/{len(code)}",
        "fact": f"{sum(bool(pass_of(r)) for r in fact)}/{len(fact)}",
        "needle": f"{sum(bool(pass_of(r)) for r in needle)}/{len(needle)}",
        "cuts": sum(r.get("finish") == "length" for r in rows),
        "loops": sum(bool(r.get("loop")) for r in rows),
    }


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cfg_path = next((a for a in sys.argv[1:] if a.endswith(".toml")), "config.toml")
    cfg = load(cfg_path)
    out_dir = Path(cfg.run.out_dir)

    if "--all" in sys.argv:
        labels = sorted({
            m.group(1) for p in out_dir.glob("results_*.jsonl")
            if (m := re.match(r"results_(.+)\.jsonl$", p.name))
        })
    else:
        labels = args
    if not labels:
        print("no series found / no labels given")
        return

    data = {lbl: load_series(lbl, out_dir, cfg.retry.max_tokens) for lbl in labels}
    data = {lbl: d for lbl, d in data.items() if d}
    depths = sorted({d for series in data.values() for d in series})
    if not depths:
        print("no data")
        return

    metrics = ("n", "code", "fact", "needle", "cuts", "loops")
    width = 11
    print(f"{'depth':>6} | {'metric':<6} | " + " | ".join(lbl.rjust(width) for lbl in data))
    print("-" * (12 + len(data) * (width + 3)))
    for d in depths:
        for m in metrics:
            cells = []
            for lbl in data:
                a = agg(data[lbl][d]) if d in data[lbl] else None
                cells.append((str(a[m]) if a else "-").rjust(width))
            print(f"{d:>5}k | {m:<6} | " + " | ".join(cells))
        print("-" * (12 + len(data) * (width + 3)))

    print("totals:")
    for lbl, series in data.items():
        flat = [r for rows in series.values() for r in rows]
        a = agg(flat)
        total = sum(int(x.split("/")[0]) for x in (a["code"], a["fact"], a["needle"]))
        denom = sum(int(x.split("/")[1]) for x in (a["code"], a["fact"], a["needle"]))
        print(f"  {lbl}: {total}/{denom} ({100 * total / denom:.1f}%) "
              f"cuts={a['cuts']} loops={a['loops']}")


if __name__ == "__main__":
    main()
