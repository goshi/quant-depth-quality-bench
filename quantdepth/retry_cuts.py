#!/usr/bin/env python3
"""Re-ask only the finish=length rows of a finished series with a bigger max_tokens budget.

Rationale: prompts are fully seeded, so (seed, task) reproduces the exact same prompt. A row that
finished with `stop` is unchanged by the cap, therefore re-asking only cut rows is equivalent to a
full re-run at the higher budget - at a fraction of the cost.

Usage: python -m quantdepth.retry_cuts <label> [config.toml] [--retry-out PATH]
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import tasks
from .client import body_for, chat
from .config import load

CUT = "length"


def collect_cuts(label: str, out_dir: Path) -> list[dict]:
    cuts = []
    for path in sorted(out_dir.glob(f"results_{label}*.jsonl")):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                if "error" not in r and r.get("finish") == CUT:
                    cuts.append(r)
    return cuts


def main() -> None:
    label = sys.argv[1]
    cfg_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "config.toml"
    cfg = load(cfg_path)
    retry_max = cfg.retry.max_tokens
    out = Path(cfg.run.out_dir)
    retry_path = out / f"retry{retry_max}_{label}.jsonl"

    cuts = collect_cuts(label, out)
    done: set[tuple] = set()
    if retry_path.exists():
        done = {(r["depth_k"], r["kind"], r["lang"], r["idx"], r["seed"])
                for r in map(json.loads, open(retry_path, encoding="utf-8"))}
    jobs = [r for r in cuts if (r["depth_k"], r["kind"], r["lang"], r["idx"], r["seed"]) not in done]
    print(f"[retry{retry_max}:{label}] {len(cuts)} cut rows, {len(jobs)} to retry", flush=True)

    def one(rec: dict) -> dict:
        job_kind, lang, idx, depth, seed = rec["kind"], rec["lang"], rec["idx"], rec["depth_k"], rec["seed"]
        prompt, needle = tasks.build_prompt(job_kind, lang.lower(), idx, depth, seed)
        body = body_for([{"role": "user", "content": prompt}],
                        max_tokens=retry_max, temperature=cfg.run.temperature,
                        seed=seed, thinking=cfg.run.thinking)
        new = {"depth_k": depth, "kind": job_kind, "lang": lang, "idx": idx, "seed": seed,
               "retry_max_tokens": retry_max}
        try:
            r = chat(cfg.server.base_url, body, cfg.server.model, cfg.server.request_timeout)
            choice = r["choices"][0]
            text = choice["message"].get("content") or ""
            new["finish"] = choice["finish_reason"]
            new["completion_tokens"] = r.get("usage", {}).get("completion_tokens")
            if job_kind == "code":
                new["code_ok"], new["exact_ok"] = tasks.check_code(idx, text), None
            elif job_kind == "fact":
                new["code_ok"], new["exact_ok"] = None, tasks.check_fact(idx, lang, text)
            else:
                new["code_ok"], new["exact_ok"] = None, tasks.check_needle(needle, text)
            rep = tasks.loop_repeats(text)
            new["loop_max_repeat"] = rep
            new["loop"] = rep >= 8
            new["text_head"] = text[:100]
        except Exception as exc:  # noqa: BLE001
            new["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return new

    started = time.monotonic()
    with ThreadPoolExecutor(cfg.run.workers) as ex, open(retry_path, "a", buffering=1) as fh:
        for n, rec in enumerate(ex.map(one, jobs), 1):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[retry {n}/{len(jobs)}] d{rec['depth_k']}k {rec['kind']}/{rec['lang']}/s{rec['seed']} "
                  f"finish={rec.get('finish')} pass={rec.get('code_ok') if rec['kind'] == 'code' else rec.get('exact_ok')}",
                  flush=True)
    print(f"[retry{retry_max}:{label}] done in {time.monotonic() - started:.0f}s -> {retry_path}", flush=True)


if __name__ == "__main__":
    main()
