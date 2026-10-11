"""Job generation and the concurrent runner that executes a depth sweep and writes jsonl rows."""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from . import tasks
from .client import body_for, chat
from .config import Config


@dataclass
class Job:
    depth_k: int
    kind: str  # code | fact | needle
    lang: str  # ru | ua
    idx: int
    seed: int


def build_jobs(cfg: Config) -> list[Job]:
    reduced = set(cfg.reduced_set.depths_k) if cfg.reduced_set.enabled else set()
    jobs: list[Job] = []
    for depth in cfg.run.depths_k:
        n_code, n_fact, n_needle = 6, 4, 2
        if depth in reduced:
            n_code = min(cfg.reduced_set.code_tasks, 6)
            n_fact = min(cfg.reduced_set.fact_tasks, 4)
            n_needle = min(cfg.reduced_set.needle_tasks, 2)
        for lang in cfg.run.languages:
            lang = lang.lower()
            for i in range(n_code):
                for s in cfg.run.seeds:
                    jobs.append(Job(depth, "code", lang, i, s))
            for i in range(n_fact):
                for s in cfg.run.seeds:
                    jobs.append(Job(depth, "fact", lang, i, s))
            for i in range(n_needle):
                for s in cfg.run.seeds:
                    jobs.append(Job(depth, "needle", lang, i, s))
    return jobs


def run_job(job: Job, cfg: Config) -> dict:
    prompt, needle = tasks.build_prompt(job.kind, job.lang, job.idx, job.depth_k, job.seed)
    body = body_for(
        [{"role": "user", "content": prompt}],
        max_tokens=cfg.run.max_tokens,
        temperature=cfg.run.temperature,
        seed=job.seed,
        thinking=cfg.run.thinking,
    )
    rec: dict = {
        "depth_k": job.depth_k, "kind": job.kind, "lang": job.lang.upper(),
        "idx": job.idx, "seed": job.seed,
    }
    try:
        t0 = time.monotonic()
        r = chat(cfg.server.base_url, body, cfg.server.model, cfg.server.request_timeout)
        rec["latency_s"] = round(time.monotonic() - t0, 1)
        choice = r["choices"][0]
        text = choice["message"].get("content") or ""
        rec["finish"] = choice["finish_reason"]
        rec["completion_tokens"] = r.get("usage", {}).get("completion_tokens")
        rec["prompt_tokens"] = r.get("usage", {}).get("prompt_tokens")
        if job.kind == "code":
            rec["code_ok"] = tasks.check_code(job.idx, text)
            rec["exact_ok"] = None
        elif job.kind == "fact":
            rec["code_ok"] = None
            rec["exact_ok"] = tasks.check_fact(job.idx, job.lang, text)
        else:
            rec["code_ok"] = None
            rec["exact_ok"] = tasks.check_needle(needle, text)
        rep = tasks.loop_repeats(text)
        rec["loop_max_repeat"] = rep
        rec["loop"] = rep >= 8
        rec["contamination"] = tasks.check_contamination(text)
        rec["chars"] = len(text)
        rec["text_head"] = text[:100]
    except Exception as exc:  # network / server errors must not kill the sweep
        rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
        for k in ("finish", "code_ok", "exact_ok", "loop_max_repeat"):
            rec.setdefault(k, None)
        rec["contamination"] = []
        rec["loop"] = False
        rec.setdefault("completion_tokens", None)
        rec.setdefault("prompt_tokens", None)
        rec["chars"] = 0
    return rec


def run_sweep(cfg: Config, label: str, out_dir: str | None = None) -> Path:
    """Execute all jobs, append rows to <out_dir>/results_<label>.jsonl, print progress."""
    out = Path(out_dir or cfg.run.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"results_{label}.jsonl"
    jobs = build_jobs(cfg)
    total = len(jobs)
    print(f"[{label}] {total} jobs, workers={cfg.run.workers}, "
          f"depths={cfg.run.depths_k} -> {path}", flush=True)
    started = time.monotonic()
    with ThreadPoolExecutor(cfg.run.workers) as ex, open(path, "a", buffering=1) as fh:
        for n, rec in enumerate(ex.map(lambda j: run_job(j, cfg), jobs), 1):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            verdict = rec.get("code_ok") if rec["kind"] == "code" else rec.get("exact_ok")
            print(f"[{n}/{total}] d{rec['depth_k']}k {rec['kind']}/{rec['lang']}/s{rec['seed']} "
                  f"finish={rec.get('finish')} pass={verdict} loop={rec.get('loop')}", flush=True)
    print(f"[{label}] done in {time.monotonic() - started:.0f}s", flush=True)
    return path


def summarize(label: str, out_dir: str | None = None) -> str:
    """Human-readable per-depth summary of one series (stdout)."""
    out = Path(out_dir or "results")
    path = out / f"results_{label}.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    depths = sorted({r["depth_k"] for r in rows})
    lines = [f"series '{label}' ({len(rows)} rows)"]
    for d in depths:
        rs = [r for r in rows if r["depth_k"] == d and "error" not in r]
        code = [r for r in rs if r["kind"] == "code"]
        fact = [r for r in rs if r["kind"] == "fact"]
        needle = [r for r in rs if r["kind"] == "needle"]
        lines.append(
            f"  d{d}k: n={len(rs)} code {sum(bool(r['code_ok']) for r in code)}/{len(code)} "
            f"fact {sum(bool(r['exact_ok']) for r in fact)}/{len(fact)} "
            f"needle {sum(bool(r['exact_ok']) for r in needle)}/{len(needle)} "
            f"cuts {sum(r['finish'] == 'length' for r in rs)} "
            f"loops {sum(r['loop'] for r in rs)}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # usage: python -m quantdepth.runner <label> [config.toml]
    cfg = __import__("quantdepth.config", fromlist=["load"]).load(sys.argv[2] if len(sys.argv) > 2 else "config.toml")
    run_sweep(cfg, sys.argv[1])
    print(summarize(sys.argv[1]))
