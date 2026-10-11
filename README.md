# quant-depth-bench

Measure how **quantization choices affect model quality at growing context depth** — for any
OpenAI-compatible chat-completions server, with zero third-party dependencies (Python ≥ 3.11 stdlib).

The benchmark answers a question throughput benchmarks can't: *does this checkpoint/quant variant
actually still solve tasks when the prompt is 200k, 400k tokens long?*

## Why

Quantizing dense weights (fp8 → q4) or using 4-bit KV/weights (nvfp4) usually costs a little
throughput and, supposedly, "a little quality". At short prompts that loss is invisible. At deep
contexts it is not: quality degradation with depth is **not linear** — a variant can be perfect at
64k and collapse at 384k. This benchmark measures exactly that curve.

Typical finding (real run, GLM-5.3-Flash, 3 serving stacks):

| depth | TensorFold DENSE=fp8 | TensorFold DENSE=q4 | stock vLLM nvfp4 |
|---|---|---|---|
| 8k–64k | ~96–100% | 100% | ~96–100% |
| 192k | **100%** | 93.8% | 87.5% |
| 384k | 89.6% | 81.2% | **68.8%** |
| 480k | 70.0% | 80.0% | 75.0% |

Same model, three quantization stacks — three different degradation onsets.

## What it measures

Per task (prompt = unique seeded filler + task):

1. **code** — 6 coding tasks (RU/UA), answers executed with assertions: correct/incorrect
2. **fact** — 4 exact-answer questions (capitals, arithmetic, history): substring match
3. **needle** — 2 needle-in-a-haystack tasks with a hidden Cyrillic code token: string match

Per reply:

- `finish_reason` — `stop` vs `length` (a "cut" is the primary deep-context failure mode)
- **loop detector** — any 40+ char line repeated ≥ 8× (the classic broken-quantization signature)
- **contamination** — raw special tokens / tool-call debris in the output
- completion tokens, latency

Everything is fully seeded (prompt filler included), so `(depth, kind, lang, idx, seed)` is
reproducible byte-for-byte across runs and stacks.

## Quick start

```bash
# 1. point it at any OpenAI-compatible server
$EDITOR config.toml        # [server] base_url / model

# 2. run a full sweep (one label per variant you compare)
python -m quantdepth.runner fp8

# 3. summarize / compare
python -m quantdepth.compare fp8 q4        # specific labels
python -m quantdepth.compare --all         # everything in results/

# 4. (optional) recover cut rows with a bigger budget
python -m quantdepth.retry_cuts fp8

# tests
python -m unittest discover -s tests -v
```

No install step: run from the repo root. Stdlib only.

## How it works

- **Depths** (`[run] depths_k`) are target prompt-token counts. The filler generator produces
  unique seeded Cyrillic text (~2.1 chars/token, calibrated) — no repeated boilerplate, so a model
  can't cache its way through the haystack.
- **Reduced set** (`[reduced_set]`): very deep depths are slow (prefill-dominated), so the config
  can shrink 480k to 12 code / 4 fact / 4 needle jobs (n=20 instead of 48) while keeping 2 seeds
  and both languages. Disable by setting `enabled = false`.
- **Cut recovery** (`retry_cuts`): because prompts are deterministic, only rows that hit
  `finish=length` need re-asking with a bigger budget (default 4096). Rows that finished with
  `stop` are identical at any budget — so this is a cheap equivalent of a full re-run, and it also
  quantifies *how many* failures were budget-bound vs genuine degradation.
- **Concurrency** (`[run] workers`): keep it within the server's KV-cache capacity. On small GPUs,
  3 concurrent deep-context requests can overflow the cache; 2 is a safe default there.

## Configuration

All knobs live in [`config.toml`](config.toml): server endpoint/model, depths, seeds, languages,
workers, max_tokens, temperature, thinking toggle, reduced-set rules, retry budget. CLI accepts a
path to an alternative config as an argument (`python -m quantdepth.runner q4 my.toml`).

## Output format

`results/results_<label>.jsonl` — one JSON object per reply:

```json
{"depth_k": 192, "kind": "fact", "lang": "RU", "idx": 3, "seed": 1000,
 "finish": "length", "code_ok": null, "exact_ok": false, "loop_max_repeat": 6,
 "loop": false, "contamination": [], "chars": 4831, "text_head": "Тень дорога песня...",
 "prompt_tokens": 191665, "completion_tokens": 1024}
```

`compare` aggregates rows per depth (retry rows override original cuts) and prints a table plus
totals. `results/` and `*.jsonl` are gitignored.

## Tips

- Run every variant you compare with the **same config** (same seeds/depths) — differences are
  then apples-to-apples.
- Keep `workers` low for deep depths; a KV-cache overflow manifests as errors or queue stalls, not
  quality loss, but it will wreck your wall-clock.
- The `cut+filler-echo` signature (answer degenerates into repeating haystack vocabulary) is
  characteristic of deep-context degradation; the loop detector catches the harder variant.
- A bad quantization shows up as **earlier degradation onset**, not as a uniform shift: compare
  *where* each variant's fact/cut curve breaks, not just totals.

## Limitations

Small task set (12 tasks × 2 languages × 2 seeds), synthetic filler — this is a degradation-onset
detector, not a full capability evaluation. `check_lang` for needle replies is intentionally not a
scored metric (models often reason in English before answering in RU/UA).

## Hardware used for the reference run

- TensorFold (EXL3 4bpw) on 2× NVIDIA DGX Spark (TP2), DENSE=fp8 vs DENSE=q4
- stock vLLM serving the official [nvidia/GLM-5.3-Flash-NVFP4](https://huggingface.co/nvidia/GLM-5.3-Flash-NVFP4) checkpoint on 2× DGX Spark (TP2, two nodes; head node holds the served model)

## License

MIT
