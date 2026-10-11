"""Unit tests (stdlib unittest): task checkers, filler determinism, job counts.

Run: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantdepth import tasks
from quantdepth.config import load
from quantdepth.runner import build_jobs

GOLD = {
    0: "import heapq\ndef merge_sorted(lists):\n    h=[]\n    for l in lists: h.extend(l)\n    heapq.heapify(h)\n    out=[]\n    while h: out.append(heapq.heappop(h))\n    return out\n",
    1: "class LRU:\n    def __init__(self, capacity):\n        self.cap=capacity; self.d={}\n    def get(self,key):\n        if key not in self.d: return -1\n        v=self.d.pop(key); self.d[key]=v; return v\n    def put(self,key,value):\n        if key in self.d: self.d.pop(key)\n        elif len(self.d)>=self.cap: self.d.pop(next(iter(self.d)))\n        self.d[key]=value\n",
    2: "def to_roman(n):\n    vals=[(1000,'M'),(900,'CM'),(500,'D'),(400,'CD'),(100,'C'),(90,'XC'),(50,'L'),(40,'XL'),(10,'X'),(9,'IX'),(5,'V'),(4,'IV'),(1,'I')]\n    out=''\n    for v,s in vals:\n        while n>=v: out+=s; n-=v\n    return out\n",
    3: "def balanced(s):\n    stack=[]; pairs={')':'(',']':'[','}':'{'}\n    for ch in s:\n        if ch in '([{': stack.append(ch)\n        elif ch in pairs:\n            if not stack or stack.pop()!=pairs[ch]: return False\n    return not stack\n",
    4: "from collections import deque\ndef topo_sort(n,edges):\n    adj=[[] for _ in range(n)]; indeg=[0]*n\n    for u,v in edges: adj[u].append(v); indeg[v]+=1\n    q=deque(i for i in range(n) if indeg[i]==0); out=[]\n    while q:\n        u=q.popleft(); out.append(u)\n        for v in adj[u]:\n            indeg[v]-=1\n            if indeg[v]==0: q.append(v)\n    return out if len(out)==n else None\n",
    5: "def spiral(m):\n    if not m: return []\n    res=[]\n    while m:\n        res.extend(m[0])\n        m=[list(r) for r in zip(*m[1:])][::-1]\n    return res\n",
}


class TestCheckers(unittest.TestCase):
    def test_golden_solutions_pass(self):
        for idx, src in GOLD.items():
            with self.subTest(task=idx):
                self.assertTrue(tasks.check_code(idx, f"```python\n{src}\n```"))

    def test_wrong_solution_fails(self):
        self.assertFalse(tasks.check_code(2, "```python\ndef to_roman(n):\n    return 'I'*n\n```"))
        self.assertFalse(tasks.check_code(0, "```python\ndef merge_sorted(lists):\n    return []\n```"))

    BALANCED_RAW = (
        "def balanced(s):\n"
        "    stack = []\n"
        "    pairs = {')': '(', ']': '[', '}': '{'}\n"
        "    for ch in s:\n"
        "        if ch in '([{':\n"
        "            stack.append(ch)\n"
        "        elif ch in pairs:\n"
        "            if not stack or stack.pop() != pairs[ch]:\n"
        "                return False\n"
        "    return not stack\n"
    )

    def test_no_code_block_falls_back_to_raw_text(self):
        # no fences -> extract_code falls back to the whole reply; a correct plain-text solution must pass
        self.assertTrue(tasks.check_code(3, self.BALANCED_RAW))

    def test_fact_matching(self):
        self.assertTrue(tasks.check_fact(0, "ru", "Канберра"))
        self.assertFalse(tasks.check_fact(0, "ru", "Сидней"))

    def test_needle_found(self):
        self.assertTrue(tasks.check_needle("ЖУРАВЛЬ-7741", "код: ЖУРАВЛЬ-7741"))
        self.assertFalse(tasks.check_needle(None, "ЖУРАВЛЬ-7741"))


class TestLoopsAndContamination(unittest.TestCase):
    def test_loop_detector(self):
        line = "def balanced(s):     stack = []  # some long line here indeed"
        text = "\n".join([line] * 8)
        self.assertGreaterEqual(tasks.loop_repeats(text), 8)
        self.assertEqual(tasks.loop_repeats("short\nlines\nonly"), 0)

    def test_contamination(self):
        self.assertEqual(tasks.check_contamination("clean answer"), [])
        self.assertIn("<|", tasks.check_contamination("leak <|special|>"))


class TestFillerAndPrompts(unittest.TestCase):
    def test_filler_deterministic_per_job(self):
        a = tasks.filler_text(8, 1000, "code0")
        b = tasks.filler_text(8, 1000, "code0")
        c = tasks.filler_text(8, 1001, "code0")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_needle_prompt_contains_hidden_code(self):
        prompt, needle = tasks.build_prompt("needle", "ru", 0, 1, 1000)
        self.assertIn(needle, prompt)
        self.assertGreater(len(prompt), 2000)  # 1k-token filler is present

    def test_fact_prompt_contains_question(self):
        prompt, _ = tasks.build_prompt("fact", "ua", 0, 1, 1001)
        self.assertIn("Столиця Австралії", prompt)  # the question is present, filler prepended


class TestJobCounts(unittest.TestCase):
    def test_full_and_reduced_counts(self):
        cfg = load(Path(__file__).resolve().parents[1] / "config.toml")
        jobs = build_jobs(cfg)
        full = [j for j in jobs if j.depth_k == 8]
        reduced = [j for j in jobs if j.depth_k == 480]
        # full: (6 code + 4 fact + 2 needle) x 2 langs x 2 seeds
        self.assertEqual(len(full), 48)
        self.assertEqual(len({(j.kind, j.lang, j.idx, j.seed) for j in full}), 48)
        # reduced: (3 code + 1 fact + 1 needle) x 2 langs x 2 seeds
        self.assertEqual(len(reduced), 20)

    def test_all_jobs_have_unique_keys(self):
        cfg = load(Path(__file__).resolve().parents[1] / "config.toml")
        keys = [(j.depth_k, j.kind, j.lang, j.idx, j.seed) for j in build_jobs(cfg)]
        self.assertEqual(len(keys), len(set(keys)))


if __name__ == "__main__":
    unittest.main()
