"""Task definitions: exec-checked coding tasks, exact-answer facts, needle-in-haystack (Cyrillic)."""
from __future__ import annotations

import random
import re
from collections import Counter
from functools import lru_cache

# ---------------------------------------------------------------- prompts
CODE_RU = [
    "Напиши функцию Python `merge_sorted(lists)`, которая сливает список отсортированных списков в один отсортированный за O(N log k) с помощью heapq. Ответь только кодом.",
    "Напиши класс Python `LRU(capacity)` с методами `get(key)` (возвращает -1, если ключа нет) и `put(key, value)`, вытеснение по давности использования, O(1). Ответь только кодом.",
    "Напиши функцию Python `to_roman(n)` (1 ≤ n ≤ 3999), возвращающую число римскими цифрами. Ответь только кодом.",
    "Напиши функцию Python `balanced(s)`, возвращающую True, если скобки (), [], {} в строке правильно вложены. Ответь только кодом.",
    "Напиши функцию Python `topo_sort(n, edges)`, возвращающую топологический порядок вершин 0..n-1 (edges — список (u,v): u раньше v), или None, если есть цикл. Ответь только кодом.",
    "Напиши функцию Python `spiral(m)`, возвращающую элементы матрицы (список списков) при обходе по спирали по часовой стрелке из левого верхнего угла. Ответь только кодом.",
]
CODE_UA = [
    "Напиши функцію Python `merge_sorted(lists)`, яка зливає список відсортованих списків в один відсортований за O(N log k) за допомогою heapq. Відповідай лише кодом.",
    "Напиши клас Python `LRU(capacity)` з методами `get(key)` (повертає -1, якщо ключа немає) та `put(key, value)`, витіснення за давністю використання, O(1). Відповідай лише кодом.",
    "Напиши функцію Python `to_roman(n)` (1 ≤ n ≤ 3999), що повертає число римськими цифрами. Відповідай лише кодом.",
    "Напиши функцію Python `balanced(s)`, що повертає True, якщо дужки (), [], {} у рядку правильно вкладені. Відповідай лише кодом.",
    "Напиши функцію Python `topo_sort(n, edges)`, що повертає топологічний порядок вершин 0..n-1 (edges — список (u,v): u раніше v), або None, якщо є цикл. Відповідай лише кодом.",
    "Напиши функцію Python `spiral(m)`, що повертає елементи матриці (список списків) під час обходу спіраллю за годинниковою стрілкою з лівого верхнього кута. Відповідай лише кодом.",
]

FACT_RU = [
    ("Столица Австралии? Ответь одним словом.", ["канберра"]),
    ("Сколько минут в трёх сутках? Ответь только числом.", ["4320"]),
    ("В каком году пала Берлинская стена? Ответь только годом.", ["1989"]),
    ("Какое самое глубокое озеро мира? Ответь одним словом.", ["байкал"]),
]
FACT_UA = [
    ("Столиця Австралії? Відповідь одним словом.", ["канберра"]),
    ("Скільки хвилин у трьох добах? Відповідай лише числом.", ["4320"]),
    ("У якому році впала Берлинська стіна? Відповідай лише роком.", ["1989"]),
    ("Яке найглибше озеро світу? Відповідь одним словом.", ["байкал"]),
]

NEEDLE_RU = "Где-то в этом тексте спрятан секретный код. В конце ответь только этим кодом.\n"
NEEDLE_UA = "Десь у цьому тексті захований секретний код. Наприкінці відповім лише цим кодом.\n"
NEEDLE_TOKENS = ["ЖУРАВЛЬ-7741", "БАРВИНОК-3382", "ОСОКА-5517", "КОСУЛЯ-9024"]

RU_WORDS = "поле река лес ветер камень дорога тень крыша почта море песня утро время мастер город лист "
UA_WORDS = "поле рiчка лiс вiтер камiннь дорога тiнь дах пошта море пiсня ранок час майстер мiсто лист "

# raw special tokens whose presence in a reply means contamination
RAW_TOKENS = (" Скачать", " слив", "tool_call", "<|", "|>", "<start>", "<end>")


@lru_cache(maxsize=None)
def filler_text(depth_k: int, seed: int, tag: str) -> str:
    """Unique seeded Cyrillic filler (~depth_k thousand tokens) - no repeated boilerplate across jobs."""
    rnd = random.Random(f"{depth_k}|{seed}|{tag}")
    words = (RU_WORDS * 3 + UA_WORDS * 3).split()
    n_words = int(depth_k * 1000 * 0.476)  # calibrated: cyrillic ~2.1 chars/token
    out = []
    while len(out) < n_words:
        out.append(rnd.choice(words))
    return "\n".join(
        " ".join(out[i:i + 9]).capitalize() + "." for i in range(0, len(out), 9)
    )


def build_prompt(kind: str, lang: str, idx: int, depth_k: int, seed: int):
    """Return (prompt, needle). The needle sits mid-filler; other tasks sit AFTER the filler."""
    lang = lang.upper()
    needle = None
    if kind == "code":
        base = (CODE_RU if lang == "RU" else CODE_UA)[idx % 6]
    elif kind == "fact":
        base = (FACT_RU if lang == "RU" else FACT_UA)[idx % 4][0]
    else:  # needle
        needle = NEEDLE_TOKENS[idx % 4]
        lead = NEEDLE_RU if lang == "RU" else NEEDLE_UA
        if lang == "UA":
            base = lead + "Текст: [ДАННЫЕ]\nПитання: який секретний код був у текстi? Вiдповiдь одним реченням, назви код."
        else:
            base = lead + "Текст: [ДАННЫЕ]\nВопрос: какой секретный код был в тексте? Ответь одним предложением, назови код."
    fill = filler_text(depth_k, seed, f"{kind}{idx}")
    if needle:
        pos = int(len(fill) * 0.45)
        needle_line = f"\nСекретный код: {needle}\n" if lang == "RU" else f"\nСекретний код: {needle}\n"
        fill = fill[:pos] + needle_line + fill[pos:]
        prompt = base.replace("[ДАННЫЕ]", fill)
    else:
        prompt = fill + "\n\n" + base
    return prompt, needle


# ---------------------------------------------------------------- checkers
def extract_code(text: str) -> str:
    m = re.search(r"```(?:python|sql)?\s*\n(.*?)```", text, re.S)
    return m.group(1).strip() if m else text.strip()


def check_code(idx: int, text: str) -> bool:
    ns: dict = {}
    try:
        exec(compile(extract_code(text), "<c>", "exec"), ns)  # noqa: S102 - test harness
        if idx == 0:
            import heapq  # noqa: F401 - prove importability inside the snippet's env
            return list(ns["merge_sorted"]([[1, 4, 5], [1, 3, 4], [2, 6]])) == [1, 1, 2, 3, 4, 4, 5, 6]
        if idx == 1:
            cls = [v for k, v in ns.items() if k.lower().startswith("lru")][:1] \
                  or [v for v in ns.values() if isinstance(v, type)]
            if not cls:
                return False
            lru = cls[0](2)
            lru.put(1, 1); lru.put(2, 2); lru.get(1); lru.put(3, 3)
            return lru.get(2) == -1 and lru.get(1) == 1
        if idx == 2:
            f = ns["to_roman"]
            return f(9) == "IX" and f(1994) == "MCMXCIV" and f(3999) == "MMMCMXCIX"
        if idx == 3:
            f = ns["balanced"]
            return f("([{}])") is True and f("([)]") is False and f("(((") is False
        if idx == 4:
            f = ns["topo_sort"]
            r = f(3, [(0, 1), (1, 2)])
            return r is not None and (r == [0, 1, 2] or set(r) == {0, 1, 2}) and f(2, [(0, 1), (1, 0)]) is None
        if idx == 5:
            return ns["spiral"]([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]
    except Exception:
        return False
    return False


def check_fact(idx: int, lang: str, text: str) -> bool:
    low = " " + text.lower() + " "
    return any(s in low for s in (FACT_RU if lang.upper() == "RU" else FACT_UA)[idx % 4][1])


def check_needle(needle: str | None, text: str) -> bool:
    return needle is not None and needle in text


def loop_repeats(text: str) -> int:
    """Max repeats of any 40+ char line - the loop-failure signature."""
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) >= 40]
    return max(Counter(lines).values()) if lines else 0


def check_contamination(text: str) -> list[str]:
    return [t for t in RAW_TOKENS if t in text]
