"""Thin OpenAI-compatible chat-completions client (stdlib only)."""
from __future__ import annotations

import json
import urllib.request


def chat(base_url: str, body: dict, model: str = "", timeout: int = 1800) -> dict:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    if model:
        body["model"] = model
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def body_for(messages: list[dict], *, max_tokens: int, temperature: float,
             seed: int, thinking: bool = False) -> dict:
    body: dict = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seed": seed,
    }
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body
