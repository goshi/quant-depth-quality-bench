"""Configuration loading. TOML via stdlib tomllib (Python >= 3.11)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ServerConfig:
    base_url: str = "http://127.0.0.1:8002"
    model: str = ""  # some servers (e.g. vLLM) require an explicit model id; empty = omit
    request_timeout: int = 1800


@dataclass
class RunConfig:
    depths_k: list[int] = field(default_factory=lambda: [8, 64, 192, 384, 480])
    seeds: list[int] = field(default_factory=lambda: [1000, 1001])
    languages: list[str] = field(default_factory=lambda: ["ru", "ua"])
    workers: int = 3  # concurrent requests; keep within the server's KV-cache capacity
    max_tokens: int = 1024
    temperature: float = 0.7
    thinking: bool = False  # False => chat_template_kwargs {"enable_thinking": False}
    out_dir: str = "results"


@dataclass
class ReducedSetConfig:
    enabled: bool = True
    depths_k: list[int] = field(default_factory=lambda: [480])
    code_tasks: int = 3  # of 6
    fact_tasks: int = 1  # of 4
    needle_tasks: int = 1  # of 2


@dataclass
class RetryConfig:
    max_tokens: int = 4096


@dataclass
class Config:
    server: ServerConfig
    run: RunConfig
    reduced_set: ReducedSetConfig
    retry: RetryConfig


def load(path: str | Path) -> Config:
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    server = ServerConfig(**raw.get("server", {}))
    run_raw = dict(raw.get("run", {}))
    run = RunConfig(**{k: v for k, v in run_raw.items() if k in RunConfig.__dataclass_fields__})
    reduced = ReducedSetConfig(**raw.get("reduced_set", {}))
    retry = RetryConfig(**raw.get("retry", {}))
    return Config(server=server, run=run, reduced_set=reduced, retry=retry)
