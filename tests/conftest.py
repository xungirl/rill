"""Shared fixtures.

Unit tests use a tiny random-weight Qwen3-shaped model on CPU (fast, no
download). Tests marked `real_model` load Qwen/Qwen3-0.6B from the local HF
cache and are skipped automatically when it is absent (e.g. in CI).
"""

from __future__ import annotations

import pytest
import torch

from ember.config import ModelConfig, tiny_config
from ember.loader import random_model, resolve_model_dir
from ember.model import Qwen3Model


def _real_model_available() -> bool:
    try:
        resolve_model_dir()
    except FileNotFoundError:
        return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _real_model_available():
        return
    skip = pytest.mark.skip(reason="Qwen/Qwen3-0.6B not in local HF cache")
    for item in items:
        if "real_model" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _deterministic() -> None:
    torch.manual_seed(0)


@pytest.fixture(scope="session")
def tiny_cfg() -> ModelConfig:
    return tiny_config()


@pytest.fixture(scope="session")
def tiny_model(tiny_cfg: ModelConfig) -> Qwen3Model:
    return random_model(tiny_cfg, dtype=torch.float32, seed=0)
