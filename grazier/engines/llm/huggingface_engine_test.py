import logging
import random

import pytest
import torch

from grazier.engines.llm import LLMEngine


@pytest.mark.parametrize(
    "engine",
    ["gpt2", "stablelm-3B", "falcon-7b", "llama-2-7b", "mistral-7b"],
)
def test_huggingface_llm_engine(engine: str) -> None:
    try:
        _engine = LLMEngine.from_string(engine)
        random_number = random.randint(0, 100)
        response = _engine(f"My name, followed by a colon with the number {random_number} is:")

        for r in response:
            if len(r.strip()) == 0:
                pytest.skip(f"Empty response from {engine} engine")
            if str(random_number) not in r:
                logging.warning(f'Number "{random_number}" not found in response "{r}"')
    except torch.cuda.OutOfMemoryError:
        pytest.skip(f"Out of memory error from {engine} engine")
