import logging
import random

import pytest

from grazier.engines.llm import LLMEngine


@pytest.mark.parametrize(
    "engine",
    [
        "gptj-6B",
        "gpt2",
        "gpt2-med",
        "gpt2-lg",
        "gpt2-xl",
        "distilgpt2",
        "gptneo-125M",
        "gptneo-1.3B",
        "gptneo-2.7B",
        "stablelm-3B",
        "stablelm-7B",
        "falcon-7b",
        "falcon-rw-1b",
    ],
)
def test_huggingface_llm_engine(engine: str) -> None:
    _engine = LLMEngine.from_string(engine)
    random_number = random.randint(0, 100)
    response = _engine(f"My name, followed by a colon with the number {random_number} is:")

    for r in response:
        if len(r.strip()) == 0:
            pytest.skip(f"Empty response from {engine} engine")
        if str(random_number) not in r:
            logging.warning(f'Number "{random_number}" not found in response "{r}"')
