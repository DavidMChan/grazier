import logging
import random

import pytest

from grazier.engines.llm import LLMEngine


@pytest.mark.parametrize(
    "engine",
    [
        "llama-7B",
        "llama-13B",
        "llama-30B",
        "llama-60B",
    ],
)
def test_huggingface_llm_engine(engine: str) -> None:
    if engine in (
        "llama-13B",
        "llama-30B",
        "llama-60B",
    ):
        pytest.skip("Model too large for CI")

    _engine = LLMEngine.from_string(engine)
    random_number = random.randint(0, 100)
    response = _engine(f"My name, followed by a colon with the number {random_number} is:")

    for r in response:
        assert len(r.strip()) > 0, f"Response is empty: {r}"
        if str(random_number) not in r:
            logging.warning(f'Number "{random_number}" not found in response "{r}"')
