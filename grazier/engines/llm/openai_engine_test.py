import logging
import os
import random

import pytest

from grazier.engines.llm import LLMEngine


@pytest.mark.parametrize("engine", ["gpt3-davinci3", "gpt3-davinci2", "gpt3-curie", "gpt3-babbage", "gpt3-ada"])
def test_openai_llm_engine(engine: str) -> None:
    if not os.getenv("OPENAI_API_ORG", None):
        pytest.skip("OPENAI_API_ORG not set")
    if not os.getenv("OPENAI_API_KEY", None):
        pytest.skip("OPENAI_API_KEY not set")

    _engine = LLMEngine.from_string(engine)
    random_number = random.randint(0, 100)
    response = _engine(f"My name, followed by a colon with the number {random_number} is:")
    for r in response:
        assert len(r.strip()) > 0, f"Response is empty: {r}"
        if str(random_number) not in r:
            logging.warning(f'Number "{random_number}" not found in response "{r}"')
