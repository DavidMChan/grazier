import logging
import random

import pytest

from grazier.engines.llm import LLMEngine


@pytest.mark.parametrize("engine", ["j2-light", "j2-mid", "j2-ultra"])
def test_ai21_llm_engine(engine: str) -> None:
    _engine = LLMEngine.from_string(engine)
    random_number = random.randint(0, 100)
    response = _engine(f"My name, followed by a colon with the number {random_number} is:")

    for r in response:
        if len(r.strip()) == 0:
            pytest.skip(f"Empty response from {engine} engine")
        if str(random_number) not in r:
            logging.warning(f'Number "{random_number}" not found in response "{r}"')
