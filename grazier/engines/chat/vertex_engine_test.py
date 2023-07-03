import logging

import pytest

from grazier.engines.chat import Conversation, LLMChat, Speaker


@pytest.mark.parametrize("engine", ["palm"])
def test_vertex_llm_engine(engine: str) -> None:
    # Construct a conversation
    conversation = Conversation()
    conversation.add_turn("You are an intelligent AI named Jason.", speaker=Speaker.SYSTEM)
    conversation.add_turn("Your name, followed by a colon with the number 42 is:", speaker=Speaker.USER)

    _engine = LLMChat.from_string(engine)
    responses = _engine(conversation)
    for r in responses:
        assert r.text.strip() != ""
        if "Jason" not in r.text:
            logging.warning(f'Name "Jason" not found in response "{r.text}"')
        if "42" not in r.text:
            logging.warning(f'Number "42" not found in response "{r.text}"')
