import logging
import os

from typing import Any, List


from grazier.engines.chat import Conversation, ConversationTurn, LLMChat, Speaker, register_engine
from grazier.utils.python import retry, singleton


@register_engine
@singleton
class BardEngine(LLMChat):
    name = ("Bard", "bard")

    def __init__(
        self,
    ) -> None:
        super().__init__(device="api")

        from Bard import Chatbot

        self._chatbot = Chatbot(
            os.environ.get("BARD__Secure_1PSID", None),
            os.environ.get("BARD__Secure_1PSIDTS", None),
        )

    @retry()
    def _ask(self, prompt: str) -> str:
        return self._chatbot.ask(prompt)["content"]

    def call(self, conversation: Conversation, n_completions: int = 1, **kwargs: Any) -> List[ConversationTurn]:
        if len(conversation.turns) != 1:
            raise AssertionError("BARD conversations must have exactly one turn.")

        # Extract the last user turn from the conversation.
        user_turn = conversation.turns[-1]
        new_prompt = user_turn.text
        if user_turn.speaker in (Speaker.SYSTEM, Speaker.AI):
            raise AssertionError("The last turn in the conversation must be from the user.")

        outputs = []
        for _ in range(n_completions):
            logging.info("Sending prompt to BARD engine: %s", new_prompt)
            outputs.append(self._ask(new_prompt))

        return [ConversationTurn(speaker=Speaker.AI, text=output) for output in outputs]

    @staticmethod
    def is_configured() -> bool:
        return (
            os.environ.get("BARD__Secure_1PSID", None) is not None
            and os.environ.get("BARD__Secure_1PSIDTS", None) is not None
        )
