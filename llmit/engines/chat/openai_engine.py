
import os
from abc import abstractmethod
from typing import Any, List

import openai

from llmit.engines.chat import Conversation, ConversationTurn, LLMChat, Speaker, register_engine
from llmit.engines.llm.openai_engine import OpenAI
from llmit.utils.python import retry, singleton

# Setup openai api keys
openai.organization = os.getenv("OPENAI_API_ORG", None)
openai.api_key = os.getenv("OPENAI_API_KEY", None)

class OpenAIChatEngine(LLMChat):
    def __init__(self, model: str):
        super().__init__(device="api")
        self._model = model

    @property
    @abstractmethod
    def cost_per_token(self) -> float:
        raise NotImplementedError()

    @retry(
        no_retry_on=(
            openai.error.AuthenticationError,
        )
    )
    def _retry_call(self, *args, **kwargs):
        return openai.ChatCompletion.create(*args, **kwargs)


    def call(
        self,
        conversation: Conversation,
        n_completions: int = 1,
        **kwargs: Any,
    ) -> List[ConversationTurn]:

        # Construct the messages list from the conversation
        messages = []
        for turn in conversation.turns:
            messages.append({
                "role": "user" if turn.speaker == Speaker.USER else "system" if turn.speaker == Speaker.SYSTEM else "assistant" if turn.speaker == Speaker.AI else "user",
                "content": turn.text,
            })

        # Call the OpenAI API
        cp = self._retry_call(
                model=self._model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.9),
                max_tokens=kwargs.get("max_tokens", 150),
                n=n_completions,
                **kwargs,
        )  # type: ignore
        OpenAI.USAGE += int(cp.usage.total_tokens) * self.cost_per_token

        return [
            ConversationTurn(
                text=i.message.content,
                speaker=Speaker.USER if i.message.role == "user" else Speaker.SYSTEM if i.message.role == "system" else Speaker.AI if i.message.role == "assistant" else Speaker.USER,
            )
            for i in cp.choices
        ]

@register_engine
@singleton
class ChatGPT(OpenAIChatEngine):
    name = ("Chat GPT", "chat-gpt")
    cost_per_token = 0.002 / 1000

    def __init__(self) -> None:
        super().__init__("gpt-3.5-turbo")

@register_engine
@singleton
class GPT4(OpenAIChatEngine):
    name = ("GPT-4", "gpt4")
    cost_per_token = 0.03 / 1000

    def __init__(self) -> None:
        super().__init__("gpt-4")

@register_engine
@singleton
class GPT432K(OpenAIChatEngine):
    name = ("GPT-4 32K", "gpt4-32k")
    cost_per_token = 0.06 / 1000

    def __init__(self) -> None:
        super().__init__("gpt-4-32k")
