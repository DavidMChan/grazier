import logging
from abc import abstractmethod
from typing import Any, List

import openai
from rich.prompt import Prompt

from grazier.engines.chat import Conversation, ConversationTurn, LLMChat, Speaker, register_engine
from grazier.engines.llm.openai_engine import OpenAI
from grazier.utils.python import retry, singleton
from grazier.utils.secrets import get_secret, set_secret


def _setup_api_keys():
    # Setup openai api keys
    openai.api_key = get_secret("OPENAI_API_KEY", None)
    openai.organization = get_secret("OPENAI_API_ORG", None)


class OpenAIChatEngine(LLMChat):
    def __init__(self, model: str):
        super().__init__(device="api")

        _setup_api_keys()

        self._model = model

    @property
    @abstractmethod
    def cost_per_token(self) -> float:
        raise NotImplementedError()

    @retry(no_retry_on=(openai.error.AuthenticationError,))
    def _retry_call(self, *args: Any, **kwargs: Any) -> Any:
        return openai.ChatCompletion.create(*args, **kwargs)  # type: ignore

    def call(
        self,
        conversation: Conversation,
        n_completions: int = 1,
        **kwargs: Any,
    ) -> List[ConversationTurn]:
        # Construct the messages list from the conversation
        messages = []
        for turn in conversation.turns:
            messages.append(
                {
                    "role": "user"
                    if turn.speaker == Speaker.USER
                    else "system"
                    if turn.speaker == Speaker.SYSTEM
                    else "assistant"
                    if turn.speaker == Speaker.AI
                    else "user",
                    "content": turn.text,
                }
            )

        # Update temperature and max_tokens
        kwargs["temperature"] = kwargs.get("temperature", 0.9)
        kwargs["max_tokens"] = kwargs.get("max_tokens", 150)

        # Call the OpenAI API
        cp = self._retry_call(
            model=self._model,
            messages=messages,
            n=n_completions,
            **kwargs,
        )  # type: ignore
        OpenAI.USAGE += int(cp.usage.total_tokens) * self.cost_per_token

        return [
            ConversationTurn(
                text=i.message.content,
                speaker=Speaker.USER
                if i.message.role == "user"
                else Speaker.SYSTEM
                if i.message.role == "system"
                else Speaker.AI
                if i.message.role == "assistant"
                else Speaker.USER,
            )
            for i in cp.choices
        ]

    @staticmethod
    def is_configured() -> bool:
        _setup_api_keys()
        return openai.api_key is not None

    @staticmethod
    def configure():
        if OpenAIChatEngine.is_configured():
            reconfigure = Prompt.ask("OpenAI API key is already configured. Reconfigure?", choices=["y", "n"])
            if reconfigure == "n":
                logging.info("OpenAI API key already configured.")
                return

        have_openai_key = Prompt.ask("Do you have an OpenAI API key?", choices=["y", "n"])
        if have_openai_key == "n":
            Prompt.ask(
                "Please follow the instructions here [u cyan]https://www.howtogeek.com/885918/how-to-get-an-openai-api-key/[/u cyan] to obtain an API key.\nWhen you have the key, press enter to continue",
            )

        openai_key = Prompt.ask("Please enter your OpenAI API key (It will be saved to your platform's secrets store)")
        set_secret("OPENAI_API_KEY", openai_key.strip())

        # Save the openai api key to the secrets store
        logging.info("OpenAI API successfully configured.")


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
