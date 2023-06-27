
from typing import Any, List, Optional, Type

from grazier.engines.chat import Conversation, Speaker
from grazier.engines.llm import LLMEngine


def wrap_chat_llm_engine(cls) -> Type[LLMEngine]: # type: ignore
    """
    Wrap an LLMChat engine to make it compatible with the LLMEngine interface.

    Args:
        cls (Type[LLMChat]): The LLMChat engine to wrap.

    Returns:
        Type[LLMEngine]: The wrapped LLMEngine.
    """

    class _WrappedEngine(LLMEngine):

        name = cls.name

        def __init__(self, **kwargs: Any) -> None:
            super().__init__(device="defer")
            self._engine = cls(**kwargs)

        def call(
            self,
            prompt: str,
            n_completions: int = 1,
            **kwargs: Any
        ) -> List[str]:
            # Build a conversation
            conversation = Conversation()
            conversation.add_turn(prompt, speaker=Speaker.USER)
            return [x.text for x in self._engine.call(conversation, n_completions=n_completions, **kwargs)]


    return _WrappedEngine
