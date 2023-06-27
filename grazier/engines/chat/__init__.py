from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

from grazier.engines.llm import register_engine as register_llm_engine
from grazier.utils.pytorch import select_device


class Speaker(Enum):
    USER = 'user'
    AI = 'ai'
    SYSTEM = 'system'

@dataclass
class ConversationTurn:
    text: str
    speaker: Speaker


class Conversation:
    def __init__(self, turns: Optional[List[Union[str, ConversationTurn]]] = None):
        if turns is not None:
            # Convert strings to ConversationTurns
            self.turns = [ConversationTurn(text=t, speaker=Speaker.USER) if isinstance(t, str) else t for t in turns]
        else:
            self.turns = []

    def add_turn(self, text: Union[str, ConversationTurn], speaker: Optional[Speaker] = None) -> None:
        if isinstance(text, str):
            self.turns.append(ConversationTurn(text=text, speaker=speaker or Speaker.USER))
        else:
            self.turns.append(ConversationTurn(text=text.text, speaker=speaker or text.speaker))

    def __repr__(self) -> str:
        # TODO: Make this more readable
        return f"Conversation({self.turns})"


class LLMChat(ABC):

    @property
    @abstractmethod
    def name(self) -> Tuple[str, str]:
        """ Returns a tuple of (Pretty Name, CLI name) of the language model. """
        raise NotImplementedError()

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = select_device(device)

    def __call__(
        self,
        prompt: Union[Conversation, ConversationTurn, str],
        n_completions: int = 1,
        **kwargs: Any
    ) -> List[ConversationTurn]:
        conversation = None
        if isinstance(prompt, str):
            conversation = Conversation()
            conversation.add_turn(prompt, speaker=Speaker.USER)
        elif isinstance(prompt, ConversationTurn):
            conversation = Conversation()
            conversation.add_turn(prompt)
        else:
            assert isinstance(prompt, Conversation)
            conversation = prompt

        return self.call(conversation, n_completions=n_completions, **kwargs)


    @abstractmethod
    def call(
        self,
        conversation: Conversation,
        n_completions: int = 1,
        **kwargs: Any
    ) -> List[ConversationTurn]:
        raise NotImplementedError()

    def __repr__(
        self,
    ) -> str:
        return f"{self.__class__.__name__}({self.name[0]})"


    @staticmethod
    def from_string(typestr: str, **kwargs: Any) -> "LLMChat":

        typestr = typestr.lower()
        if typestr in LM_CHAT_ENGINES:
            return LM_CHAT_ENGINES[typestr](**kwargs)  # type: ignore
        elif typestr in LM_CHAT_ENGINES_CLI:
            return LM_CHAT_ENGINES_CLI[typestr](**kwargs)  # type: ignore

        raise ValueError(f"Invalid language model type: {typestr}. Valid types are: {list(LM_CHAT_ENGINES_CLI.keys())}")

    @staticmethod
    def list_models() -> List[str]:
        return list(LM_CHAT_ENGINES_CLI.keys())

LM_CHAT_ENGINES: Dict[str, Type[LLMChat]] = {}
LM_CHAT_ENGINES_CLI: Dict[str, Type[LLMChat]] = {}

T = TypeVar("T")
def register_engine(cls: T) -> T:
    from grazier.engines.llm.chat_engine import wrap_chat_llm_engine
    LM_CHAT_ENGINES[cls.name[0].lower()] = cls # type: ignore
    LM_CHAT_ENGINES_CLI[cls.name[1].lower()] = cls # type: ignore

    # Register the engine as an LLM Engine as well
    register_llm_engine(wrap_chat_llm_engine(cls))

    return cls


from grazier.engines.chat.anthropic_engine import *  # noqa: F403, E402
from grazier.engines.chat.bard_engine import *  # noqa: F403, E402
from grazier.engines.chat.llama_engine import *  # noqa: F403, E402
from grazier.engines.chat.openai_engine import *  # noqa: F403, E402
from grazier.engines.chat.stable_lm_engine import *  # noqa: F403, E402
from grazier.engines.chat.vertex_engine import *  # noqa: F403, E402
