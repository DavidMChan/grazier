# isort: skip_file

from dotenv import load_dotenv
from typing import Union, List, Any
import logging

from rich.logging import RichHandler

__version__ = "0.0.2"
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Ignore import sorting, since LLMEngine needs to be imported first
from grazier.engines.llm import LLMEngine  # noqa: E402
from grazier.engines.chat import Conversation, LLMChat, Speaker, ConversationTurn  # noqa: F401, E402
from grazier.engines.image import ILMEngine  # noqa: E402


def get(name: str, chat: bool = False, type: str = "llm", **kwargs: Any) -> Union[LLMChat, LLMEngine]:
    """Get an engine by name."""
    name = name.lower()

    if chat:
        # Deprecated
        logging.warning("The `chat` argument is deprecated. Please use `type='chat'` instead.")
        type = "chat"

    if type == "chat":
        if name in LLMChat.list_models():
            return LLMChat.from_string(name, **kwargs)
        else:
            raise ValueError(f"Chat engine {name} not found")
    elif type == "image":
        if name in ILMEngine.list_models():
            return ILMEngine.from_string(name, **kwargs)
        else:
            raise ValueError(f"Image engine {name} not found")
    elif type == "llm":
        if name in LLMEngine.list_models():
            return LLMEngine.from_string(name, **kwargs)
        else:
            raise ValueError(f"LLM engine {name} not found")

    raise ValueError(f"Invalid engine type {type} (Select from 'chat', 'image', 'llm')")


def list_models(chat: bool = False, type: str = "llm") -> List[str]:
    """List available LLMs."""
    if chat:
        # Deprecated
        logging.warning("The `chat` argument is deprecated. Please use `type='chat'` instead.")
        type = "chat"

    if type == "chat":
        return LLMChat.list_models()
    elif type == "image":
        return ILMEngine.list_models()
    elif type == "llm":
        return LLMEngine.list_models()

    raise ValueError(f"Invalid engine type {type} (Select from 'chat', 'image', 'llm')")
