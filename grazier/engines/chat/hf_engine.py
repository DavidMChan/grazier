from typing import Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from grazier.engines.chat import Conversation, ConversationTurn, LLMChat, Speaker, register_engine
from grazier.utils.huggingface import check_huggingface_model_files_are_local
from grazier.utils.python import singleton
from grazier.utils.pytorch import select_device


class HuggingfaceChatEngine(LLMChat):
    def __init__(self, model: str, device: Optional[str] = None):
        device = select_device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model)
        self._model.half().to(device)
        self._device = device

    def call(self, conversation: Conversation, n_completions: int = 1, **kwargs: Any) -> List[ConversationTurn]:
        # Build the prompt
        chat = conversation.to_huggingface_chat()
        self._tokenizer.use_default_system_prompt = kwargs.get("use_default_system_prompt", False)
        inputs = self._tokenizer.apply_chat_template(chat, return_tensors="pt").to(self._device)

        # Get the params, and outputs
        _params = {
            "max_new_tokens": kwargs.get("max_new_tokens", kwargs.pop("max_tokens", 256)),
            "min_new_tokens": kwargs.get("min_new_tokens", 10),
            "num_return_sequences": n_completions,
            "temperature": kwargs.get("temperature", None),
            # "return_full_text": False,
        } | kwargs

        tokens = self._model.generate(
            inputs,
            do_sample=n_completions > 1,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            **_params,
        )

        last_turn_is_user = conversation.turns[-1].speaker == Speaker.USER
        outputs = [self._decode_tokens(t[inputs.shape[-1] :], last_turn_is_user) for t in tokens]  # type: ignore

        return [
            ConversationTurn(text=output, speaker=Speaker.AI if last_turn_is_user else Speaker.USER)
            for output in outputs
        ]

    def _decode_tokens(self, tokens: torch.Tensor, last_turn_is_user: bool) -> str:
        output = self._tokenizer.decode(tokens, skip_special_tokens=True).strip()
        return output

    @staticmethod
    def requires_configuration() -> bool:
        return False

    @classmethod
    def configure(cls, *args, **kwargs) -> None:
        pass


@register_engine
@singleton
class MistralInstruct7B(HuggingfaceChatEngine):
    name = ("Mistral Instruct (7B)", "mistral-instruct-7b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("mistralai/Mistral-7B-Instruct-v0.1", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("mistralai/Mistral-7B-Instruct-v0.1")
