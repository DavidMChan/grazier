import logging
from typing import Any, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from grazier.engines.chat import Conversation, ConversationTurn, LLMChat, Speaker, register_engine
from grazier.utils.huggingface import check_huggingface_model_files_are_local
from grazier.utils.python import singleton
from grazier.utils.pytorch import select_device


class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class StableLMChatEngine(LLMChat):
    SYSTEM_PROMPT = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human."""

    def __init__(self, model: str, device: Optional[str] = None):
        device = select_device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model)
        self._model.half().to(device)
        self._device = device

    def _decode_tokens(self, tokens: torch.Tensor, last_turn_is_user: bool) -> str:
        output = self._tokenizer.decode(tokens).strip()
        if last_turn_is_user:
            output, _, _ = output.partition("<|ASSISTANT|>")
        else:
            output, _, _ = output.partition("<|USER|>")

        output = (
            output.replace("<|USER|>", "")
            .replace("<|ASSISTANT|>", "")
            .replace("<|SYSTEM|>", "")
            .replace("<|endoftext|>", "")
            .strip()
        )
        return output

    def call(self, conversation: Conversation, n_completions: int = 1, **kwargs: Any) -> List[ConversationTurn]:
        # Build the prompt
        prompt = ""
        for idx, turn in enumerate(conversation.turns):
            if idx == 0:
                if turn.speaker != Speaker.SYSTEM:
                    # Use the default prompt if the first turn is not from the system
                    prompt += StableLMChatEngine.SYSTEM_PROMPT
                else:
                    prompt += turn.text
            elif turn.speaker == Speaker.USER:
                prompt += f"<|USER|>{turn.text}"
            elif turn.speaker == Speaker.AI:
                prompt += f"<|ASSISTANT|>{turn.text}"
            elif turn.speaker == Speaker.SYSTEM:
                logging.warning(
                    f"System turn detected at index {idx} in conversation {conversation}. This turn will be ignored."
                )

        # Add the beginning of the last turn
        last_turn_is_user = True
        if conversation.turns[-1].speaker == Speaker.AI:
            prompt += "<|USER|>"
            last_turn_is_user = False
        else:
            prompt += "<|ASSISTANT|>"
            last_turn_is_user = True

        # Actually build the completions
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        temperature = kwargs.get("temperature", None)

        if temperature is not None and temperature > 0:
            tokens = self._model.generate(
                **inputs,
                max_new_tokens=256,
                min_new_tokens=5,
                temperature=temperature,
                num_return_sequences=n_completions,
                do_sample=True,
                stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
            )
        else:
            tokens = self._model.generate(
                **inputs,
                max_new_tokens=256,
                min_new_tokens=5,
                num_return_sequences=n_completions,
                do_sample=False,
                stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
            )

        outputs = [self._decode_tokens(t[inputs["input_ids"].shape[-1] :], last_turn_is_user) for t in tokens]  # type: ignore
        return [
            ConversationTurn(text=output, speaker=Speaker.AI if last_turn_is_user else Speaker.USER)
            for output in outputs
        ]


@register_engine
@singleton
class StableLM3B(StableLMChatEngine):
    name = ("Stable LM (3B)", "stablelm-3b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("stabilityai/stablelm-tuned-alpha-3b", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("stabilityai/stablelm-tuned-alpha-3b")


@register_engine
@singleton
class StableLM7B(StableLMChatEngine):
    name = ("Stable LM (7B)", "stablelm-7b")

    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("stabilityai/stablelm-tuned-alpha-7b", device=device)

    @staticmethod
    def is_configured() -> bool:
        return check_huggingface_model_files_are_local("stabilityai/stablelm-tuned-alpha-7b")
