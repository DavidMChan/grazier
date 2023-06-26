
import logging
import os
from typing import Any, List, Optional, Tuple

from transformers import LlamaForCausalLM, LlamaTokenizer

from llmit.engines.chat import Conversation, ConversationTurn, LLMChat, Speaker, register_engine
from llmit.utils.python import singleton


class LlamaLMEngine(LLMChat):
    def __init__(self,
                model: str,
                weight_root: str,
                max_new_tokens: int = 128,
                device: Optional[str] = None,
    ) -> None:
        super().__init__(device=device)
        self._max_new_tokens = max_new_tokens
        self.tokenizer = LlamaTokenizer.from_pretrained(f"{os.environ.get(weight_root, '')}{model}")
        self._generator = LlamaForCausalLM.from_pretrained(
            f"{os.environ.get(weight_root, '')}{model}", device_map="auto"
        )

    def _build_prompt_from_conversation(self, conversation: Conversation) -> Tuple[str, bool]:
        raise NotImplementedError()

    def _extract_last_turn(self, outputs: List[str], last_prompt_is_user: bool) -> List[str]:
        raise NotImplementedError()


    def call(
        self, conversation: Conversation, n_completions: int = 1, **kwargs: Any
    ) -> List[ConversationTurn]:

        prompt, last_prompt_is_user = self._build_prompt_from_conversation(conversation)

        input = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self._generator.device)
        temperature = kwargs.get("temperature", None)

        print(prompt)

        # Select the generation process based on the temperature
        if temperature is not None and temperature > 0:
            outputs = self._generator.generate(
                input,
                max_new_tokens=self._max_new_tokens,
                num_return_sequences=n_completions,
                temperature=temperature,
                do_sample=True,
            )
        elif n_completions > 1:
            outputs = self._generator.generate(
                input,
                max_new_tokens=self._max_new_tokens,
                num_return_sequences=n_completions,
                do_sample=True,
            )
        else:
            outputs = self._generator.generate(
                input,
                max_new_tokens=self._max_new_tokens,
                num_return_sequences=n_completions,
                do_sample=False,
            )

        # Strip the prompt from the output
        outputs_filtered = outputs[:, input.shape[-1]:]
        # Decode and return the output
        outputs_filtered = self.tokenizer.batch_decode(outputs_filtered, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        outputs_no_filter = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        print(outputs_no_filter)

        # Pick out the last continued turn
        outputs = self._extract_last_turn(outputs_filtered, last_prompt_is_user)

        return [ConversationTurn(text=o, speaker=Speaker.AI if last_prompt_is_user else Speaker.USER) for o in outputs]



class KoalaLMEngine(LlamaLMEngine):

    def _build_prompt_from_conversation(self, conversation: Conversation) -> Tuple[str, bool]:
        # Step 1: Build the prompt from the conversation
        prompt = "BEGINNING OF CONVERSATION: "
        for turn in conversation.turns:
            if turn.speaker in (Speaker.USER, Speaker.SYSTEM):
                prompt += "USER: "
                prompt += turn.text + " "
            elif turn.speaker == Speaker.AI:
                prompt += "GPT: "
                prompt += turn.text + " "
                prompt += '</s>'

        # Append the next prompt
        last_prompt_is_user = True
        if conversation.turns[-1].speaker in (Speaker.USER, Speaker.SYSTEM):
            prompt += "GPT: "
            last_prompt_is_user = True
        else:
            logging.warning("The last turn in the conversation is from the AI, however this model does not (usually) support continuing conversation from the user perspective.")
            prompt += "USER: "
            last_prompt_is_user = False

        return prompt, last_prompt_is_user

    def _extract_last_turn(self, outputs: List[str], last_prompt_is_user: bool) -> List[str]:
        return [o.split("GPT:" if last_prompt_is_user else "USER:")[0].strip() for o in outputs]

class Vicuna11LMEngine(LlamaLMEngine):

    def _build_prompt_from_conversation(self, conversation: Conversation) -> Tuple[str, bool]:
        # Step 1: Build the prompt from the conversation
        prompt = ""
        for idx, turn in enumerate(conversation.turns):
            if turn.speaker == Speaker.SYSTEM:
                if idx == 0:
                    prompt += f"{turn.text} "
                else:
                    logging.warning("Tried to add a system turn to the conversation at index != 0. This is not supported by the Vicuna 1.1 style model.")

            if turn.speaker == Speaker.USER:
                prompt += "USER: "
                prompt += turn.text + " "
            elif turn.speaker == Speaker.AI:
                prompt += "ASSISTANT: "
                prompt += turn.text + " "
                prompt += '</s>'

        # Append the next prompt
        last_prompt_is_user = True
        if conversation.turns[-1].speaker in (Speaker.USER, Speaker.SYSTEM):
            prompt += "ASSISTANT: "
            last_prompt_is_user = True
        else:
            logging.warning("The last turn in the conversation is from the AI, however this model does not (usually) support continuing conversation from the user perspective.")
            prompt += "USER: "
            last_prompt_is_user = False

        return prompt, last_prompt_is_user

    def _extract_last_turn(self, outputs: List[str], last_prompt_is_user: bool) -> List[str]:
        return [o.split("ASSISTANT:" if last_prompt_is_user else "USER:")[0].strip() for o in outputs]

class AlpacaLMEngine(LlamaLMEngine):

    def _build_prompt_from_conversation(self, conversation: Conversation) -> Tuple[str, bool]:
        # Step 1: Build the prompt from the conversation
        prompt = ""
        for idx, turn in enumerate(conversation.turns):
            if turn.speaker == Speaker.SYSTEM:
                if idx == 0:
                    prompt += f"{turn.text} "
                else:
                    logging.warning("Tried to add a system turn to the conversation at index != 0. This is not supported an Alpaca style model.")

            if turn.speaker == Speaker.USER:
                prompt += "### Instruction:\n\n "
                prompt += turn.text + " "
            elif turn.speaker == Speaker.AI:
                prompt += "### Response:\n\n "
                prompt += turn.text + " "
                prompt += '</s>'

        # Append the next prompt
        last_prompt_is_user = True
        if conversation.turns[-1].speaker in (Speaker.USER, Speaker.SYSTEM):
            prompt += "### Response:\n\n "
            last_prompt_is_user = True
        else:
            logging.warning("The last turn in the conversation is from the AI, however this model does not (usually) support continuing conversation from the user perspective.")
            prompt += "### Instruction:\n\n "
            last_prompt_is_user = False

        return prompt, last_prompt_is_user

    def _extract_last_turn(self, outputs: List[str], last_prompt_is_user: bool) -> List[str]:
        return [o.split("### Response:" if last_prompt_is_user else "### Instruction:")[0].strip() for o in outputs]


@register_engine
@singleton
class Koala7B(KoalaLMEngine):
    name = ("Koala (7B)", "koala-7B")
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("koala_7B", "KOALA_WEIGHTS_ROOT", device=device)

@register_engine
@singleton
class Koala13BV1(KoalaLMEngine):
    name = ("Koala (13B, v1)", "koala-13B-v1")
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("koala_13B_v1", "KOALA_WEIGHTS_ROOT", device=device)

@register_engine
@singleton
class Koala13BV2(KoalaLMEngine):
    name = ("Koala (13B, v2)", "koala-13B-v2")
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("koala_13B_v2", "KOALA_WEIGHTS_ROOT", device=device)

@register_engine
@singleton
class Vicuna7B(Vicuna11LMEngine):
    name = ("Vicuna (7B)", "vicuna-7B")
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("vicuna_7B", "VICUNA_WEIGHTS_ROOT", device=device)

@register_engine
@singleton
class Vicuna13B(Vicuna11LMEngine):
    name = ("Vicuna (13B)", "vicuna-13B")
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("vicuna_13B", "VICUNA_WEIGHTS_ROOT", device=device)


@register_engine
@singleton
class Alpaca13B(AlpacaLMEngine):
    name = ("Alpaca (13B)", "alpaca-13B")
    def __init__(self, device: Optional[str] = None) -> None:
        super().__init__("alpaca_13B", "ALPACA_WEIGHTS_ROOT", device=device)
