import os
from typing import Any, List, Optional

from transformers import LlamaForCausalLM, LlamaTokenizer

from grazier.engines.llm import LLMEngine, register_engine
from grazier.utils.python import singleton


class HuggingFaceLlamaLMEngine(LLMEngine):
    def __init__(
        self,
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

    def call(self, prompt: str, n_completions: int = 1, **kwargs: Any) -> List[str]:
        input = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self._generator.device)

        # Handle the kwargs
        _params = {
            "max_new_tokens": kwargs.get("max_new_tokens", kwargs.pop("max_tokens", self._max_new_tokens)),
            "num_return_sequences": n_completions,
            "temperature": kwargs.get("temperature", None),
        } | kwargs

        outputs = self._generator.generate(
            input,
            do_sample=n_completions > 1,
            **_params,
        )

        # Strip the prompt from the output
        outputs = outputs[:, input.shape[-1] :]
        # Decode and return the output
        outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return outputs


@register_engine
@singleton
class Llama7B(HuggingFaceLlamaLMEngine):
    name = ("LLAMA 7B", "llama-7B")

    def __init__(self, device: Optional[str] = "defer") -> None:
        super().__init__("7B", "LLAMA_WEIGHTS_ROOT", device=device)


@register_engine
@singleton
class Llama13B(HuggingFaceLlamaLMEngine):
    name = ("LLAMA 13B", "llama-13B")

    def __init__(self, device: Optional[str] = "defer") -> None:
        super().__init__("13B", "LLAMA_WEIGHTS_ROOT", device=device)


@register_engine
@singleton
class Llama30B(HuggingFaceLlamaLMEngine):
    name = ("LLAMA 30B", "llama-30B")

    def __init__(self, device: Optional[str] = "defer") -> None:
        super().__init__("30B", "LLAMA_WEIGHTS_ROOT", device=device)


@register_engine
@singleton
class Llama65B(HuggingFaceLlamaLMEngine):
    name = ("LLAMA 65B", "llama-65B")

    def __init__(self, device: Optional[str] = "defer") -> None:
        super().__init__("65B", "LLAMA_WEIGHTS_ROOT", device=device)
