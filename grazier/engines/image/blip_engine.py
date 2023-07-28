from typing import Any, List, Optional

import torch
from PIL import Image
from transformers import BitsAndBytesConfig, Blip2ForConditionalGeneration, Blip2Processor

from grazier.engines.image import ILMEngine, register_engine
from grazier.utils.python import singleton
from grazier.utils.pytorch import select_device


class Blip2ILMEngine(ILMEngine):
    def __init__(self, model: str, device: Optional[str] = None, quantize: bool = False) -> None:
        super().__init__(device=device)

        # Setup quantization
        if quantize:
            nf4_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            nf4_config = None

        self._model = Blip2ForConditionalGeneration.from_pretrained(
            model,
            device_map="auto" if not self.device_specified else None,
            torch_dtype=torch.float16 if quantize else None,
            quantization_config=nf4_config,
        )
        if self.device_specified:
            self._model.to(self.device)
        self._processor = Blip2Processor.from_pretrained(
            model,
            # device=self.device if self.device_specified else None,
            device_map="auto" if not self.device_specified else None,
            torch_dtype=torch.float16 if quantize else None,
        )

    def call(self, image: Image, prompt: Optional[str] = None, n_completions: int = 1, **kwargs: Any) -> List[str]:
        image = image.convert("RGB")
        if prompt is None:
            prompt = "An photo of "

        inputs = self._processor(image, prompt, return_tensors="pt")

        # Convert the inputs to the correct device
        if self.device_specified:
            inputs = {k: v.to(self.device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}

        outputs = self._model.generate(
            **inputs,
            max_new_tokens=kwargs.get("max_new_tokens", kwargs.pop("max_tokens", 128)),
            num_return_sequences=n_completions,
            do_sample=n_completions > 1,
            **kwargs,
        )

        outputs = self._processor.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return outputs


@register_engine
@singleton
class Blip2OPT27B(Blip2ILMEngine):
    name = ("Blip2 (OPT 2.7B)", "blip2-opt-2.7b")

    def __init__(self, device: Optional[str] = None, quantize: bool = False) -> None:
        super().__init__("Salesforce/blip2-opt-2.7b", device=device, quantize=quantize)

    @staticmethod
    def is_configured() -> bool:
        return True


@register_engine
@singleton
class Blip2OPT67B(Blip2ILMEngine):
    name = ("Blip2 (OPT 6.7B)", "blip2-opt-6.7b")

    def __init__(self, device: Optional[str] = None, quantize: bool = False) -> None:
        super().__init__("Salesforce/blip2-opt-6.7b", device=device, quantize=quantize)

    @staticmethod
    def is_configured() -> bool:
        return True


@register_engine
@singleton
class Blip2OPT27BCoco(Blip2ILMEngine):
    name = ("Blip2 (OPT 2.7B, Coco Fine-tuned)", "blip2-opt-2.7b-coco")

    def __init__(self, device: Optional[str] = None, quantize: bool = False) -> None:
        super().__init__("Salesforce/blip2-opt-2.7b-coco", device=device, quantize=quantize)

    @staticmethod
    def is_configured() -> bool:
        return True


@register_engine
@singleton
class Blip2OPT67BCoco(Blip2ILMEngine):
    name = ("Blip2 (OPT 6.7B, Coco Fine-tuned)", "blip2-opt-6.7b-coco")

    def __init__(self, device: Optional[str] = None, quantize: bool = False) -> None:
        super().__init__("Salesforce/blip2-opt-6.7b-coco", device=device, quantize=quantize)

    @staticmethod
    def is_configured() -> bool:
        return True


@register_engine
@singleton
class Blip2FlanT5XL(Blip2ILMEngine):
    name = ("Blip2 (Flan-T5 XL)", "blip2-flan-t5-xl")

    def __init__(self, device: Optional[str] = None, quantize: bool = False) -> None:
        # This forces a single device to be used, since there's a bug in Blip2 + Flan-T5, when the model is split across
        # multiple devices.
        device = select_device(device)
        super().__init__("Salesforce/blip2-flan-t5-xl", device=device, quantize=quantize)

    @staticmethod
    def is_configured() -> bool:
        return True


@register_engine
@singleton
class Blip2FlanT5XXL(Blip2ILMEngine):
    name = ("Blip2 (Flan-T5 XXL)", "blip2-flan-t5-xxl")

    def __init__(self, device: Optional[str] = None, quantize: bool = False) -> None:
        super().__init__("Salesforce/blip2-flan-t5-xxl", device=device, quantize=quantize)

    @staticmethod
    def is_configured() -> bool:
        return True


@register_engine
@singleton
class Blip2FlanT5XLCoco(Blip2ILMEngine):
    name = ("Blip2 (Flan-T5 XL, Coco Fine-tuned)", "blip2-flan-t5-xl-coco")

    def __init__(self, device: Optional[str] = None, quantize: bool = False) -> None:
        super().__init__("Salesforce/blip2-flan-t5-xl-coco", device=device, quantize=quantize)

    @staticmethod
    def is_configured() -> bool:
        return True
