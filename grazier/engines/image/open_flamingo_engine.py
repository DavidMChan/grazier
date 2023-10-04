import logging
from typing import Any, List, Optional

import torch
from huggingface_hub import hf_hub_download
from PIL import Image

from grazier.engines.image import ILMEngine, register_engine
from grazier.utils.python import singleton

try:
    from open_flamingo import create_model_and_transforms
except ImportError:
    create_model_and_transforms = None


class OpenFlamingoILMEngine(ILMEngine):
    def __init__(
        self,
        checkpoint_str: str,
        vision_encoder: str,
        language_model: str,
        cross_attn_every_n_layers=1,
        device: Optional[str] = None,
        quantize: bool = False,
    ) -> None:
        super().__init__(device=device)

        if create_model_and_transforms is None:
            raise ImportError("OpenFlamingo is not installed. Please install it with `pip install open-flamingo`.")

        if quantize:
            logging.warning("Quantization is not supported for OpenFlamingo models.")

        self._model, self._image_processor, self._tokenizer = create_model_and_transforms(
            clip_vision_encoder_path=vision_encoder,
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=language_model,
            tokenizer_path=language_model,
            cross_attn_every_n_layers=cross_attn_every_n_layers,
        )
        self._model = self._model.to(self.device)

        checkpoint_path = hf_hub_download(checkpoint_str, "checkpoint.pt")
        self._model.load_state_dict(torch.load(checkpoint_path), strict=False)

        self._tokenizer.paddding_side = "left"

    def call(self, image: Image, prompt: Optional[str] = None, n_completions: int = 1, **kwargs: Any) -> List[str]:
        if prompt is None:
            prompt = "<image>A photo of"
        if "<image>" not in prompt:
            prompt = "<image>" + prompt
        if prompt.count("<image>") > 1:
            raise ValueError("Prompt must contain at most one <image> token.")

        processed_image = (
            torch.cat(
                [
                    self._image_processor(image).unsqueeze(0),
                ],
                dim=0,
            )
            .unsqueeze(1)
            .unsqueeze(0)
        )

        text_inputs = self._tokenizer([prompt], return_tensors="pt")

        # Move the inputs to the correct device
        processed_image = processed_image.to(self.device)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items() if isinstance(v, torch.Tensor)}

        outputs = self._model.generate(
            vision_x=processed_image,
            lang_x=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            max_new_tokens=kwargs.get("max_new_tokens", kwargs.pop("max_tokens", 128)),
            num_return_sequences=n_completions,
            do_sample=n_completions > 1,
            pad_token_id=self._tokenizer.pad_token_id,
            eos_token_id=self._tokenizer.eos_token_id,
            **kwargs,
        )

        # Remove the prompt from the output
        outputs = outputs[:, text_inputs["input_ids"].shape[-1] :]
        outputs = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return [o.strip() for o in outputs]


@register_engine
@singleton
class OpenFlamingo3B(OpenFlamingoILMEngine):
    name = ("Open-Flamingo (3B, ViT-L, MPT-1B)", "OpenFlamingo-3B-vitl-mpt1b")

    def __init__(self, device: Optional[str] = None, quantize: bool = False) -> None:
        super().__init__(
            checkpoint_str="openflamingo/OpenFlamingo-3B-vitl-mpt1b",
            vision_encoder="ViT-L-14",
            language_model="mosaicml/mpt-1b-redpajama-200b",
            device=device,
            quantize=quantize,
        )

    @staticmethod
    def is_configured() -> bool:
        return True


@register_engine
@singleton
class OpenFlamingo3BInstruct(OpenFlamingoILMEngine):
    name = ("Open-Flamingo (3B, ViT-L, MPT-1B-Dolly)", "OpenFlamingo-3B-vitl-mpt1b-dolly")

    def __init__(self, device: Optional[str] = None, quantize: bool = False) -> None:
        super().__init__(
            checkpoint_str="openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct",
            vision_encoder="ViT-L-14",
            language_model="mosaicml/mpt-1b-redpajama-200b-dolly",
            device=device,
            quantize=quantize,
        )

    @staticmethod
    def is_configured() -> bool:
        return True


@register_engine
@singleton
class OpenFlamingo9B(OpenFlamingoILMEngine):
    name = ("Open-Flamingo (9B, ViT-L, MPT-7B)", "OpenFlamingo-9B-vitl-mpt7b")

    def __init__(self, device: Optional[str] = None, quantize: bool = False) -> None:
        super().__init__(
            checkpoint_str="openflamingo/OpenFlamingo-9B-vitl-mpt7b",
            vision_encoder="ViT-L-14",
            language_model="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=4,
            device=device,
            quantize=quantize,
        )

    @staticmethod
    def is_configured() -> bool:
        return True


@register_engine
@singleton
class OpenFlamingo4B(OpenFlamingoILMEngine):
    name = ("Open-Flamingo (4B, ViT-L, INCITE-Base-3B)", "OpenFlamingo-4B-vitl-rpj3b")

    def __init__(self, device: Optional[str] = None, quantize: bool = False) -> None:
        super().__init__(
            checkpoint_str="openflamingo/OpenFlamingo-4B-vitl-rpj3b",
            vision_encoder="ViT-L-14",
            language_model="atogethercomputer/RedPajama-INCITE-Base-3B-v1",
            cross_attn_every_n_layers=2,
            device=device,
            quantize=quantize,
        )

    @staticmethod
    def is_configured() -> bool:
        return True


@register_engine
@singleton
class OpenFlamingo4BInstruct(OpenFlamingoILMEngine):
    name = ("Open-Flamingo (4B, ViT-L, INCITE-Instruct-3B)", "OpenFlamingo-4B-vitl-rpj3b")

    def __init__(self, device: Optional[str] = None, quantize: bool = False) -> None:
        super().__init__(
            checkpoint_str="openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct",
            vision_encoder="ViT-L-14",
            language_model="togethercomputer/RedPajama-INCITE-Instruct-3B-v1",
            cross_attn_every_n_layers=2,
            device=device,
            quantize=quantize,
        )

    @staticmethod
    def is_configured() -> bool:
        return True
