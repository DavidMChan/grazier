import logging

import pytest
from PIL import Image

import grazier


@pytest.mark.parametrize(
    "engine",
    [
        "blip2-opt-2.7b",
        "blip2-flan-t5-xl-coco",
    ],
)
def test_blip_ilm_engine(engine: str) -> None:
    _engine = grazier.get(engine, type="image")

    # Load the test image
    try:
        image = Image.open("test_data/dog.jpg")
    except FileNotFoundError:
        pytest.skip("Test image not found")

    # Generate a caption
    caption = _engine(image)

    # Check that the caption is not empty
    assert len(caption[0].strip()) > 0, f"Caption is empty: {caption}"

    # Check that the caption contains the word "llama"
    if "dog" not in caption[0].lower() and "puppy" not in caption[0].lower():
        logging.warning(f'Word "dog/puppy" not found in caption "{caption}"')

    print(caption)


if __name__ == "__main__":
    test_blip_ilm_engine("blip2-opt-2.7b")
