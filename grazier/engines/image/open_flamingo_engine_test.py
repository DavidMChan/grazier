import logging

import pytest
from PIL import Image

import grazier


@pytest.mark.parametrize(
    "engine",
    [
        "OpenFlamingo-3B-vitl-mpt1b",
    ],
)
def test_open_flamingo_ilm_engine(engine: str) -> None:
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
    test_open_flamingo_ilm_engine("OpenFlamingo-9B-vitl-mpt7b")
