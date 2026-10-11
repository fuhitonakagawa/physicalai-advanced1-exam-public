"""通信画像の寸法と内容の検証。"""

import numpy as np
import pytest

from src.experiment.images import decode_image, encode_image


def test_roundtrip() -> None:
    """画像がPNG通信で変化しない。"""
    pixels = np.random.default_rng(42).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    assert np.array_equal(pixels, decode_image(encode_image(pixels)))


@pytest.mark.parametrize("encoded", [None, "x" * 400_001, "not-base64!"])
def test_bad_image(encoded: str | None) -> None:
    """欠落・過大・不正符号化の画像を拒否する。"""
    with pytest.raises(ValueError):
        decode_image(encoded)
