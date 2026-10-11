"""ROS JSONに含めるカメラ画像の可逆変換。"""

import base64
import io

import numpy as np
from PIL import Image

from src.experiment.types import ImageArray


def encode_image(pixels: ImageArray) -> str:
    """256角のRGB画像をPNGとして符号化する。"""
    if pixels.shape != (256, 256, 3) or pixels.dtype != np.uint8:
        raise ValueError("RGB画像は256×256で指定してください")
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode("ascii")


def decode_image(encoded: str | None) -> ImageArray:
    """画像の形式と寸法を検証してRGB配列を返す。"""
    if encoded is None or len(encoded) > 400_000:
        raise ValueError("カメラ画像がないか、サイズ上限を超えています")
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(io.BytesIO(raw)) as picture:
        if picture.size != (256, 256) or picture.mode != "RGB":
            raise ValueError("カメラ画像の形式が一致しません")
        return np.asarray(picture, dtype=np.uint8).copy()
