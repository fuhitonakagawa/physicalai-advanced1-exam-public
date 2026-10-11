"""シミュレータと推論モデルの入出力契約。"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]
FloatArray = NDArray[np.float32]


@dataclass(frozen=True)
class Observation:
    """2台のカメラ画像と8次元の腕の状態。"""

    main_image: ImageArray
    wrist_image: ImageArray
    state: FloatArray


class Policy(Protocol):
    """観測から7次元の動作を返す方策。"""

    def reset(self) -> None:
        """試行間で内部の動作列を初期化する。"""
        ...

    def select_action(self, observation: Observation, instruction: str) -> FloatArray:
        """画像、状態、指示に対応する動作を返す。"""
        ...
