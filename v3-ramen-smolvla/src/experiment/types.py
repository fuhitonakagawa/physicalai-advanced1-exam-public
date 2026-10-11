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


@dataclass(frozen=True)
class StepResult:
    """環境が返す終了状態と実際の成功判定。"""

    observation: Observation
    terminated: bool
    success: bool


class Policy(Protocol):
    """観測から7次元の動作を返す方策。"""

    def reset(self) -> None:
        """試行間で内部の動作列を初期化する。"""
        ...

    def select_action(self, observation: Observation, instruction: str) -> FloatArray:
        """画像、状態、指示に対応する動作を返す。"""
        ...


class Environment(Protocol):
    """物理環境の制御境界。"""

    def reset(self, seed: int, initial_state_index: int | None) -> Observation:
        """指定した初期状態から試行を始める。"""
        ...

    def step(self, action: FloatArray) -> StepResult:
        """動作を1制御周期だけ実行する。"""
        ...


@dataclass(frozen=True)
class EpisodeResult:
    """成功・失敗の双方を含む試行結果。"""

    seed: int
    initial_state_index: int | None
    success: bool
    steps: int
    clipped_values: int
    wall_seconds: float
