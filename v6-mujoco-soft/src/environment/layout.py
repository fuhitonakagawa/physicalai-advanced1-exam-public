"""左右の具材の初期配置。"""

import math
from dataclasses import dataclass

import numpy as np

from src.experiment.messages import Quaternion, Vector3


@dataclass(frozen=True)
class FoodPose:
    """具材を置く世界座標と向き。"""

    position: Vector3
    quaternion: Quaternion


def food_layout(seed: int) -> dict[str, FoodPose]:
    """同じseedで同じトレー配置を返す。"""
    generator = np.random.default_rng(seed)
    result = {}
    for name, y in (("egg", -0.225), ("fish_cake", 0.225)):
        x = -0.14 + float(generator.uniform(-0.025, 0.025))
        y += float(generator.uniform(-0.015, 0.015))
        angle = float(generator.uniform(-math.pi, math.pi))
        result[name] = FoodPose(
            (x, y, 0.84), (math.cos(angle / 2), 0, 0, math.sin(angle / 2))
        )
    return result
