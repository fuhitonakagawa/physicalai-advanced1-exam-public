"""LIBERO観測の向きと状態量の契約。"""

import math

import numpy as np

from src.experiment.types import FloatArray


def quaternion_to_axis_angle(quaternion: FloatArray) -> FloatArray:
    """xyzw四元数をLIBERO方策の軸角表現へ変換する。"""
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("姿勢は有限なxyzw四元数である必要があります")
    w = float(np.clip(quaternion[3], -1, 1))
    denominator = math.sqrt(max(0, 1 - w * w))
    if denominator < 1e-10:
        return np.zeros(3, dtype=np.float32)
    return np.asarray(
        quaternion[:3] / denominator * (2 * math.acos(w)), dtype=np.float32
    )
