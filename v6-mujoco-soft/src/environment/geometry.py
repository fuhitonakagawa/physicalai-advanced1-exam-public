"""姿勢変換と盛り付け判定。"""

import numpy as np
from numpy.typing import NDArray

from src.experiment.messages import Quaternion, Vector3


def rotation_matrix(quaternion: Quaternion) -> NDArray[np.float64]:
    """wxyzクォータニオンを回転行列へ変換する。"""
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def grasp_position(position: Vector3, quaternion: Quaternion) -> Vector3:
    """手首から97mm先の把持点を世界座標で返す。"""
    result = np.asarray(position) + rotation_matrix(quaternion) @ [0, 0, 0.097]
    return float(result[0]), float(result[1]), float(result[2])
