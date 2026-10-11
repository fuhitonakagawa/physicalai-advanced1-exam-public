"""FEM頂点と指腹の距離による、持ち上げ検証前の接触候補。"""

import numpy as np

from src.environment.soft_geometry import Points


def bilateral_proximity(points: Points, left: float, right: float) -> bool:
    """手首座標の頂点が左右の指腹の内側面付近にあるかを返す。"""
    pad = (
        (np.abs(points[:, 0]) < 0.0085)
        & (points[:, 2] > 0.0944)
        & (points[:, 2] < 0.1114)
    )
    left_gap = left + 0.0015 - points[:, 1]
    right_gap = right + 0.0015 + points[:, 1]
    return bool(
        np.any(pad & (left_gap >= -0.003) & (left_gap <= 0.0015))
        and np.any(pad & (right_gap >= -0.003) & (right_gap <= 0.0015))
    )
