"""変形する具材の形状、把持方向、配置の評価。"""

import math

import numpy as np
from numpy.typing import NDArray

from src.experiment.messages import Quaternion, Vector3

Points = NDArray[np.float64]


def vector(values: Points) -> Vector3:
    """三成分を通信可能な値へ変換する。"""
    return float(values[0]), float(values[1]), float(values[2])


def shape_change(points: Points, rest: Points) -> float:
    """平行移動と回転を除いた頂点位置のRMS差を返す。"""
    if points.shape != rest.shape or not np.isfinite(points).all():
        raise ValueError("弾性メッシュの頂点が不正です")
    a = rest - rest.mean(axis=0)
    b = points - points.mean(axis=0)
    u, _, vh = np.linalg.svd(a.T @ b)
    correction = np.diag([1.0, 1.0, float(np.linalg.det(u @ vh))])
    fitted = a @ (u @ correction @ vh)
    return float(np.sqrt(np.mean(np.sum((fitted - b) ** 2, axis=1))))


def grasp_geometry(points: Points) -> tuple[Quaternion, float]:
    """水平面の短い方向から挟む下向き姿勢と半幅を返す。"""
    centered = points[:, :2] - points[:, :2].mean(axis=0)
    values, axes = np.linalg.eigh(centered.T @ centered)
    minor = (
        axes[:, 0]
        if values[1] > 1.15 * max(values[0], 1e-12)
        else np.array([0.0, -1.0])
    )
    if minor[1] > 0:
        minor = -minor
    yaw = math.atan2(float(minor[0]), float(-minor[1]))
    quaternion: Quaternion = (0.0, math.cos(yaw / 2), math.sin(yaw / 2), 0.0)
    projection = centered @ minor
    width = float(np.ptp(projection) / 2)
    return quaternion, width


def served(points: Points, speed: float, hand: Vector3) -> bool:
    """全頂点が丼内にあり、手先から離れて静止しているか判定する。"""
    distance = np.linalg.norm(points[:, :2] - [-0.06, 0], axis=1)
    return bool(
        np.max(distance) < 0.105
        and np.min(points[:, 2]) > 0.867
        and np.max(points[:, 2]) < 0.94
        and np.linalg.norm(points.mean(axis=0) - hand) > 0.07
        and speed < 0.04
    )
