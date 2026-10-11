"""変形の測定と、柔軟物全体の配置判定。"""

import numpy as np

from src.environment.geometry import rotation_matrix
from src.environment.soft_geometry import grasp_geometry, served, shape_change


def test_rigid_motion_is_not_deformation() -> None:
    """平行移動と回転だけを柔らかさの証拠にしない。"""
    points = np.random.default_rng(42).normal(size=(30, 3)) * 0.02
    rotation = rotation_matrix((0, 1, 0, 0))
    moved = points @ rotation.T + [0.1, -0.3, 0.8]
    assert shape_change(moved, points) < 1e-12
    compressed = points.copy()
    compressed[:, 1] *= 0.7
    assert shape_change(compressed, points) > 0.001


def test_short_axis_grasp() -> None:
    """楕円体の短軸を指の開閉方向にする。"""
    points = np.array(
        [
            [0.03, 0, 0],
            [-0.03, 0, 0],
            [0, 0.02, 0],
            [0, -0.02, 0],
            [0, 0, 0.01],
            [0, 0, -0.01],
        ]
    )
    quaternion, width = grasp_geometry(points)
    assert abs(width - 0.02) < 1e-12
    assert abs(rotation_matrix(quaternion)[1, 1]) > 0.99


def test_every_vertex_must_be_inside() -> None:
    """中心が丼内でも一部が外にはみ出た配置を拒否する。"""
    points = np.array(
        [
            [-0.08, -0.02, 0.88],
            [-0.04, -0.02, 0.88],
            [-0.08, 0.02, 0.9],
            [-0.04, 0.02, 0.9],
        ]
    )
    assert served(points, 0, (-0.06, 0, 1.1))
    outside = points.copy()
    outside[0, 0] = 0.06
    assert not served(outside, 0, (-0.06, 0, 1.1))
    assert not served(points, 0.1, (-0.06, 0, 1.1))
    assert not served(points, 0, (-0.06, 0, 0.9))
