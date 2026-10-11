"""学習環境とカメラ・姿勢・開閉指令の対応。"""

import math

import numpy as np

from src.environment.ramen import RamenEnvironment
from src.environment.scene import INITIAL_ARM_POSITION
from src.policy.smolvla import policy_image


def test_native_pixels_match_libero_opengl_preprocessing() -> None:
    """OpenGLの下原点画像に対する変換と同じ画素位置になる。"""
    raw = np.zeros((256, 256, 3), dtype=np.uint8)
    raw[3, 7] = [255, 40, 10]
    native = raw[::-1].copy()
    expected = np.zeros_like(raw)
    expected[252, 248] = [255, 40, 10]
    np.testing.assert_array_equal(policy_image(native), expected)


def test_camera_mount_and_initial_hand_pose(environment: RamenEnvironment) -> None:
    """手首カメラのロール角、画角、初期関節姿勢を検証する。"""
    camera = environment.model.camera("wrist")
    np.testing.assert_allclose(camera.pos, [0.05, 0, 0])
    np.testing.assert_allclose(camera.quat, [0, math.sqrt(0.5), math.sqrt(0.5), 0])
    assert camera.fovy[0] == 75
    np.testing.assert_allclose(
        environment.data.qpos[:7], INITIAL_ARM_POSITION, atol=0.025
    )
    assert environment.data.body("hand").xmat.reshape(3, 3)[2, 2] < -0.98


def test_observed_orientation_uses_hand_not_control_site(
    environment: RamenEnvironment,
) -> None:
    """入力姿勢の座標系と把持制御サイトを混同しない。"""
    observation = environment.observation("test")
    assert abs(observation.state[3]) > 3.0
    assert abs(observation.state[4]) < 0.05
    relative = environment.data.body("hand").xmat.reshape(
        3, 3
    ).T @ environment.data.site("grasp").xmat.reshape(3, 3)
    np.testing.assert_allclose(relative, [[0, 1, 0], [-1, 0, 0], [0, 0, 1]], atol=1e-6)
