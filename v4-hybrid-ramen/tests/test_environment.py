"""実観測、成功判定、物理状態の有限性。"""

import mujoco
import numpy as np

from src.environment.ramen import RamenEnvironment
from src.environment.scene import FOOD_NAMES, WORLD_OFFSET


def place_food(environment: RamenEnvironment, name: str, xyz: list[float]) -> None:
    """成功判定の境界を検証するためにテスト状態を設定する。"""
    joint = environment.model.joint(f"{name}_joint")
    address = int(joint.qposadr[0])
    environment.data.qpos[address : address + 3] = xyz
    environment.data.qvel[:] = 0
    mujoco.mj_forward(environment.model, environment.data)


def test_real_camera_and_state_are_finite(environment: RamenEnvironment) -> None:
    """画像に場面が描かれ、状態が8次元であることを確認する。"""
    observation = environment.observation()
    assert observation.main_image.shape == (256, 256, 3)
    assert observation.wrist_image.shape == (256, 256, 3)
    assert observation.main_image.std() > 20
    assert observation.state.shape == (8,)
    assert np.isfinite(observation.state).all()
    assert observation.state[6] > 0
    assert observation.state[7] < 0


def test_initial_trays_are_not_served(environment: RamenEnvironment) -> None:
    """初期位置を盛り付け成功と判定しない。"""
    for _ in range(20):
        environment.update_serving_status()
    assert all(count == 0 for count in environment.stable_steps.values())


def test_serving_requires_settled_release_in_bowl(
    environment: RamenEnvironment,
) -> None:
    """丼内の高さと安定判定の継続を確認する。"""
    center = environment.data.body("ramen_bowl").xpos
    for offset, name in zip((-0.03, 0.03), FOOD_NAMES, strict=True):
        place_food(
            environment, name, [float(center[0]), float(center[1] + offset), 0.89]
        )
    for _ in range(9):
        environment.update_serving_status()
    assert not all(count >= 10 for count in environment.stable_steps.values())
    environment.update_serving_status()
    assert all(count >= 10 for count in environment.stable_steps.values())
    place_food(
        environment, "egg", [float(center[0]), float(center[1]), float(WORLD_OFFSET[2])]
    )
    environment.update_serving_status()
    assert environment.stable_steps["egg"] == 0


def test_prediction_changes_robot_without_teleporting_food(
    environment: RamenEnvironment,
) -> None:
    """ロボットへ相対動作を与えても具材の座標を直接設定しない。"""
    before = {name: environment.data.body(name).xpos.copy() for name in FOOD_NAMES}
    result = environment.step(np.array([0.1, 0, 0, 0, 0, 0, -1], dtype=np.float32))
    assert np.isfinite(result.observation.state).all()
    assert not result.success
    for name in FOOD_NAMES:
        np.testing.assert_allclose(
            environment.data.body(name).xpos, before[name], atol=0.002
        )


def test_whole_food_must_fit_inside_bowl_opening(environment: RamenEnvironment) -> None:
    """中心だけが丼内にあっても、具材全体が縁を越える配置を拒否する。"""
    center = environment.data.body("ramen_bowl").xpos
    place_food(environment, "egg", [float(center[0] + 0.09), float(center[1]), 0.89])
    for _ in range(12):
        environment.update_serving_status()
    assert environment.stable_steps["egg"] == 0
    place_food(environment, "egg", [float(center[0] + 0.07), float(center[1]), 0.89])
    for _ in range(12):
        environment.update_serving_status()
    assert environment.stable_steps["egg"] >= 10
