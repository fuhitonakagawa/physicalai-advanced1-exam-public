"""実際のカメラ観測を返すラーメン盛り付け環境。"""

import mujoco
import numpy as np

from src.config.settings import Settings
from src.environment.camera import Camera
from src.environment.control import CartesianController
from src.environment.scene import (
    FOOD_NAMES,
    INITIAL_ARM_POSITION,
    SOUP_HEIGHT,
    WORLD_OFFSET,
    scene_xml,
)
from src.experiment.observations import quaternion_to_axis_angle
from src.experiment.types import FloatArray, Observation, StepResult


class RamenEnvironment:
    """Pandaが左右の具材を丼へ運ぶ実験。"""

    def __init__(self, settings: Settings) -> None:
        """物理モデルとカメラを構築する。"""
        self.settings = settings
        self.model = mujoco.MjModel.from_xml_string(scene_xml(settings.seed))
        self.data = mujoco.MjData(self.model)
        self.controller = CartesianController(self.model)
        self.camera = Camera(self.model, settings.renderer)
        self.stable_steps = {name: 0 for name in FOOD_NAMES}
        self.lifted = {name: False for name in FOOD_NAMES}
        self.steps = 0

    def reset(self, seed: int, initial_state_index: int | None) -> Observation:
        """場面の乱数シードに対応する初期状態を返す。"""
        if seed != self.settings.seed or initial_state_index is not None:
            raise ValueError("この環境は設定したseedの単一試行に対応します")
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:9] = np.r_[INITIAL_ARM_POSITION, 0.04, 0.04]
        self.data.ctrl[:] = np.r_[self.data.qpos[:7], 255]
        self.controller.reset()
        mujoco.mj_forward(self.model, self.data)
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)
        self.stable_steps = {name: 0 for name in FOOD_NAMES}
        self.lifted = {name: False for name in FOOD_NAMES}
        self.steps = 0
        return self.observation()

    def observation(self) -> Observation:
        """現在の画像2枚と手先・指の実測状態を返す。"""
        quaternion = np.zeros(4)
        mujoco.mju_mat2Quat(quaternion, self.data.body("hand").xmat)
        axis_angle = quaternion_to_axis_angle(
            np.roll(quaternion, -1).astype(np.float32)
        )
        state = np.r_[
            self.data.site("grasp").xpos,
            axis_angle,
            self.data.qpos[7],
            -self.data.qpos[8],
        ].astype(np.float32)
        return Observation(
            self.camera.render(self.data, "agentview"),
            self.camera.render(self.data, "wrist"),
            state,
        )

    def update_serving_status(self) -> None:
        """丼の内側で静止し、指から離れた具材だけを盛り付けと判定する。"""
        bowl = self.data.body("ramen_bowl").xpos
        for name in FOOD_NAMES:
            position = self.data.body(name).xpos
            self.lifted[name] |= bool(position[2] > WORLD_OFFSET[2] + 0.10)
            inside = np.linalg.norm(position[:2] - bowl[:2]) < 0.068
            at_surface = (
                WORLD_OFFSET[2] + SOUP_HEIGHT < position[2] < WORLD_OFFSET[2] + 0.14
            )
            released = np.linalg.norm(position - self.data.site("grasp").xpos) > 0.07
            quiet = np.linalg.norm(self.data.body(name).cvel[3:]) < 0.04
            valid = bool(inside and at_surface and released and quiet)
            self.stable_steps[name] = self.stable_steps[name] + 1 if valid else 0

    def step(self, action: FloatArray) -> StepResult:
        """SmolVLAの指令に応じて物理状態を50ms進める。"""
        self.controller.apply(self.data, action)
        for _ in range(25):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        self.update_serving_status()
        success = all(value >= 10 for value in self.stable_steps.values())
        return StepResult(self.observation(), success, success)

    def serving_results(self) -> dict[str, object]:
        """各具材の実位置と安定配置判定を返す。"""
        return {
            name: {
                "served": self.stable_steps[name] >= 10,
                "lifted": self.lifted[name],
                "position": self.data.body(name).xpos.tolist(),
            }
            for name in FOOD_NAMES
        }

    def diagnostics(self) -> dict[str, object]:
        """手首の視野と動作指令を評価用に返す。"""
        camera = self.data.camera("wrist")
        rotation = camera.xmat.reshape(3, 3)
        half_fov = np.tan(np.deg2rad(self.model.camera("wrist").fovy[0]) / 2)
        in_view = {}
        for name in FOOD_NAMES:
            relative = rotation.T @ (self.data.body(name).xpos - camera.xpos)
            depth = -relative[2]
            in_view[name] = bool(
                depth > 0 and np.max(np.abs(relative[:2])) < depth * half_fov
            )
        return {
            "action": self.controller.last_action.tolist(),
            "eef_position": self.data.site("grasp").xpos.tolist(),
            "wrist_optical_axis": (-rotation[:, 2]).tolist(),
            "food_in_wrist_frustum": in_view,
        }

    def close(self) -> None:
        """カメラのリソースを解放する。"""
        self.camera.close()
