"""変形した形状と把持負荷に基づく、再試行可能な盛り付け。"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.config.settings import Settings
from src.experiment.images import decode_image
from src.experiment.messages import FoodName, ObservationMessage, Quaternion, Vector3
from src.experiment.types import Observation, Policy


class ControlEnvironment(Protocol):
    """実観測と手先目標の境界。"""

    current: ObservationMessage

    def step(
        self,
        position: Vector3,
        quaternion: Quaternion,
        opening: float,
        phase: str,
        food: FoodName,
    ) -> ObservationMessage:
        """一周期の指令に対する実観測を返す。"""
        ...


class GraspTask(BaseModel):
    """把持する具材とSmolVLAへの指示。"""

    model_config = ConfigDict(extra="forbid")
    food: FoodName
    instruction: str = Field(min_length=1, max_length=120)


@dataclass(frozen=True)
class TaskResult:
    """補助制御と推論を区別した、具材単位の実測結果。"""

    food: str
    success: bool
    reason: str
    attempts: int
    model_steps: int
    assisted_steps: int
    ai_triggered_closure: bool
    max_deformation_rms: float


class AssistedRunner:
    """接触閉鎖、保持確認、低い位置からの開放を組み合わせる。"""

    def __init__(
        self, environment: ControlEnvironment, policy: Policy | None, settings: Settings
    ) -> None:
        """実環境、方策、把持条件を保持する。"""
        self.environment = environment
        self.policy = policy
        self.settings = settings
        self.opening = 0.04
        self.minimum_opening = 0.002
        self.last_prediction: list[float] | None = None
        self.model_steps = 0
        self.assisted_steps = 0
        self.ai_triggered = False
        self.peak_deformation = 0.0

    def step(
        self, position: Vector3, quaternion: Quaternion, phase: str, food: FoodName
    ) -> ObservationMessage:
        """物理状態を進め、形状変化の最大値を記録する。"""
        state = self.environment.step(position, quaternion, self.opening, phase, food)
        self.peak_deformation = max(
            self.peak_deformation, state.ingredients[food].deformation_rms
        )
        return state

    def move(
        self,
        target: Vector3,
        quaternion: Quaternion,
        phase: str,
        food: FoodName,
        opening: float,
        holding: bool = False,
    ) -> bool:
        """加減速を抑えた移動中も、保持の喪失を監視する。"""
        start = np.asarray(self.environment.current.state[:3])
        initial_opening = self.opening
        steps = round(self.settings.rule_move_seconds * 20)
        missing = 0
        for index in range(steps + 5):
            ratio = min(1.0, (index + 1) / steps)
            ratio = ratio * ratio * (3 - 2 * ratio)
            position = start + ratio * (np.asarray(target) - start)
            if holding:
                ingredient = self.environment.current.ingredients[food]
                if (
                    max(ingredient.left_load, ingredient.right_load)
                    < self.settings.grip_load_limit
                ):
                    if not ingredient.finger_contacts:
                        self.opening = max(
                            self.minimum_opening,
                            self.opening - self.settings.close_step,
                        )
            else:
                self.opening = initial_opening + ratio * (opening - initial_opening)
            state = self.step(tuple(position), quaternion, phase, food)
            if holding:
                ingredient = state.ingredients[food]
                missing = missing + 1 if not ingredient.finger_contacts else 0
                distance = np.linalg.norm(
                    np.asarray(ingredient.position) - state.state[:3]
                )
                if (
                    missing >= 5
                    or distance > 0.065
                    or ingredient.deformation_rms > self.settings.max_deformation_rms
                ):
                    return False
        return bool(
            np.linalg.norm(np.asarray(self.environment.current.state[:3]) - target)
            < 0.012
        )

    def approach(self, food: FoodName) -> tuple[Vector3, Quaternion] | None:
        """現在の短軸方向を選び、指先がトレーに当たらない高さへ接近する。"""
        state = self.environment.current
        ingredient = state.ingredients[food]
        x, y, z = ingredient.position
        if not (-0.35 < x < 0.2 and abs(y) < 0.4 and 0.80 < z < 0.96):
            return None
        quaternion = ingredient.grasp_quaternion
        target: Vector3 = (
            x,
            y,
            max(
                z - 0.001,
                ingredient.bounds_min[2] + 0.0144 + self.settings.approach_clearance,
            ),
        )
        self.minimum_opening = max(
            0.002, ingredient.half_width * (1 - self.settings.max_compression) - 0.0015
        )
        for waypoint in ((state.state[0], state.state[1], 1.10), (x, y, 1.10), target):
            if not self.move(waypoint, quaternion, "RULE: approach", food, 0.04):
                return None
        return target, quaternion

    def propose(self, task: GraspTask, target: Vector3, quaternion: Quaternion) -> None:
        """SmolVLAの閉じる提案を待ち、未提案なら補助制御へ渡す。"""
        if self.policy is None:
            return
        self.policy.reset()
        for _ in range(self.settings.model_proposal_steps):
            state = self.environment.current
            observation = Observation(
                decode_image(state.main_png),
                decode_image(state.wrist_png),
                np.asarray(state.state, dtype=np.float32),
            )
            action = self.policy.select_action(observation, task.instruction)
            if action.shape != (7,) or not np.isfinite(action).all():
                raise ValueError("モデルの出力が不正です")
            self.last_prediction = action.tolist()
            self.model_steps += 1
            self.step(target, quaternion, "MODEL: closure proposal", task.food)
            if action[6] > 0:
                self.ai_triggered = True
                return

    def close_gripper(
        self, food: FoodName, target: Vector3, quaternion: Quaternion
    ) -> str:
        """接触負荷と変形量を見ながら少しずつ指を閉じる。"""
        stable = 0
        for _ in range(self.settings.grasp_max_steps):
            current = self.environment.current.ingredients[food]
            if current.deformation_rms > self.settings.max_deformation_rms:
                return "deformation_limit"
            if (
                max(current.left_load, current.right_load)
                > self.settings.grip_load_limit
            ):
                return "load_limit"
            ready = current.finger_contacts
            if not ready:
                self.opening = max(
                    self.minimum_opening, self.opening - self.settings.close_step
                )
            state = self.step(target, quaternion, "ASSIST: contact closure", food)
            self.assisted_steps += 1
            item = state.ingredients[food]
            stable = stable + 1 if item.finger_contacts else 0
            if stable >= 3:
                return "contact_verified"
        return "contact_timeout"

    def lift(self, food: FoodName, quaternion: Quaternion) -> bool:
        """具材の中心が持ち上がり、指付近に保たれることを検証する。"""
        state = self.environment.current
        before = state.ingredients[food].position[2]
        x, y, z = state.state[:3]
        if not self.move(
            (x, y, z + 0.08),
            quaternion,
            "ASSIST: lift check",
            food,
            self.opening,
            holding=True,
        ):
            return False
        current = self.environment.current
        item = current.ingredients[food]
        return bool(
            item.position[2] - before > 0.055
            and item.finger_contacts
            and np.linalg.norm(np.asarray(item.position) - current.state[:3]) < 0.05
        )

    def transfer(self, food: FoodName, quaternion: Quaternion) -> bool:
        """丼の縁を越えてから低い位置へ下ろし、開放する。"""
        state = self.environment.current
        destination = (-0.06, -0.032 if food == "egg" else 0.032)
        for target in (
            (state.state[0], state.state[1], 1.07),
            (*destination, 1.07),
            (*destination, 0.909),
        ):
            if not self.move(
                target,
                quaternion,
                "ASSIST: carry and hold",
                food,
                self.opening,
                holding=True,
            ):
                return False
        self.move((*destination, 0.909), quaternion, "RULE: release", food, 0.04)
        self.move((*destination, 1.10), quaternion, "RULE: retreat", food, 0.04)
        for _ in range(self.settings.settle_steps):
            state = self.step((*destination, 1.10), quaternion, "RULE: settle", food)
            if state.ingredients[food].stable_steps >= 10:
                return True
        return False

    def recover(self, food: FoodName) -> None:
        """その場で指を開き、形状を再観測できる位置へ退避する。"""
        state = self.environment.current
        x, y, z = state.state[:3]
        self.move(
            (x, y, z), state.hand_quaternion, "ASSIST: release for retry", food, 0.04
        )
        self.move(
            (x, y, 1.10), state.hand_quaternion, "ASSIST: retreat for retry", food, 0.04
        )

    def run_task(self, task: GraspTask) -> TaskResult:
        """失敗を記録し、現在の具材位置から再把持を試みる。"""
        self.model_steps = self.assisted_steps = 0
        self.ai_triggered = False
        self.peak_deformation = 0.0
        reason = "approach_failed"
        for attempt in range(1, self.settings.max_attempts + 1):
            approach = self.approach(task.food)
            if approach is None:
                break
            target, quaternion = approach
            self.propose(task, target, quaternion)
            reason = self.close_gripper(task.food, target, quaternion)
            if reason == "contact_verified":
                if self.lift(task.food, quaternion):
                    if self.transfer(task.food, quaternion):
                        return self.result(task.food, True, "served", attempt)
                    reason = "transfer_failed"
                else:
                    reason = "lift_failed"
            self.recover(task.food)
        return self.result(task.food, False, reason, attempt)

    def result(
        self, food: FoodName, success: bool, reason: str, attempts: int
    ) -> TaskResult:
        """推論、補助制御、変形の観測値をまとめる。"""
        return TaskResult(
            food,
            success,
            reason,
            attempts,
            self.model_steps,
            self.assisted_steps,
            self.ai_triggered,
            self.peak_deformation,
        )
