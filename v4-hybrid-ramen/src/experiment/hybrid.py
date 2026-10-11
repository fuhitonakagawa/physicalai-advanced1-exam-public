"""接近・運搬のルールと、学習済み方策による把持の分担。"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from src.config.settings import Settings
from src.environment.ramen import RamenEnvironment
from src.environment.scene import WORLD_OFFSET
from src.experiment.types import Observation, Policy

Record = Callable[[Observation, int, str, str], None]
DOWN = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])


class GraspTask(BaseModel):
    """推論で把持する対象と短い指示。"""

    model_config = ConfigDict(extra="forbid", frozen=True)
    food: Literal["egg", "fish_cake"]
    instruction: str = Field(min_length=1, max_length=120)


@dataclass(frozen=True)
class TaskResult:
    """推論と物理判定に基づく具材単位の結果。"""

    food: str
    success: bool
    reason: str
    inference_steps: int
    contact_candidate: bool
    lift_verified: bool


class HybridRunner:
    """状態遷移に従ってルール制御と方策を切り替える。"""

    def __init__(
        self,
        environment: RamenEnvironment,
        policy: Policy,
        settings: Settings,
        record: Record,
    ) -> None:
        """物理環境、把持方策、記録先を保持する。"""
        self.environment = environment
        self.policy = policy
        self.settings = settings
        self.record = record
        self.last_prediction: NDArray[np.float32] = np.zeros(7, dtype=np.float32)

    def move(
        self,
        target: NDArray[np.float64],
        rotation: NDArray[np.float64],
        opening: float,
        phase: str,
        food: str,
    ) -> bool:
        """具材の座標を変更せず、手先だけをルールで目標へ動かす。"""
        env = self.environment
        start = env.data.site("grasp").xpos.copy()
        start_opening = env.controller.gripper_opening
        steps = round(self.settings.rule_move_seconds * 20)
        for index in range(steps + 5):
            ratio = min(1.0, (index + 1) / steps)
            ratio = ratio * ratio * (3 - 2 * ratio)
            env.controller.command_pose(
                env.data,
                start + ratio * (target - start),
                rotation,
                start_opening + ratio * (opening - start_opening),
            )
            result = env.advance()
            self.record(result.observation, env.steps, phase, food)
        return bool(np.linalg.norm(env.data.site("grasp").xpos - target) < 0.012)

    def grasp(
        self, task: GraspTask, anchor: NDArray[np.float64]
    ) -> tuple[bool, int, str]:
        """方策が生成する動作だけで把持接触に到達するかを評価する。"""
        self.policy.reset()
        env = self.environment
        observation = env.observation()
        held_rotation = env.data.site("grasp").xmat.reshape(3, 3).copy()
        touching_steps = 0
        for step in range(1, self.settings.grasp_max_steps + 1):
            action = self.policy.select_action(observation, task.instruction)
            if action.shape != (7,) or not np.isfinite(action).all():
                return False, step, "invalid_model_action"
            self.last_prediction = np.asarray(action, dtype=np.float32)
            if self.settings.grasp_control == "gripper":
                applied = np.zeros(7, dtype=np.float32)
                applied[6] = float(np.clip(action[6], -1, 1))
                env.controller.last_action = applied
                opening = float(
                    np.clip(
                        env.controller.gripper_opening
                        - 0.005 * float(np.sign(applied[6])),
                        0,
                        0.04,
                    )
                )
                env.controller.command_pose(env.data, anchor, held_rotation, opening)
                result = env.advance()
            else:
                result = env.step(np.clip(action, -1, 1).astype(np.float32))
            observation = result.observation
            self.record(
                observation,
                env.steps,
                f"MODEL: {self.settings.grasp_control}",
                task.food,
            )
            touching_steps = touching_steps + 1 if env.finger_contacts(task.food) else 0
            if touching_steps >= 3:
                return True, step, "contact_candidate"
            if np.linalg.norm(env.data.site("grasp").xpos - anchor) > 0.08:
                return False, step, "left_grasp_region"
        return False, self.settings.grasp_max_steps, "grasp_timeout"

    def approach(self, task: GraspTask) -> NDArray[np.float64] | None:
        """トレー上の対象に、開いた指と下向きの手先で接近する。"""
        env = self.environment
        target = env.data.body(task.food).xpos.copy()
        target[2] += self.settings.approach_clearance
        safe = env.data.site("grasp").xpos.copy()
        safe[2] = WORLD_OFFSET[2] + 0.27
        over_food = np.array([target[0], target[1], safe[2]])
        for waypoint in (safe, over_food, target):
            if not self.move(waypoint, DOWN, 0.04, "RULE: approach", task.food):
                return None
        return np.asarray(target, dtype=np.float64)

    def verify_lift(self, task: GraspTask, opening: float) -> bool:
        """把持した幅を維持し、具材が手先と一緒に持ち上がるか確かめる。"""
        env = self.environment
        before_height = float(env.data.body(task.food).xpos[2])
        target = env.data.site("grasp").xpos.copy()
        target[2] += 0.10
        rotation = env.data.site("grasp").xmat.reshape(3, 3).copy()
        if not self.move(target, rotation, opening, "RULE: lift check", task.food):
            return False
        position = env.data.body(task.food).xpos
        return bool(
            position[2] - before_height > 0.06
            and np.linalg.norm(position - env.data.site("grasp").xpos) < 0.07
            and env.finger_contacts(task.food)
        )

    def transfer(self, task: GraspTask, opening: float) -> bool:
        """把持状態を保って丼へ運び、指を開いて落とす。"""
        env = self.environment
        rotation = env.data.site("grasp").xmat.reshape(3, 3).copy()
        target = env.data.site("grasp").xpos.copy()
        target[2] = WORLD_OFFSET[2] + 0.17
        if not self.move(target, rotation, opening, "RULE: carry", task.food):
            return False
        bowl = env.data.body("ramen_bowl").xpos
        target[:2] = bowl[:2] + [0, -0.03 if task.food == "egg" else 0.03]
        if not self.move(target, rotation, opening, "RULE: carry", task.food):
            return False
        if not env.finger_contacts(task.food):
            return False
        self.move(target, rotation, 0.04, "RULE: release", task.food)
        target[2] = WORLD_OFFSET[2] + 0.27
        self.move(target, DOWN, 0.04, "RULE: retreat", task.food)
        self.move(target, DOWN, 0.04, "RULE: settle", task.food)
        return env.stable_steps[task.food] >= 10

    def run_task(self, task: GraspTask) -> TaskResult:
        """接近・推論把持・持ち上げ判定・運搬を実行する。"""
        anchor = self.approach(task)
        if anchor is None:
            return TaskResult(task.food, False, "approach_failed", 0, False, False)
        candidate, steps, reason = self.grasp(task, anchor)
        if not candidate:
            self.recover(task)
            return TaskResult(task.food, False, reason, steps, False, False)
        opening = self.environment.controller.gripper_opening
        verified = self.verify_lift(task, opening)
        if not verified:
            self.recover(task)
            return TaskResult(task.food, False, "lift_failed", steps, True, False)
        served = self.transfer(task, opening)
        return TaskResult(
            task.food,
            served,
            "served" if served else "transfer_failed",
            steps,
            True,
            True,
        )

    def recover(self, task: GraspTask) -> None:
        """失敗した具材を運ばず、指を開いて上へ退避する。"""
        target = self.environment.data.site("grasp").xpos.copy()
        self.move(target, DOWN, 0.04, "RULE: release after failure", task.food)
        target[2] = WORLD_OFFSET[2] + 0.27
        self.move(target, DOWN, 0.04, "RULE: retreat after failure", task.food)
