"""MuJoCoの弾性メッシュを把持し、変形と配置を評価する。"""

import mujoco
import numpy as np

from src.config.settings import Settings
from src.environment.camera import Camera
from src.environment.control import CartesianController
from src.environment.geometry import rotation_matrix
from src.environment.scene import FOOD_NAMES, INITIAL_ARM_POSITION, scene_xml
from src.environment.soft_geometry import (
    Points,
    grasp_geometry,
    served,
    shape_change,
    vector,
)
from src.experiment.images import encode_image
from src.experiment.messages import Command, FoodState, ObservationMessage, Quaternion
from src.experiment.observations import quaternion_to_axis_angle

SITE_FROM_HAND = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


class RamenEnvironment:
    """剛体の腕と柔軟な具材による接触物理。"""

    def __init__(self, settings: Settings, images: bool) -> None:
        """弾性体とカメラを用意し、具材をトレー上で落ち着かせる。"""
        self.model = mujoco.MjModel.from_xml_string(scene_xml(settings))
        self.data = mujoco.MjData(self.model)
        self.controller = CartesianController(self.model)
        self.camera = Camera(self.model, settings.renderer) if images else None
        self.flex_ids = {
            name: mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_FLEX, name)
            for name in FOOD_NAMES
        }
        self.rest = {}
        self.stable = {name: 0 for name in FOOD_NAMES}
        self.speeds = {name: 0.0 for name in FOOD_NAMES}
        self.max_deformation = {name: 0.0 for name in FOOD_NAMES}
        self.steps = 0
        self.data.qpos[:9] = np.r_[INITIAL_ARM_POSITION, 0.04, 0.04]
        self.data.ctrl[:] = np.r_[self.data.qpos[:7], 255]
        mujoco.mj_forward(self.model, self.data)
        self.rest = {name: self.points(name) for name in FOOD_NAMES}
        for _ in range(250):
            mujoco.mj_step(self.model, self.data)

    def points(self, name: str) -> Points:
        """現在の柔軟物の全頂点を返す。"""
        index = self.flex_ids[name]
        start = self.model.flex_vertadr[index]
        count = self.model.flex_vertnum[index]
        return np.asarray(
            self.data.flexvert_xpos[start : start + count], dtype=np.float64
        ).copy()

    def contact_loads(self, name: str) -> tuple[float, float]:
        """対象flexと左右の指との法線接触力を合計する。"""
        finger_ids = [
            self.model.body(side).id for side in ("left_finger", "right_finger")
        ]
        loads = [0.0, 0.0]
        force = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            for side in (0, 1):
                if contact.flex[side] != self.flex_ids[name]:
                    continue
                geom = int(contact.geom[1 - side])
                if geom < 0:
                    continue
                body = int(self.model.geom_bodyid[geom])
                if body in finger_ids:
                    mujoco.mj_contactForce(self.model, self.data, index, force)
                    loads[finger_ids.index(body)] += max(0.0, float(force[0]))
        return loads[0], loads[1]

    def advance(self, command: Command) -> None:
        """指令に従って50ms進め、全頂点の静止と変形を評価する。"""
        old = {name: self.points(name) for name in FOOD_NAMES}
        rotation = rotation_matrix(command.quaternion) @ SITE_FROM_HAND
        self.controller.command_pose(
            self.data, np.asarray(command.position), rotation, command.opening
        )
        for _ in range(25):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1
        hand = vector(self.data.site("grasp").xpos)
        for name in FOOD_NAMES:
            points = self.points(name)
            self.speeds[name] = float(
                np.max(np.linalg.norm(points - old[name], axis=1)) / 0.05
            )
            self.max_deformation[name] = max(
                self.max_deformation[name], shape_change(points, self.rest[name])
            )
            self.stable[name] = (
                self.stable[name] + 1 if served(points, self.speeds[name], hand) else 0
            )
        if not np.isfinite(self.data.qpos).all() or np.any(self.data.warning.number):
            raise RuntimeError("柔軟物の物理計算が不安定です")

    def food_state(self, name: str) -> FoodState:
        """変形した形状と接触力をまとめる。"""
        points = self.points(name)
        quaternion, width = grasp_geometry(points)
        left, right = self.contact_loads(name)
        return FoodState(
            position=vector(points.mean(axis=0)),
            speed=self.speeds[name],
            finger_contacts=left > 0.05 and right > 0.05,
            stable_steps=self.stable[name],
            bounds_min=vector(points.min(axis=0)),
            bounds_max=vector(points.max(axis=0)),
            grasp_quaternion=quaternion,
            half_width=width,
            left_load=left,
            right_load=right,
            deformation_rms=shape_change(points, self.rest[name]),
            vertex_count=len(points),
            contact_kind="flex_contact_force",
        )

    def observation(self, request_id: str) -> ObservationMessage:
        """同じ物理時刻のカメラと柔軟物の状態を返す。"""
        raw = self.data.body("hand").xquat
        quaternion: Quaternion = (
            float(raw[0]),
            float(raw[1]),
            float(raw[2]),
            float(raw[3]),
        )
        axis = quaternion_to_axis_angle(
            np.roll(np.asarray(quaternion, dtype=np.float32), -1)
        )
        state = [
            *self.data.site("grasp").xpos.tolist(),
            *axis.tolist(),
            float(self.data.qpos[7]),
            -float(self.data.qpos[8]),
        ]
        return ObservationMessage(
            request_id=request_id,
            steps=self.steps,
            state=state,
            hand_quaternion=quaternion,
            joint_positions=self.data.qpos[:9].tolist(),
            ingredients={name: self.food_state(name) for name in FOOD_NAMES},
            main_png=encode_image(self.camera.render(self.data, "agentview"))
            if self.camera
            else None,
            wrist_png=encode_image(self.camera.render(self.data, "wrist"))
            if self.camera
            else None,
        )

    def close(self) -> None:
        """カメラを解放する。"""
        if self.camera:
            self.camera.close()
