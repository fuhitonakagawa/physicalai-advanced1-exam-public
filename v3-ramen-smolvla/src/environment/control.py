"""手先の相対動作をPandaの関節制御へ変換する。"""

import mujoco
import numpy as np
from numpy.typing import NDArray

from src.environment.scene import WORLD_OFFSET


class CartesianController:
    """7次元の相対手先指令に対応する関節目標。"""

    def __init__(self, model: mujoco.MjModel) -> None:
        """逆運動学用の作業状態を用意する。"""
        self.model = model
        self.work = mujoco.MjData(model)
        self.site = model.site("grasp").id
        self.jacobian_position = np.zeros((3, model.nv))
        self.jacobian_rotation = np.zeros((3, model.nv))
        self.gripper_opening = 0.04
        self.last_action: NDArray[np.float32] = np.zeros(7, dtype=np.float32)

    def reset(self) -> None:
        """開いた指の目標と動作記録を初期化する。"""
        self.gripper_opening = 0.04
        self.last_action[:] = 0

    def apply(self, data: mujoco.MjData, action: NDArray[np.float32]) -> None:
        """予測された相対位置・回転・開閉を制御目標にする。"""
        self.last_action = action.copy()
        current_position = data.site_xpos[self.site]
        target = np.clip(
            current_position + action[:3] * 0.05,
            WORLD_OFFSET + [0.1, -0.45, 0.03],
            WORLD_OFFSET + [0.72, 0.45, 0.72],
        )
        rotation = action[3:6].astype(np.float64) * 0.5
        angle = float(np.linalg.norm(rotation))
        quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        if angle > 1e-10:
            mujoco.mju_axisAngle2Quat(quaternion, rotation / angle, angle)
        delta = np.zeros(9)
        mujoco.mju_quat2Mat(delta, quaternion)
        target_rotation = delta.reshape(3, 3) @ data.site_xmat[self.site].reshape(3, 3)
        self.work.qpos[:] = data.qpos
        for _ in range(12):
            mujoco.mj_forward(self.model, self.work)
            relative_rotation = (
                target_rotation @ self.work.site_xmat[self.site].reshape(3, 3).T
            )
            error_quaternion = np.zeros(4)
            mujoco.mju_mat2Quat(error_quaternion, relative_rotation.flatten())
            rotational_error = np.zeros(3)
            mujoco.mju_quat2Vel(rotational_error, error_quaternion, 1)
            error = np.r_[target - self.work.site_xpos[self.site], rotational_error]
            if np.linalg.norm(error) < 0.001:
                break
            mujoco.mj_jacSite(
                self.model,
                self.work,
                self.jacobian_position,
                self.jacobian_rotation,
                self.site,
            )
            jacobian = np.vstack(
                [self.jacobian_position[:, :7], self.jacobian_rotation[:, :7]]
            )
            change = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + np.eye(6) * 0.005, error
            )
            self.work.qpos[:7] = np.clip(
                self.work.qpos[:7] + np.clip(change, -0.15, 0.15),
                self.model.jnt_range[:7, 0],
                self.model.jnt_range[:7, 1],
            )
        data.ctrl[:7] = self.work.qpos[:7]
        self.gripper_opening = float(
            np.clip(self.gripper_opening - 0.005 * float(np.sign(action[6])), 0, 0.04)
        )
        data.ctrl[7] = self.gripper_opening / 0.04 * 255
