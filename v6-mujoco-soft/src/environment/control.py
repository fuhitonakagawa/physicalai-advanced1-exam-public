"""手先の相対動作をPandaの関節制御へ変換する。"""

import mujoco
import numpy as np
from numpy.typing import NDArray


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

    def command_pose(
        self,
        data: mujoco.MjData,
        target: NDArray[np.float64],
        target_rotation: NDArray[np.float64],
        opening: float,
    ) -> None:
        """手先の位置・姿勢・指の開きに対応する関節目標を設定する。"""
        if not np.isfinite(target).all() or not np.isfinite(target_rotation).all():
            raise ValueError("手先の目標は有限値である必要があります")
        if not 0 <= opening <= 0.04:
            raise ValueError("指の開きは0〜0.04mである必要があります")
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
        self.gripper_opening = opening
        data.ctrl[7] = opening / 0.04 * 255
