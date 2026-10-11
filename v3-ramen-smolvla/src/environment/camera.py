"""MuJoCoのカメラ姿勢に対応する実観測の生成。"""

from typing import Literal

import mujoco
import numpy as np
from numpy.typing import NDArray

from src.experiment.types import ImageArray


class Camera:
    """ネイティブ描画または明示的なCPU診断描画。"""

    def __init__(self, model: mujoco.MjModel, kind: Literal["native", "ray"]) -> None:
        """描画バックエンドを用意する。"""
        self.model = model
        self.native = (
            mujoco.Renderer(model, height=256, width=256) if kind == "native" else None
        )
        self.directions: dict[int, NDArray[np.float64]] = {}
        row, column = np.indices((256, 256))
        for index in range(model.ncam):
            focal = 128 / np.tan(np.radians(model.cam_fovy[index]) / 2)
            rays = np.stack(
                [(column - 127.5) / focal, -(row - 127.5) / focal, -np.ones_like(row)],
                axis=-1,
            ).reshape(-1, 3)
            rays /= np.linalg.norm(rays, axis=1, keepdims=True)
            self.directions[index] = rays

    def render(self, data: mujoco.MjData, name: str) -> ImageArray:
        """現在の物理状態に対応するカメラ画像を返す。"""
        if self.native is not None:
            self.native.update_scene(data, camera=name)
            return np.asarray(self.native.render(), dtype=np.uint8)
        index = self.model.camera(name).id
        rays = np.einsum(
            "ij,kj->ik", self.directions[index], data.cam_xmat[index].reshape(3, 3)
        )
        identifiers = np.full(256 * 256, -1, dtype=np.int32)
        distances = np.zeros(256 * 256, dtype=np.float64)
        normals = np.zeros((256 * 256, 3), dtype=np.float64)
        groups = np.array([1, 1, 1, 0, 1, 1], dtype=np.uint8)
        mujoco.mj_multiRay(
            self.model,
            data,
            data.cam_xpos[index],
            rays.flatten(),
            groups,
            True,
            -1,
            identifiers,
            distances,
            normals.reshape(-1),
            256 * 256,
            5.0,
        )
        image = np.tile(np.array([186.0, 201.0, 217.0]), (256 * 256, 1))
        hit = identifiers >= 0
        ids = identifiers[hit]
        colors = self.model.geom_rgba[ids, :3].copy()
        material_ids = self.model.geom_matid[ids]
        material = material_ids >= 0
        colors[material] = self.model.mat_rgba[material_ids[material], :3]
        light = np.array([0.3, -0.4, 0.866])
        brightness = 0.45 + 0.55 * np.maximum(
            0, np.einsum("ij,j->i", normals[hit], light)
        )
        image[hit] = colors * brightness[:, None] * 255
        return np.asarray(np.clip(image, 0, 255), dtype=np.uint8).reshape(256, 256, 3)

    def close(self) -> None:
        """描画リソースを閉じる。"""
        if self.native is not None:
            self.native.close()
