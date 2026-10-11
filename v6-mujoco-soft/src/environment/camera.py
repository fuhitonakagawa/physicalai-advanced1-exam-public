"""MuJoCoのカメラ姿勢に対応する実観測の生成。"""

from typing import Literal

import mujoco
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from src.config.settings import ROOT
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
        self.soft_textures = {}
        self.soft_uv = {}
        self.soft_faces = {}
        for flex_id in range(model.nflex):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_FLEX, flex_id)
            with Image.open(ROOT / f"assets/soft_food/{name}.png") as texture:
                self.soft_textures[flex_id] = np.asarray(texture, dtype=np.uint8).copy()
            with np.load(
                ROOT / f"assets/soft_food/{name}.npz", allow_pickle=False
            ) as mesh:
                self.soft_uv[flex_id] = mesh["uv"].copy()
                self.soft_faces[flex_id] = mesh["faces"].copy()
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
        self.paint_flex(data, data.cam_xpos[index], rays, distances, image, light)
        return np.asarray(np.clip(image, 0, 255), dtype=np.uint8).reshape(256, 256, 3)

    def paint_flex(
        self,
        data: mujoco.MjData,
        origin: NDArray[np.float64],
        rays: NDArray[np.float64],
        distances: NDArray[np.float64],
        pixels: NDArray[np.float64],
        light: NDArray[np.float64],
    ) -> None:
        """変形後の表面三角形と光線の交点を描く。"""
        for flex_id in range(self.model.nflex):
            start = int(self.model.flex_vertadr[flex_id])
            count = int(self.model.flex_vertnum[flex_id])
            points = data.flexvert_xpos[start : start + count]
            center = points.mean(axis=0)
            radius = float(np.max(np.linalg.norm(points - center, axis=1))) + 0.001
            offset = center - origin
            projection = np.einsum("ij,j->i", rays, offset)
            discriminant = projection**2 - float(np.sum(offset * offset)) + radius**2
            root = np.sqrt(np.maximum(discriminant, 0))
            candidates = np.flatnonzero(
                (discriminant >= 0)
                & (projection + root > 0)
                & ((distances < 0) | (projection - root < distances))
            )
            triangles = points[self.soft_faces[flex_id]]
            for begin in range(0, len(candidates), 512):
                indices = candidates[begin : begin + 512]
                self.paint_triangles(
                    flex_id,
                    triangles,
                    origin,
                    rays[indices],
                    indices,
                    distances,
                    pixels,
                    light,
                )

    def paint_triangles(
        self,
        flex_id: int,
        triangles: NDArray[np.float64],
        origin: NDArray[np.float64],
        rays: NDArray[np.float64],
        indices: NDArray[np.int64],
        distances: NDArray[np.float64],
        pixels: NDArray[np.float64],
        light: NDArray[np.float64],
    ) -> None:
        """三角形内のUVと深度からRGBを決める。"""
        edge1 = triangles[:, 1] - triangles[:, 0]
        edge2 = triangles[:, 2] - triangles[:, 0]
        cross = np.cross(rays[:, None, :], edge2[None, :, :])
        determinant = np.einsum("tj,rtj->rt", edge1, cross)
        inverse = np.divide(
            1.0,
            determinant,
            out=np.zeros_like(determinant),
            where=np.abs(determinant) > 1e-12,
        )
        offset = origin - triangles[:, 0]
        u = np.einsum("tj,rtj->rt", offset, cross) * inverse
        q = np.cross(offset, edge1)
        v = np.einsum("rj,tj->rt", rays, q) * inverse
        depth = np.einsum("tj,tj->t", edge2, q)[None, :] * inverse
        valid = (
            (np.abs(determinant) > 1e-12)
            & (u >= 0)
            & (v >= 0)
            & (u + v <= 1)
            & (depth > 0)
        )
        depth = np.where(valid, depth, np.inf)
        faces = np.argmin(depth, axis=1)
        rows = np.arange(len(indices))
        nearest = depth[rows, faces]
        visible = np.isfinite(nearest) & (
            (distances[indices] < 0) | (nearest < distances[indices])
        )
        rows, indices, faces = rows[visible], indices[visible], faces[visible]
        if not len(rows):
            return
        uv = self.soft_uv[flex_id][self.soft_faces[flex_id][faces]]
        a, b = u[rows, faces, None], v[rows, faces, None]
        coordinates = uv[:, 0] * (1 - a - b) + uv[:, 1] * a + uv[:, 2] * b
        coordinates = np.clip((coordinates * 255).astype(int), 0, 255)
        normals = np.cross(edge1[faces], edge2[faces])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        shade = 0.45 + 0.55 * np.maximum(0, np.einsum("ij,j->i", normals, light))
        pixels[indices] = (
            self.soft_textures[flex_id][255 - coordinates[:, 1], coordinates[:, 0]]
            * shade[:, None]
        )
        distances[indices] = nearest[visible]

    def close(self) -> None:
        """描画リソースを閉じる。"""
        if self.native is not None:
            self.native.close()
