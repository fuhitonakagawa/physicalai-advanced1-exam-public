"""卵・なるとの弾性メッシュと表面模様を生成する。"""

import math
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def build(name: str, kind: str, spacing: str) -> None:
    """体積メッシュ、表面メッシュ、テクスチャを保存する。"""
    xml = f'''<mujoco><option integrator="discrete"/><worldbody>
    <flexcomp name="{name}" type="{kind}" dim="3" dof="trilinear"
    radius=".0005" count="5 5 5" spacing="{spacing}" mass=".035">
    <elasticity young="12000" poisson=".3" damping=".002"/>
    <contact selfcollide="none"/></flexcomp></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    vertices = data.flexvert_xpos.copy()
    elements = model.flex_elem.reshape(-1, 4).copy()
    faces = model.flex_shell.reshape(-1, 3).copy()
    normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    inward = np.einsum("ij,ij->i", normals, vertices[faces].mean(axis=1)) < 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    uv = 0.5 + vertices[:, :2] / (2 * np.max(np.abs(vertices[:, :2]), axis=0))
    folder = ROOT / "assets/soft_food"
    folder.mkdir(exist_ok=True)
    np.savez_compressed(
        folder / f"{name}.npz", vertices=vertices, elements=elements, faces=faces, uv=uv
    )
    picture = Image.new("RGB", (256, 256), (248, 238, 218))
    draw = ImageDraw.Draw(picture)
    if name == "egg":
        draw.ellipse((75, 75, 181, 181), fill=(255, 166, 5))
    else:
        points = [
            (
                128 + (5 + 68 * t / (4 * math.pi)) * math.cos(t),
                128 + (5 + 68 * t / (4 * math.pi)) * math.sin(t),
            )
            for t in np.linspace(0, 4 * math.pi, 180)
        ]
        draw.line(points, fill=(235, 87, 133), width=12)
    picture.save(folder / f"{name}.png")
    (folder / f"{name}.mtl").write_text(f"newmtl food\nKd 1 1 1\nmap_Kd {name}.png\n")
    lines = [f"mtllib {name}.mtl", "usemtl food"]
    lines += ["v " + " ".join(map(str, v)) for v in vertices]
    lines += ["vt " + " ".join(map(str, t)) for t in uv]
    lines += ["f " + " ".join(f"{i + 1}/{i + 1}" for i in face) for face in faces]
    (folder / f"{name}.obj").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    build("egg", "ellipsoid", ".0135 .010 .009")
    build("fish_cake", "cylinder", ".0125 .0125 .0065")
