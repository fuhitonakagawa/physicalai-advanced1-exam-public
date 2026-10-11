"""既存Pandaモデルと固定の麺・スープ、可動具材の場面。"""

import math
import xml.etree.ElementTree as ET

import numpy as np
from defusedxml.ElementTree import parse

from src.config.settings import ROOT

BOWL_X = 0.50
BOWL_Y = 0.0
SOUP_HEIGHT = 0.07
BOWL_INNER_RIM_RADIUS = 0.114 - 0.008
FOOD_NAMES = ("egg", "fish_cake")
WORLD_OFFSET = np.array([-0.56, 0.0, 0.80])
INITIAL_ARM_POSITION = np.array(
    [0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.22675220, math.pi / 4]
)


def add_bowl(world: ET.Element) -> None:
    """底と側面を持つ器に、固定のスープ面と麺を配置する。"""
    bowl = ET.SubElement(world, "body", name="ramen_bowl", pos=f"{BOWL_X} {BOWL_Y} 0")
    ET.SubElement(
        bowl,
        "geom",
        type="cylinder",
        size="0.087 0.014",
        pos="0 0 0.02",
        rgba="0.85 0.85 0.78 1",
    )
    for angle in np.linspace(0, 2 * math.pi, 48, endpoint=False):
        c, s = math.cos(angle), math.sin(angle)
        ET.SubElement(
            bowl,
            "geom",
            type="capsule",
            size="0.008",
            fromto=f"{0.082 * c} {0.082 * s} .023 {0.114 * c} {0.114 * s} .105",
            rgba="0.78 0.84 0.80 1",
        )
    ET.SubElement(
        bowl,
        "geom",
        name="soup_support",
        type="cylinder",
        size="0.095 0.018",
        pos="0 0 0.052",
        rgba="0.63 0.32 0.10 1",
        friction="1.5 0.01 0.001",
    )
    for strand in range(9):
        points = [
            (
                0.065 * math.sin(t),
                -0.065 + strand * 0.016 + 0.004 * math.sin(t * 3),
                0.073,
            )
            for t in np.linspace(-math.pi / 2, math.pi / 2, 15)
        ]
        for start, end in zip(points[:-1], points[1:], strict=True):
            ET.SubElement(
                bowl,
                "geom",
                type="capsule",
                size="0.0018",
                contype="0",
                conaffinity="0",
                rgba="0.93 0.75 0.28 1",
                fromto=" ".join(str(v) for v in (*start, *end)),
            )


def add_trays(world: ET.Element) -> None:
    """左右の固定トレーを配置する。"""
    for name, y in (("egg", -0.225), ("fish_cake", 0.225)):
        ET.SubElement(
            world,
            "geom",
            name=f"{name}_tray",
            type="box",
            size=".09 .065 .008",
            pos=f".42 {y} .008",
            rgba=".84 .57 .35 1",
        )


def scene_xml() -> str:
    """Pandaの物理モデルとラーメンの場面を結合したMJCFを返す。"""
    asset_root = ROOT / "assets/franka_emika_panda"
    tree = parse(asset_root / "panda.xml")
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        raise ValueError("Pandaモデルのcompiler設定がありません")
    compiler.set("meshdir", str(asset_root / "assets"))
    option = root.find("option")
    if option is not None:
        option.set("timestep", "0.002")
        option.set("cone", "elliptic")
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Pandaモデルの構成が不正です")
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        size=".65 .52 .025",
        pos=".24 0 -.025",
        rgba=".58 .40 .27 1",
    )
    add_bowl(world)
    add_trays(world)
    for child in world:
        if child.tag in ("body", "geom", "light"):
            position = np.fromstring(child.get("pos", "0 0 0"), sep=" ") + WORLD_OFFSET
            child.set("pos", " ".join(map(str, position)))
    return ET.tostring(root, encoding="unicode")
