"""既存Pandaモデルと固定の麺・スープ、可動具材の場面。"""

import math
import xml.etree.ElementTree as ET

import numpy as np
from defusedxml.ElementTree import parse

from src.config.settings import ROOT

BOWL_X = 0.50
BOWL_Y = 0.0
SOUP_HEIGHT = 0.07
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


def add_food(world: ET.Element, seed: int) -> None:
    """左右のトレーに位置・向きが変わる具材を置く。"""
    generator = np.random.default_rng(seed)
    for name, y, color in (
        ("egg", -0.225, "0.84 0.57 0.35 1"),
        ("fish_cake", 0.225, "0.84 0.57 0.35 1"),
    ):
        ET.SubElement(
            world,
            "geom",
            name=f"{name}_tray",
            type="box",
            size="0.09 0.065 0.008",
            pos=f".42 {y} .008",
            rgba=color,
        )
        x = 0.42 + float(generator.uniform(-0.025, 0.025))
        y += float(generator.uniform(-0.015, 0.015))
        angle = float(generator.uniform(-math.pi, math.pi))
        body = ET.SubElement(
            world,
            "body",
            name=name,
            pos=f"{x} {y} .04",
            quat=f"{math.cos(angle / 2)} 0 0 {math.sin(angle / 2)}",
        )
        ET.SubElement(body, "freejoint", name=f"{name}_joint")
        if name == "egg":
            ET.SubElement(
                body,
                "geom",
                type="ellipsoid",
                size=".027 .020 .018",
                rgba=".97 .94 .82 1",
                mass=".035",
                friction="1.5 .01 .001",
            )
            ET.SubElement(
                body,
                "geom",
                type="ellipsoid",
                size=".013 .013 .006",
                pos="0 0 .015",
                rgba="1 .63 .03 1",
                mass="0",
                contype="0",
                conaffinity="0",
            )
        else:
            ET.SubElement(
                body,
                "geom",
                type="cylinder",
                size=".025 .013",
                rgba=".98 .92 .86 1",
                mass=".035",
                friction="1.5 .01 .001",
            )
            for t in np.linspace(0, 4 * math.pi, 36):
                radius = 0.001 + 0.013 * t / (4 * math.pi)
                ET.SubElement(
                    body,
                    "geom",
                    type="sphere",
                    size=".002",
                    mass="0",
                    pos=f"{radius * math.cos(t)} {radius * math.sin(t)} .014",
                    contype="0",
                    conaffinity="0",
                    rgba=".96 .31 .44 1",
                )


def scene_xml(seed: int) -> str:
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
    hand = root.find('.//body[@name="hand"]')
    if world is None or hand is None:
        raise ValueError("Pandaモデルの構成が不正です")
    ET.SubElement(
        hand,
        "site",
        name="grasp",
        pos="0 0 .097",
        quat="0.7071067811865476 0 0 -0.7071067811865476",
        size=".003",
        rgba="0 0 0 0",
    )
    ET.SubElement(
        hand,
        "camera",
        name="wrist",
        pos=".05 0 0",
        quat="0 0.7071067811865476 0.7071067811865476 0",
        fovy="75",
    )
    ET.SubElement(
        world,
        "geom",
        name="table",
        type="box",
        size=".65 .52 .025",
        pos=".24 0 -.025",
        rgba=".58 .40 .27 1",
    )
    ET.SubElement(world, "light", pos=".3 -.5 1.5", dir="0 0 -1", diffuse=".8 .8 .8")
    ET.SubElement(
        world,
        "camera",
        name="agentview",
        pos="0.5886131746834771 0 1.4903500240372423",
        quat=(
            "0.6380177736282349 0.3048497438430786 "
            "0.30484986305236816 0.6380177736282349"
        ),
        fovy="45",
    )
    add_bowl(world)
    add_food(world, seed)
    for child in world:
        if child.tag in ("body", "geom", "light"):
            position = np.fromstring(child.get("pos", "0 0 0"), sep=" ") + WORLD_OFFSET
            child.set("pos", " ".join(map(str, position)))
    return ET.tostring(root, encoding="unicode")
