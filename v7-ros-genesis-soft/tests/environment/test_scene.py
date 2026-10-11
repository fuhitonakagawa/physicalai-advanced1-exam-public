"""Genesisへ渡すMJCFの構造と盛り付け判定。"""

import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest

from src.environment.genesis_scene import entity_xml
from src.environment.geometry import grasp_position


@pytest.mark.parametrize("kind", ["robot", "static"])
def test_mjcf_is_valid(kind: str) -> None:
    """切り分けたモデルが参照切れなくMJCFとして読み込める。"""
    xml = entity_xml(kind)
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    assert model.ngeom > 0
    assert model.nu == 0
    assert xml.find("actuator") is None


def test_grasp_point_down() -> None:
    """下向きの手首から把持点を97mm下に置く。"""
    assert np.allclose(grasp_position((0, 0, 1.1), (0, 1, 0, 0)), (0, 0, 1.003))
