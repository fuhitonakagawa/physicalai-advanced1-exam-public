"""柔軟物の機械動作と制御契約のテスト設定。"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from dotenv import dotenv_values

from src.config.settings import ROOT, Settings
from src.environment.ramen import RamenEnvironment
from src.experiment.messages import FoodState, ObservationMessage


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """設定例から診断用の必須条件を作る。"""
    values = dotenv_values(ROOT / ".env.example")
    data = {name: values[name.upper()] for name in Settings.model_fields}
    data.update(device="cpu", renderer="ray", policy_mode="mechanics")
    data["task_file"] = str(ROOT / "input/tasks.json")
    data["output_dir"] = str(tmp_path)
    return Settings.model_validate(data)


@pytest.fixture
def environment(settings: Settings) -> Iterator[RamenEnvironment]:
    """画像と方策を使わない実物理環境を返す。"""
    instance = RamenEnvironment(settings, images=False)
    yield instance
    instance.close()


@pytest.fixture
def observation() -> ObservationMessage:
    """実測結果と区別する制御テスト用の観測。"""
    return ObservationMessage(
        request_id="test",
        steps=0,
        state=[0, 0, 1.1, 0, 0, 0, 0.04, -0.04],
        hand_quaternion=(0, 1, 0, 0),
        joint_positions=[0] * 9,
        ingredients={
            name: FoodState(
                position=(0, 0, 0.84),
                speed=0,
                finger_contacts=False,
                stable_steps=0,
                bounds_min=(-0.02, -0.02, 0.825),
                bounds_max=(0.02, 0.02, 0.855),
                grasp_quaternion=(0, 1, 0, 0),
                half_width=0.02,
                left_load=0,
                right_load=0,
                deformation_rms=0,
                vertex_count=8,
                contact_kind="flex_contact_force",
            )
            for name in ("egg", "fish_cake")
        },
        main_png=None,
        wrist_png=None,
    )
