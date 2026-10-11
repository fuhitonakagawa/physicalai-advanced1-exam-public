"""実機やROSサーバーを使わない契約テストの設定。"""

from pathlib import Path

import pytest
from dotenv import dotenv_values

from src.config.settings import ROOT, Settings
from src.experiment.messages import FoodState, ObservationMessage


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """設定例と一時保存先による必須設定を返す。"""
    values = dotenv_values(ROOT / ".env.example")
    data = {name: values[name.upper()] for name in Settings.model_fields}
    data.update(device="cpu", genesis_backend="cpu", role="controller")
    for name in ("hf_home", "genesis_cache_dir", "output_dir"):
        data[name] = str(tmp_path / name)
    data["task_file"] = str(ROOT / "input/tasks.json")
    return Settings.model_validate(data)


@pytest.fixture
def observation() -> ObservationMessage:
    """物理実験の結果と区別する通信テスト用の観測。"""
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
                contact_kind="vertex_proximity_and_drive_force",
            )
            for name in ("egg", "fish_cake")
        },
        main_png=None,
        wrist_png=None,
    )
