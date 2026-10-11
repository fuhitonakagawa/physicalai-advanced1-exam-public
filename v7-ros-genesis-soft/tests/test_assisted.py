"""補助制御の負荷制限と失敗時の運搬禁止。"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from src.config.settings import Settings
from src.experiment.assisted import AssistedRunner, ControlEnvironment, GraspTask
from src.experiment.images import encode_image
from src.experiment.messages import ObservationMessage, Quaternion, Vector3


def test_no_contact_never_transports(
    settings: Settings, observation: ObservationMessage
) -> None:
    """AIの閉鎖提案があっても、接触できなければ運搬しない。"""
    remote = MagicMock(spec=ControlEnvironment)
    pixels = encode_image(np.zeros((256, 256, 3), dtype=np.uint8))
    remote.current = observation.model_copy(
        update={"main_png": pixels, "wrist_png": pixels}
    )
    phases = []

    def step(
        position: Vector3, quaternion: Quaternion, opening: float, phase: str, food: str
    ) -> ObservationMessage:
        """手先だけを更新する制御テスト用の代役。"""
        current: ObservationMessage = remote.current
        current = current.model_copy(
            update={
                "steps": current.steps + 1,
                "state": [*position, 0, 0, 0, opening, -opening],
            }
        )
        remote.current = current
        phases.append(phase)
        return current

    remote.step.side_effect = step
    policy = MagicMock()
    policy.select_action.return_value = np.array(
        [1, 1, 1, 1, 1, 1, 1], dtype=np.float32
    )
    runner = AssistedRunner(
        remote,
        policy,
        settings.model_copy(
            update={"grasp_max_steps": 2, "max_attempts": 2, "rule_move_seconds": 1.0}
        ),
    )
    result = runner.run_task(GraspTask(food="egg", instruction="pick"))
    assert not result.success
    assert result.attempts == 2
    assert result.ai_triggered_closure
    assert result.assisted_steps == 4
    assert "ASSIST: carry and hold" not in phases
    assert phases.count("MODEL: closure proposal") == 2


@pytest.mark.parametrize(
    "changes,reason",
    [
        ({"left_load": 3.0}, "load_limit"),
        ({"deformation_rms": 0.02}, "deformation_limit"),
    ],
)
def test_overload_stops_closing(
    settings: Settings,
    observation: ObservationMessage,
    changes: dict[str, float],
    reason: str,
) -> None:
    """過大な負荷・変形がある状態から閉鎖を続けない。"""
    remote = MagicMock(spec=ControlEnvironment)
    foods = dict(observation.ingredients)
    foods["egg"] = foods["egg"].model_copy(update=changes)
    remote.current = observation.model_copy(update={"ingredients": foods})
    runner = AssistedRunner(remote, None, settings)
    assert runner.close_gripper("egg", (0, 0, 0.84), (0, 1, 0, 0)) == reason
    remote.step.assert_not_called()


def test_proximity_without_lift_never_transports(
    settings: Settings, observation: ObservationMessage
) -> None:
    """左右に近い頂点があっても具材が浮かなければ盛り付けは失敗とする。"""
    remote = MagicMock(spec=ControlEnvironment)
    foods = dict(observation.ingredients)
    foods["egg"] = foods["egg"].model_copy(update={"finger_contacts": True})
    remote.current = observation.model_copy(update={"ingredients": foods})
    phases: list[str] = []

    def step(
        position: Vector3, quaternion: Quaternion, opening: float, phase: str, food: str
    ) -> ObservationMessage:
        """具材がトレーに残り、手先だけが移動する観測を返す。"""
        current: ObservationMessage = remote.current
        updated = current.model_copy(
            update={"state": [*position, 0, 0, 0, opening, -opening]}
        )
        remote.current = updated
        phases.append(phase)
        return updated

    remote.step.side_effect = step
    runner = AssistedRunner(
        remote, None, settings.model_copy(update={"max_attempts": 1})
    )
    result = runner.run_task(GraspTask(food="egg", instruction="pick"))
    assert not result.success
    assert result.reason == "lift_failed"
    assert "ASSIST: lift check" in phases
    assert "ASSIST: carry and hold" not in phases
