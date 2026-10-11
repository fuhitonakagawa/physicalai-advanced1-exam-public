"""ルール制御が推論の把持失敗を成功扱いしない契約。"""

from unittest.mock import Mock

import numpy as np
import pytest

from src.environment.ramen import RamenEnvironment
from src.experiment.hybrid import GraspTask, HybridRunner


def test_no_contact_never_transfers(
    environment: RamenEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """推論把持の失敗時に運搬処理を呼ばない。"""
    runner = HybridRunner(environment, Mock(), environment.settings, Mock())
    patches: dict[str, Mock] = {}
    patches["approach"] = Mock(return_value=np.array([0.0, 0.0, 1.0]))
    monkeypatch.setattr(runner, "approach", patches["approach"])
    patches["grasp"] = Mock(return_value=(False, 2, "grasp_timeout"))
    monkeypatch.setattr(runner, "grasp", patches["grasp"])
    patches["recover"] = Mock()
    monkeypatch.setattr(runner, "recover", patches["recover"])
    patches["verify_lift"] = Mock()
    monkeypatch.setattr(runner, "verify_lift", patches["verify_lift"])
    patches["transfer"] = Mock()
    monkeypatch.setattr(runner, "transfer", patches["transfer"])
    result = runner.run_task(GraspTask(food="egg", instruction="pick up the egg"))
    assert not result.success
    patches["verify_lift"].assert_not_called()
    patches["transfer"].assert_not_called()
    patches["recover"].assert_called_once()


def test_contact_without_lift_never_transfers(
    environment: RamenEnvironment, monkeypatch: pytest.MonkeyPatch
) -> None:
    """両指接触だけでは把持成功と扱わない。"""
    runner = HybridRunner(environment, Mock(), environment.settings, Mock())
    patches: dict[str, Mock] = {}
    patches["approach"] = Mock(return_value=np.array([0.0, 0.0, 1.0]))
    monkeypatch.setattr(runner, "approach", patches["approach"])
    patches["grasp"] = Mock(return_value=(True, 2, "contact_candidate"))
    monkeypatch.setattr(runner, "grasp", patches["grasp"])
    patches["verify_lift"] = Mock(return_value=False)
    monkeypatch.setattr(runner, "verify_lift", patches["verify_lift"])
    patches["recover"] = Mock()
    monkeypatch.setattr(runner, "recover", patches["recover"])
    patches["transfer"] = Mock()
    monkeypatch.setattr(runner, "transfer", patches["transfer"])
    result = runner.run_task(GraspTask(food="egg", instruction="pick up the egg"))
    assert not result.success
    assert result.contact_candidate
    assert not result.lift_verified
    patches["transfer"].assert_not_called()


def test_gripper_mode_does_not_use_model_arm_motion(
    environment: RamenEnvironment,
) -> None:
    """指だけの推論モードでは手先位置を保持し、開く指令を閉じる指令へ変えない。"""
    policy = Mock()
    policy.select_action.return_value = np.array(
        [1, 1, 1, 1, 1, 1, -1], dtype=np.float32
    )
    runner = HybridRunner(environment, policy, environment.settings, Mock())
    anchor = environment.data.site("grasp").xpos.copy()
    candidate, steps, reason = runner.grasp(
        GraspTask(food="egg", instruction="pick"), anchor
    )
    assert not candidate
    assert steps == environment.settings.grasp_max_steps
    assert reason == "grasp_timeout"
    assert np.linalg.norm(environment.data.site("grasp").xpos - anchor) < 0.01
    assert environment.controller.gripper_opening == 0.04
    np.testing.assert_array_equal(environment.controller.last_action[:6], np.zeros(6))
    np.testing.assert_array_equal(runner.last_prediction[:6], np.ones(6))


def test_open_gripper_does_not_grasp_food(environment: RamenEnvironment) -> None:
    """具材がトレー上にあるだけでは把持接触と判定しない。"""
    assert not environment.finger_contacts("egg")
    assert not environment.finger_contacts("fish_cake")
