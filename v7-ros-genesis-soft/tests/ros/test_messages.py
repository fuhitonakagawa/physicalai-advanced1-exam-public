"""不正なROS指令と不完全な観測の拒否。"""

import math

import pytest
from pydantic import ValidationError

from src.config.settings import Settings
from src.experiment.messages import Command, ObservationMessage


def command() -> dict[str, object]:
    """作業範囲内の一つの指令を返す。"""
    return dict(
        request_id="id",
        operation="step",
        position=(0, 0, 1),
        quaternion=(1, 0, 0, 0),
        opening=0.04,
        phase="RULE: approach",
        food="egg",
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"position": (math.nan, 0, 1)},
        {"position": (0, 0, 0.2)},
        {"quaternion": (0, 0, 0, 0)},
        {"opening": -0.01},
        {"food": "unknown"},
    ],
)
def test_invalid_motion(changes: dict[str, object]) -> None:
    """異常な数値や作業範囲外の指令を実行対象にしない。"""
    with pytest.raises(ValidationError):
        Command.model_validate(command() | changes)


def test_camel_case_wire() -> None:
    """ROS内のJSONをcamelCaseで往復できる。"""
    value = Command.model_validate(command())
    encoded = value.model_dump_json(by_alias=True)
    assert '"requestId"' in encoded
    assert '"request_id"' not in encoded
    assert Command.model_validate_json(encoded) == value


def test_missing_food(observation: ObservationMessage) -> None:
    """片方の具材がない観測を完全な結果として扱わない。"""
    raw = observation.model_dump()
    del raw["ingredients"]["egg"]
    with pytest.raises(ValidationError):
        ObservationMessage.model_validate(raw)


@pytest.mark.parametrize(
    "url", ["ws://example.com:9090", "ws://user:pass@127.0.0.1:9090"]
)
def test_local_bridge_only(settings: Settings, url: str) -> None:
    """公開ネットワークやURL内認証情報を拒否する。"""
    with pytest.raises(ValidationError):
        Settings.model_validate(settings.model_dump() | {"rosbridge_url": url})
