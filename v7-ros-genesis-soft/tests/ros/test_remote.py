"""要求IDと制御ステップの対応を検証する。"""

from unittest.mock import MagicMock

import pytest

from src.experiment.messages import ObservationMessage
from src.ros.bridge import RosBridge
from src.ros.remote import RemoteEnvironment


def test_old_reply_is_not_current(observation: ObservationMessage) -> None:
    """古い要求への応答は新しい制御結果として採用しない。"""
    bridge = MagicMock(spec=RosBridge)
    bridge.timeout = 5
    received: list[ObservationMessage] = []

    def publish(topic: str, message: dict[str, object]) -> None:
        """テスト対象の要求IDに対応する応答を用意する。"""
        from src.experiment.messages import Command

        command = Command.model_validate_json(str(message["data"]))
        reply = observation.model_copy(update={"request_id": command.request_id})
        bridge.receive_data.side_effect = [
            observation.model_dump_json(by_alias=True),
            reply.model_dump_json(by_alias=True),
        ]

    bridge.publish.side_effect = publish
    remote = RemoteEnvironment(
        bridge, "/test", lambda state, command: received.append(state)
    )
    assert remote.current.request_id != observation.request_id
    assert len(received) == 1
    assert bridge.receive_data.call_count == 2


def test_discontinuous_step_rejected(observation: ObservationMessage) -> None:
    """要求した一周期と一致しない観測では進行を止める。"""
    bridge = MagicMock(spec=RosBridge)
    bridge.timeout = 5

    def publish(topic: str, message: dict[str, object]) -> None:
        """一周期を飛ばした応答を返す。"""
        from src.experiment.messages import Command

        command = Command.model_validate_json(str(message["data"]))
        reply = observation.model_copy(
            update={
                "request_id": command.request_id,
                "steps": 2 if command.operation == "step" else 0,
            }
        )
        bridge.receive_data.return_value = reply.model_dump_json(by_alias=True)

    bridge.publish.side_effect = publish
    remote = RemoteEnvironment(bridge, "/test", lambda state, command: None)
    with pytest.raises(RuntimeError, match="不連続"):
        remote.step((0, 0, 1), (1, 0, 0, 0), 0.04, "RULE: approach", "egg")
    assert remote.current.steps == 0
