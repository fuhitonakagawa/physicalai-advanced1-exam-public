"""ROSトピック経由でGenesisを一周期ずつ進める制御境界。"""

import time
from collections.abc import Callable
from uuid import uuid4

from src.experiment.messages import (
    Command,
    FoodName,
    ObservationMessage,
    Quaternion,
    Vector3,
)
from src.ros.bridge import RosBridge

Record = Callable[[ObservationMessage, Command], None]


class RemoteEnvironment:
    """指令と応答を対応付け、未受信のまま次の動作へ進まない。"""

    def __init__(self, bridge: RosBridge, prefix: str, record: Record) -> None:
        """ROSトピックと記録先を用意する。"""
        self.bridge = bridge
        self.command_topic = prefix + "/command"
        self.observation_topic = prefix + "/observation"
        self.record = record
        bridge.subscribe(self.observation_topic)
        bridge.advertise(self.command_topic, "std_msgs/msg/String")
        bridge.wait_subscriber(self.command_topic)
        self.current = self.exchange(
            Command(
                request_id=uuid4().hex,
                operation="observe",
                position=(0, 0, 0),
                quaternion=(1, 0, 0, 0),
                opening=0.04,
                phase="observe",
                food="egg",
            )
        )

    def exchange(self, command: Command) -> ObservationMessage:
        """要求IDが一致する観測を受信してから返す。"""
        self.bridge.publish(
            self.command_topic, {"data": command.model_dump_json(by_alias=True)}
        )
        deadline = time.monotonic() + self.bridge.timeout
        while True:
            raw = self.bridge.receive_data(self.observation_topic, deadline)
            result = ObservationMessage.model_validate_json(raw)
            if result.request_id == command.request_id:
                self.record(result, command)
                return result

    def step(
        self,
        position: Vector3,
        quaternion: Quaternion,
        opening: float,
        phase: str,
        food: FoodName,
    ) -> ObservationMessage:
        """Genesisに手先目標を送信し、50ms後の観測を得る。"""
        command = Command(
            request_id=uuid4().hex,
            operation="step",
            position=position,
            quaternion=quaternion,
            opening=opening,
            phase=phase,
            food=food,
        )
        result = self.exchange(command)
        if result.steps != self.current.steps + 1:
            raise RuntimeError("ROSの応答に対応する制御ステップが不連続です")
        self.current = result
        return result

    def finish(self) -> None:
        """シミュレータへ終了要求を送り、最後の状態を受け取る。"""
        self.exchange(
            Command(
                request_id=uuid4().hex,
                operation="finish",
                position=(0, 0, 0),
                quaternion=(1, 0, 0, 0),
                opening=0.04,
                phase="finish",
                food="fish_cake",
            )
        )
