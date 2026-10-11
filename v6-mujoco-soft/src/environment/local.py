"""MuJoCoの一周期制御を工程制御へ公開する。"""

from collections.abc import Callable
from uuid import uuid4

from src.environment.ramen import RamenEnvironment
from src.experiment.messages import (
    Command,
    FoodName,
    ObservationMessage,
    Quaternion,
    Vector3,
)

Record = Callable[[ObservationMessage, Command], None]


class LocalEnvironment:
    """物理エンジンの観測をそのまま制御側へ返す。"""

    def __init__(self, environment: RamenEnvironment, record: Record) -> None:
        """物理環境と記録先を保持する。"""
        self.environment = environment
        self.record = record
        self.current = environment.observation("initial")

    def step(
        self,
        position: Vector3,
        quaternion: Quaternion,
        opening: float,
        phase: str,
        food: FoodName,
    ) -> ObservationMessage:
        """手先目標で一周期進め、実際の観測を返す。"""
        command = Command(
            request_id=uuid4().hex,
            operation="step",
            position=position,
            quaternion=quaternion,
            opening=opening,
            phase=phase,
            food=food,
        )
        self.environment.advance(command)
        self.current = self.environment.observation(command.request_id)
        self.record(self.current, command)
        return self.current
