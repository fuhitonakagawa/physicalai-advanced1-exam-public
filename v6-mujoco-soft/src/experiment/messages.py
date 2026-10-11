"""ROSトピック内で送受信する指令と観測。"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.alias_generators import to_camel

Vector3 = tuple[float, float, float]
Quaternion = tuple[float, float, float, float]
FoodName = Literal["egg", "fish_cake"]


class Message(BaseModel):
    """JSON境界のcamelCaseと有限値を保証する。"""

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        allow_inf_nan=False,
    )


class Command(Message):
    """観測要求、手先目標、または終了要求。"""

    request_id: str = Field(min_length=1, max_length=80)
    operation: Literal["observe", "step", "finish"]
    position: Vector3
    quaternion: Quaternion
    opening: float = Field(ge=0, le=0.04)
    phase: str = Field(min_length=1, max_length=64)
    food: FoodName

    @model_validator(mode="after")
    def valid_pose(self) -> "Command":
        """単位姿勢と実験台付近の動作範囲を検証する。"""
        norm = sum(value * value for value in self.quaternion)
        if abs(norm - 1) > 1e-3:
            raise ValueError("クォータニオンは単位長にしてください")
        if self.operation == "step":
            x, y, z = self.position
            if not (-0.5 <= x <= 0.35 and abs(y) <= 0.5 and 0.8 <= z <= 1.45):
                raise ValueError("手先目標が作業範囲外です")
        return self


class FoodState(Message):
    """具材の実位置と物理的な保持・配置判定。"""

    position: Vector3
    speed: float = Field(ge=0)
    finger_contacts: bool
    stable_steps: int = Field(ge=0)
    bounds_min: Vector3
    bounds_max: Vector3
    grasp_quaternion: Quaternion
    half_width: float = Field(gt=0)
    left_load: float = Field(ge=0)
    right_load: float = Field(ge=0)
    deformation_rms: float = Field(ge=0)
    vertex_count: int = Field(ge=4)
    contact_kind: Literal["flex_contact_force", "vertex_proximity_and_drive_force"]


class ObservationMessage(Message):
    """一つの制御ステップに対応する画像と状態。"""

    request_id: str
    steps: int = Field(ge=0)
    state: list[float] = Field(min_length=8, max_length=8)
    hand_quaternion: Quaternion
    joint_positions: list[float] = Field(min_length=9, max_length=9)
    ingredients: dict[str, FoodState]
    main_png: str | None
    wrist_png: str | None

    @model_validator(mode="after")
    def both_foods(self) -> "ObservationMessage":
        """左右両方の具材を含む観測だけを受け付ける。"""
        if set(self.ingredients) != {"egg", "fish_cake"}:
            raise ValueError("卵となるとの両方の状態が必要です")
        return self
