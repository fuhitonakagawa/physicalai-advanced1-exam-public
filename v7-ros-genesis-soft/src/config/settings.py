"""ROS通信とGenesis実験の必須設定。"""

import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    """各プロセスが共有する実験条件。"""

    model_config = ConfigDict(frozen=True)
    role: Literal["simulation", "controller", "probe", "physics_check"]
    genesis_backend: Literal["cpu", "metal"]
    policy_mode: Literal["smolvla", "mechanics"]
    device: Literal["cpu", "mps"]
    show_viewer: bool
    rosbridge_url: str
    ros_timeout_seconds: float = Field(ge=5, le=600)
    topic_prefix: str = Field(pattern=r"^/[a-z][a-z0-9_]*$")
    model_id: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    hf_home: Path
    genesis_cache_dir: Path
    output_dir: Path
    task_file: Path
    ffmpeg_path: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    seed: int = Field(ge=0, le=2**32 - 1)
    young_modulus_pa: float = Field(gt=0)
    poisson_ratio: float = Field(ge=0, lt=0.49)
    elastic_damping: float = Field(ge=0)
    soft_mass_kg: float = Field(gt=0)
    model_proposal_steps: int = Field(ge=1, le=50)
    close_step: float = Field(gt=0, le=0.002)
    grip_load_limit: float = Field(gt=0)
    max_compression: float = Field(gt=0, lt=0.5)
    max_attempts: int = Field(ge=1, le=3)
    settle_steps: int = Field(ge=10, le=100)
    max_deformation_rms: float = Field(gt=0, le=0.02)
    action_steps: int = Field(ge=1, le=50)
    denoising_steps: int = Field(ge=1, le=100)
    grasp_max_steps: int = Field(ge=1, le=400)
    rule_move_seconds: float = Field(ge=1, le=5)
    approach_clearance: float = Field(ge=0.001, le=0.01)
    max_simulation_steps: int = Field(ge=100, le=5000)

    @field_validator("rosbridge_url")
    @classmethod
    def local_bridge(cls, value: str) -> str:
        """ローカルのROSブリッジだけを接続先として認める。"""
        parsed = urlparse(value)
        if parsed.scheme != "ws" or parsed.hostname not in ("127.0.0.1", "localhost"):
            raise ValueError("ROSブリッジはローカルのws接続で指定してください")
        if parsed.username or parsed.password or not parsed.port:
            raise ValueError("認証情報を含めず、ポートを指定してください")
        return value


def load_settings() -> Settings:
    """必須設定を読み、キャッシュと入力の場所を確定する。"""
    load_dotenv(ROOT / ".env")
    values: dict[str, object] = {}
    for field in Settings.model_fields:
        value = os.environ.get(field.upper(), "").strip()
        if not value:
            raise ValueError(f"必須設定がありません: {field.upper()}")
        values[field] = value
    for field in (
        "hf_home",
        "genesis_cache_dir",
        "output_dir",
        "task_file",
        "ffmpeg_path",
    ):
        values[field] = (ROOT / str(values[field])).resolve()
    settings = Settings.model_validate(values)
    if not settings.task_file.is_file() or not settings.ffmpeg_path.is_file():
        raise ValueError("タスクファイルまたはffmpegがありません")
    os.environ["HF_HOME"] = str(settings.hf_home)
    os.environ["GS_CACHE_FILE_PATH"] = str(settings.genesis_cache_dir)
    os.environ["QD_OFFLINE_CACHE_FILE_PATH"] = str(
        settings.genesis_cache_dir / "quadrants"
    )
    os.environ["MPLCONFIGDIR"] = str(settings.genesis_cache_dir / "matplotlib")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return settings
