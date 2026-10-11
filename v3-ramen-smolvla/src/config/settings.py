"""ラーメン盛り付け実験の必須設定。"""

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseModel):
    """装置、描画、モデル、試行条件。"""

    model_config = ConfigDict(frozen=True)
    device: Literal["mps", "cpu"]
    renderer: Literal["native", "ray"]
    model_id: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    hf_home: Path
    output_dir: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    seed: int = Field(ge=0, le=2**32 - 1)
    max_steps: int = Field(ge=1, le=2000)
    action_steps: int = Field(ge=1, le=50)
    denoising_steps: int = Field(ge=1, le=100)
    instruction_file: Path
    ffmpeg_path: Path


def load_settings() -> Settings:
    """未設定の環境変数を拒否し、モデルのキャッシュ先を定める。"""
    load_dotenv(ROOT / ".env")
    values: dict[str, object] = {}
    for field in Settings.model_fields:
        value = os.environ.get(field.upper(), "").strip()
        if not value:
            raise ValueError(f"必須設定がありません: {field.upper()}")
        values[field] = value
    for field in ("hf_home", "output_dir", "instruction_file", "ffmpeg_path"):
        values[field] = (ROOT / str(values[field])).resolve()
    settings = Settings.model_validate(values)
    if not settings.instruction_file.is_file():
        raise ValueError("指示ファイルがありません")
    os.environ["HF_HOME"] = str(settings.hf_home)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    return settings
