"""CPUで検証する物理環境。"""

from collections.abc import Iterator
from pathlib import Path

import pytest

from src.config.settings import ROOT, Settings
from src.environment.ramen import RamenEnvironment


@pytest.fixture
def environment(tmp_path: Path) -> Iterator[RamenEnvironment]:
    """API呼び出しを含まないラーメン環境を作る。"""
    settings = Settings(
        device="cpu",
        renderer="ray",
        model_id="public/model",
        model_revision="a" * 40,
        hf_home=tmp_path,
        output_dir=tmp_path,
        log_level="INFO",
        seed=42,
        max_steps=2,
        action_steps=10,
        denoising_steps=10,
        instruction_file=ROOT / "input/instruction.txt",
        ffmpeg_path=Path("/opt/homebrew/bin/ffmpeg"),
    )
    instance = RamenEnvironment(settings)
    instance.reset(settings.seed, None)
    yield instance
    instance.close()
