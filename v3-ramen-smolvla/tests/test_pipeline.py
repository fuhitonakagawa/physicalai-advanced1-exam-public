"""GPU指定時の明示的な停止。"""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

import src.pipeline as pipeline
from src.environment.ramen import RamenEnvironment


def test_unavailable_gpu_never_falls_back_to_cpu(
    environment: RamenEnvironment, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GPUが利用不可なら、CPU推論を開始せず未評価と記録する。"""
    settings = environment.settings.model_copy(
        update={"device": "mps", "output_dir": tmp_path}
    )
    model_factory = Mock()
    monkeypatch.setattr(pipeline, "SmolVlaPolicy", model_factory)
    monkeypatch.setattr(pipeline, "device_report", lambda: {"mps_available": False})
    monkeypatch.setattr(
        pipeline, "require_device", Mock(side_effect=RuntimeError("GPU unavailable"))
    )
    with pytest.raises(RuntimeError, match="GPU unavailable"):
        pipeline.run(settings)
    model_factory.assert_not_called()
    report = json.loads(next(tmp_path.glob("*/results.json")).read_text())
    assert report["physical_task_evaluated"] is False
    assert report["success"] is None
    assert report["device"] == "mps"
