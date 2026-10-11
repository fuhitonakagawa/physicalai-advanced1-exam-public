"""固定リビジョンのSmolVLAと付属の正規化器。"""

import contextlib
import io
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

from src.config.settings import Settings
from src.experiment.types import FloatArray, ImageArray, Observation


def policy_image(image: ImageArray) -> ImageArray:
    """上原点の表示画像をLIBERO方策の画像座標へ変換する。"""
    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
        raise ValueError("入力画像は256×256のRGB画像である必要があります")
    return np.flip(image, axis=1).copy()


def require_device(device: str) -> None:
    """要求された計算装置の利用可否を検証する。"""
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("このプロセスからMetal GPUを利用できません")


class SmolVlaPolicy:
    """画像と状態を継続観測して動作を生成する方策。"""

    def __init__(self, settings: Settings) -> None:
        """公開済みの重みと前後処理を読み込む。"""
        require_device(settings.device)
        checkpoint = snapshot_download(
            settings.model_id,
            revision=settings.model_revision,
            allow_patterns=["*.json", "*.safetensors"],
        )
        config = SmolVLAConfig.from_pretrained(
            checkpoint,
            cli_overrides=[
                f"--device={settings.device}",
                "--load_vlm_weights=false",
                f"--n_action_steps={settings.action_steps}",
                f"--num_steps={settings.denoising_steps}",
            ],
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.model = SmolVLAPolicy.from_pretrained(
                checkpoint, config=config, strict=True
            )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            checkpoint,
            preprocessor_overrides={"device_processor": {"device": settings.device}},
        )
        self.parameter_count = sum(
            parameter.numel() for parameter in self.model.parameters()
        )
        self.actual_device = str(next(self.model.parameters()).device)

    def reset(self) -> None:
        """前の試行の動作列を破棄する。"""
        self.model.reset()
        self.preprocessor.reset()
        self.postprocessor.reset()

    def select_action(self, observation: Observation, instruction: str) -> FloatArray:
        """RGB画像と8次元状態から実単位の7次元動作を得る。"""
        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(observation.state.copy()),
            "task": instruction,
        }
        for name, image in (
            ("image", observation.main_image),
            ("image2", observation.wrist_image),
        ):
            batch[f"observation.images.{name}"] = (
                torch.from_numpy(policy_image(image)).permute(2, 0, 1).float() / 255.0
            )
        if observation.state.shape != (8,) or not np.isfinite(observation.state).all():
            raise ValueError("腕の状態は有限な8次元ベクトルである必要があります")
        with torch.inference_mode():
            action = self.model.select_action(self.preprocessor(batch))
            output = self.postprocessor(action)
        return np.asarray(output.detach().cpu().numpy()[0], dtype=np.float32)
