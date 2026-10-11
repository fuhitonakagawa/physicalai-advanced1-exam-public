"""カメラ観測からSmolVLAを駆動し、盛り付け結果を記録する。"""

import ctypes
import json
import logging
import platform
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import structlog
import torch
from PIL import Image

from src.config.settings import Settings
from src.environment.ramen import RamenEnvironment
from src.experiment.runner import run_episode
from src.experiment.types import Observation
from src.policy.smolvla import SmolVlaPolicy, policy_image, require_device
from src.recording import EpisodeRecorder

logger = structlog.wrap_logger(logging.getLogger(__name__))


def write_json(path: Path, value: object) -> None:
    """実測結果をJSONで保存する。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def device_report() -> dict[str, object]:
    """このプロセスに公開されたGPUの状態を返す。"""
    metal_visible = False
    if platform.system() == "Darwin":
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        metal_visible = bool(metal.MTLCreateSystemDefaultDevice())
    return {
        "os": platform.system(),
        "macos_version": platform.mac_ver()[0],
        "torch_version": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "metal_device_visible": metal_visible,
    }


def run(settings: Settings) -> Path:
    """指定装置で一つの盛り付け試行を最後まで評価する。"""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id=uuid4().hex)
    directory = settings.output_dir / datetime.now(ZoneInfo("Asia/Tokyo")).strftime(
        "%Y%m%d-%H%M%S"
    )
    directory.mkdir(parents=True)
    write_json(directory / "device.json", device_report())
    write_json(
        directory / "settings.json",
        settings.model_dump(
            mode="json",
            exclude={"hf_home", "output_dir", "instruction_file", "ffmpeg_path"},
        ),
    )
    logger.info("run_started", mode=settings.renderer)
    try:
        require_device(settings.device)
    except RuntimeError:
        write_json(
            directory / "results.json",
            {
                "status": "device_unavailable",
                "device": settings.device,
                "physical_task_evaluated": False,
                "success": None,
            },
        )
        raise
    instruction = settings.instruction_file.read_text(encoding="utf-8").strip()
    if not 1 <= len(instruction) <= 400:
        raise ValueError("指示は1〜400文字で指定してください")
    torch.manual_seed(settings.seed)
    environment = RamenEnvironment(settings)
    try:
        logger.info("model_loading")
        policy = SmolVlaPolicy(settings)
        logger.info("model_ready")
        with (
            (directory / "telemetry.jsonl").open("w", encoding="utf-8") as telemetry,
            EpisodeRecorder(
                directory / "rollout.mp4", 20, 1, instruction, settings.ffmpeg_path
            ) as recorder,
        ):

            def record(observation: Observation, step: int) -> None:
                """実観測を保存し、途中の具材位置を記録する。"""
                recorder.record(observation, step)
                diagnostic = environment.diagnostics()
                diagnostic.update(
                    {
                        "steps": step,
                        "state": observation.state.tolist(),
                        "wrist_black_fraction": float(
                            np.mean(np.max(observation.wrist_image, axis=2) < 8)
                        ),
                    }
                )
                telemetry.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")
                if step == 0:
                    pixels = np.concatenate(
                        [
                            policy_image(observation.main_image),
                            policy_image(observation.wrist_image),
                        ],
                        axis=1,
                    )
                    Image.fromarray(pixels).save(directory / "model_input.png")
                if step % 50 == 0:
                    write_json(
                        directory / "progress.json",
                        {"steps": step, "ingredients": environment.serving_results()},
                    )
                    logger.info("episode_progress", steps=step)

            result = run_episode(
                environment,
                policy,
                instruction,
                settings.seed,
                None,
                settings.max_steps,
                record,
            )
        write_json(
            directory / "results.json",
            {
                "status": "evaluated",
                "physical_task_evaluated": True,
                "device": settings.device,
                "parameter_device": policy.actual_device,
                "renderer": settings.renderer,
                "model_id": settings.model_id,
                "model_revision": settings.model_revision,
                "instruction": instruction,
                "trained_for_this_task": False,
                "result": asdict(result),
                "ingredients": environment.serving_results(),
            },
        )
        logger.info("run_finished", success=result.success, steps=result.steps)
    finally:
        environment.close()
    return directory
