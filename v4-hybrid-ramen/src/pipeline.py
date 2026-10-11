"""ルール移動と推論把持の物理実験。"""

import ctypes
import json
import logging
import platform
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import structlog
import torch
from pydantic import TypeAdapter

from src.config.settings import Settings
from src.environment.ramen import RamenEnvironment
from src.experiment.hybrid import GraspTask, HybridRunner
from src.experiment.types import Observation
from src.policy.smolvla import SmolVlaPolicy, require_device
from src.recording import EpisodeRecorder

logger = structlog.wrap_logger(logging.getLogger(__name__))


def write_json(path: Path, value: object) -> None:
    """実測結果をJSONで保存する。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def device_report() -> dict[str, object]:
    """このプロセスのGPU利用可否を返す。"""
    metal_visible = False
    if platform.system() == "Darwin":
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        metal_visible = bool(metal.MTLCreateSystemDefaultDevice())
    return {
        "device_available": torch.backends.mps.is_available(),
        "metal_device_visible": metal_visible,
        "torch_version": torch.__version__,
    }


def load_tasks(path: Path) -> list[GraspTask]:
    """重複のない対象と把持指示を読み込む。"""
    tasks = TypeAdapter(list[GraspTask]).validate_json(path.read_text(encoding="utf-8"))
    if not 1 <= len(tasks) <= 2 or len({task.food for task in tasks}) != len(tasks):
        raise ValueError("対象は重複なしの1〜2種類で指定してください")
    return tasks


def run(settings: Settings) -> Path:
    """接近から開放までを実行し、段階ごとの結果を保存する。"""
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
            mode="json", exclude={"hf_home", "output_dir", "task_file", "ffmpeg_path"}
        ),
    )
    require_device(settings.device)
    tasks = load_tasks(settings.task_file)
    torch.manual_seed(settings.seed)
    environment = RamenEnvironment(settings)
    try:
        logger.info("model_loading")
        policy = SmolVlaPolicy(settings)
        logger.info("model_ready")
        environment.reset(settings.seed, None)
        results = []
        with (
            (directory / "telemetry.jsonl").open("w", encoding="utf-8") as telemetry,
            EpisodeRecorder(
                directory / "rollout.mp4",
                20,
                "Approach / carry / release: RULE | Grasp: SmolVLA",
                settings.ffmpeg_path,
            ) as recorder,
        ):
            last_phase = ""

            def record(
                observation: Observation, step: int, phase: str, food: str
            ) -> None:
                """各フレームに制御主体と対象を記録する。"""
                nonlocal last_phase
                recorder.record(observation, step, phase, food)
                diagnostic = environment.diagnostics()
                if not phase.startswith("MODEL"):
                    diagnostic["action"] = None
                diagnostic["raw_model_action"] = (
                    runner.last_prediction.tolist()
                    if phase.startswith("MODEL")
                    else None
                )
                diagnostic.update(
                    {
                        "steps": step,
                        "phase": phase,
                        "food": food,
                        "finger_contacts": environment.finger_contacts(food),
                        "gripper_target": environment.controller.gripper_opening,
                    }
                )
                telemetry.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")
                if phase != last_phase:
                    logger.info("phase_changed", phase=phase)
                    last_phase = phase

            runner = HybridRunner(environment, policy, settings, record)
            for task in tasks:
                result = runner.run_task(task)
                results.append(result)
                write_json(
                    directory / "results.json",
                    {
                        "physical_task_evaluated": True,
                        "device": policy.actual_device,
                        "renderer": settings.renderer,
                        "control": "hybrid",
                        "grasp_control": settings.grasp_control,
                        "model_id": settings.model_id,
                        "model_revision": settings.model_revision,
                        "trained_for_this_task": False,
                        "model_outputs_used": ["gripper"]
                        if settings.grasp_control == "gripper"
                        else ["position", "rotation", "gripper"],
                        "tasks": [task.model_dump() for task in tasks],
                        "results": [asdict(item) for item in results],
                        "success": len(results) == len(tasks)
                        and all(item.success for item in results)
                        and all(
                            environment.stable_steps[task.food] >= 10 for task in tasks
                        ),
                        "ingredients": environment.serving_results(),
                    },
                )
                logger.info(
                    "episode_finished",
                    success=result.success,
                    steps=result.inference_steps,
                )
    finally:
        environment.close()
    logger.info(
        "run_finished",
        success=all(item.success for item in results)
        and all(environment.stable_steps[task.food] >= 10 for task in tasks),
    )
    return directory
