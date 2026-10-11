"""柔軟物の接触補助制御と推論提案を区別して評価する。"""

import json
import logging
from contextlib import ExitStack
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import structlog
from pydantic import TypeAdapter

from src.config.settings import Settings
from src.environment.local import LocalEnvironment
from src.environment.ramen import RamenEnvironment
from src.experiment.assisted import AssistedRunner, GraspTask
from src.experiment.images import decode_image
from src.experiment.messages import Command, ObservationMessage
from src.experiment.types import Observation
from src.recording import EpisodeRecorder

logger = structlog.wrap_logger(logging.getLogger(__name__))


def write_json(path: Path, value: object) -> None:
    """実測値をJSONへ保存する。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run(settings: Settings) -> Path:
    """機械診断またはSmolVLAを含む一連の実験を実行する。"""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id=uuid4().hex)
    directory = settings.output_dir / datetime.now(ZoneInfo("Asia/Tokyo")).strftime(
        "%Y%m%d-%H%M%S"
    )
    directory.mkdir(parents=True)
    write_json(
        directory / "settings.json",
        settings.model_dump(
            mode="json", exclude={"hf_home", "output_dir", "task_file", "ffmpeg_path"}
        ),
    )
    tasks = TypeAdapter(list[GraspTask]).validate_json(settings.task_file.read_text())
    if not 1 <= len(tasks) <= 2 or len({task.food for task in tasks}) != len(tasks):
        raise ValueError("対象は重複しない1〜2種類にしてください")
    logger.info("run_started", mode=settings.policy_mode)
    environment = RamenEnvironment(settings, images=settings.policy_mode == "smolvla")
    try:
        with ExitStack() as stack:
            trace = stack.enter_context(
                (directory / "telemetry.jsonl").open("w", encoding="utf-8")
            )
            recorder = (
                stack.enter_context(
                    EpisodeRecorder(
                        directory / "rollout.mp4",
                        20,
                        "MuJoCo soft | SmolVLA: trigger | ASSIST: contact/hold",
                        settings.ffmpeg_path,
                    )
                )
                if settings.policy_mode == "smolvla"
                else None
            )
            runner: AssistedRunner | None = None
            phase = ""

            def record(state: ObservationMessage, command: Command) -> None:
                """指令、実観測、推論提案、変形を記録する。"""
                nonlocal phase
                if recorder:
                    observation = Observation(
                        decode_image(state.main_png),
                        decode_image(state.wrist_png),
                        np.asarray(state.state, dtype=np.float32),
                    )
                    recorder.record(
                        observation, state.steps, command.phase, command.food
                    )
                item = state.model_dump(exclude={"main_png", "wrist_png"})
                item["command"] = command.model_dump()
                item["raw_model_action"] = (
                    runner.last_prediction
                    if runner and command.phase.startswith("MODEL")
                    else None
                )
                trace.write(json.dumps(item, ensure_ascii=False) + "\n")
                if phase != command.phase:
                    logger.info("phase_changed", phase=command.phase)
                    phase = command.phase

            local = LocalEnvironment(environment, record)
            policy = None
            if settings.policy_mode == "smolvla":
                import torch

                from src.policy.smolvla import SmolVlaPolicy

                torch.manual_seed(settings.seed)
                logger.info("model_loading")
                policy = SmolVlaPolicy(settings)
                logger.info("model_ready")
            runner = AssistedRunner(local, policy, settings)
            results = []
            for task in tasks:
                result = runner.run_task(task)
                results.append(result)
                state = local.current
                write_json(
                    directory / "results.json",
                    {
                        "simulator": "mujoco",
                        "soft_model": "flex_trilinear_elasticity",
                        "policy_evaluated": policy is not None,
                        "policy_device": policy.actual_device if policy else None,
                        "model_id": settings.model_id,
                        "model_revision": settings.model_revision,
                        "model_outputs_used": ["closure_trigger"] if policy else [],
                        "gripper_control": "contact_assisted",
                        "seed": settings.seed,
                        "episode_complete": len(results) == len(tasks),
                        "success": len(results) == len(tasks)
                        and all(item.success for item in results)
                        and all(
                            state.ingredients[task.food].stable_steps >= 10
                            for task in tasks
                        ),
                        "results": [asdict(item) for item in results],
                        "ingredients": {
                            name: value.model_dump()
                            for name, value in state.ingredients.items()
                        },
                        "max_deformation_rms": environment.max_deformation,
                    },
                )
                logger.info(
                    "episode_finished", success=result.success, steps=state.steps
                )
    finally:
        environment.close()
    logger.info("run_finished", success=all(item.success for item in results))
    return directory
