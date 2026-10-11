"""GenesisとROSを使うシミュレーション・推論・接続診断。"""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import structlog
from pydantic import TypeAdapter

from src.config.settings import Settings
from src.experiment.assisted import AssistedRunner, GraspTask
from src.experiment.images import decode_image
from src.experiment.messages import Command, ObservationMessage
from src.experiment.types import Observation
from src.recording import EpisodeRecorder
from src.ros.bridge import RosBridge
from src.ros.remote import RemoteEnvironment

logger = structlog.wrap_logger(logging.getLogger(__name__))


def write_json(path: Path, value: object) -> None:
    """実測値をJSONで保存する。"""
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_tasks(path: Path) -> list[GraspTask]:
    """一回ずつ実行する対象と指示を読み込む。"""
    tasks = TypeAdapter(list[GraspTask]).validate_json(path.read_text(encoding="utf-8"))
    if not 1 <= len(tasks) <= 2 or len({task.food for task in tasks}) != len(tasks):
        raise ValueError("対象は重複しない1〜2種類にしてください")
    return tasks


def controller(settings: Settings, directory: Path) -> dict[str, object]:
    """実モデルとROS観測による盛り付けを録画・評価する。"""
    import torch

    from src.policy.smolvla import SmolVlaPolicy

    tasks = load_tasks(settings.task_file)
    torch.manual_seed(settings.seed)
    runner: AssistedRunner | None = None
    last_phase = ""
    with (
        RosBridge(settings.rosbridge_url, settings.ros_timeout_seconds) as bridge,
        EpisodeRecorder(
            directory / "rollout.mp4",
            20,
            "ROS 2 + Genesis soft | SmolVLA: trigger | ASSIST: contact/hold",
            settings.ffmpeg_path,
        ) as recorder,
        (directory / "telemetry.jsonl").open("w", encoding="utf-8") as trace,
    ):

        def record(state: ObservationMessage, command: Command) -> None:
            """同一ステップのカメラとROS往復結果を保存する。"""
            nonlocal last_phase
            if command.operation == "finish":
                return
            observation = Observation(
                decode_image(state.main_png),
                decode_image(state.wrist_png),
                np.asarray(state.state, dtype=np.float32),
            )
            recorder.record(observation, state.steps, command.phase, command.food)
            diagnostic = state.model_dump(exclude={"main_png", "wrist_png"})
            diagnostic["command"] = command.model_dump()
            diagnostic["raw_model_action"] = (
                runner.last_prediction
                if runner and command.phase.startswith("MODEL")
                else None
            )
            trace.write(json.dumps(diagnostic, ensure_ascii=False) + "\n")
            if last_phase != command.phase:
                logger.info("phase_changed", phase=command.phase)
                last_phase = command.phase

        remote = RemoteEnvironment(bridge, settings.topic_prefix, record)
        logger.info("model_loading")
        policy = SmolVlaPolicy(settings) if settings.policy_mode == "smolvla" else None
        logger.info("model_ready")
        runner = AssistedRunner(remote, policy, settings)
        results = [runner.run_task(task) for task in tasks]
        state = remote.current
        remote.finish()
        return {
            "status": "evaluated",
            "simulator": "genesis",
            "genesis_backend": settings.genesis_backend,
            "ros_connected": True,
            "policy_device": policy.actual_device if policy else None,
            "model_id": settings.model_id,
            "model_revision": settings.model_revision,
            "trained_for_this_task": False,
            "model_outputs_used": ["closure_trigger"] if policy else [],
            "policy_evaluated": policy is not None,
            "soft_model": "FEM_linear_corotated",
            "gripper_control": "contact_assisted",
            "success": all(result.success for result in results)
            and all(state.ingredients[task.food].stable_steps >= 10 for task in tasks),
            "results": [asdict(result) for result in results],
            "ingredients": {
                name: value.model_dump() for name, value in state.ingredients.items()
            },
        }


def physics_check(settings: Settings) -> dict[str, object]:
    """画像と方策を使わず、Genesisの物理状態だけを検証する。"""
    from src.environment.genesis_scene import GenesisEnvironment

    environment = GenesisEnvironment(settings, cameras=False)
    try:
        state = environment.observation("initial")
        target = (state.state[0], state.state[1], state.state[2])
        for index in range(20):
            environment.advance(
                Command(
                    request_id=str(index),
                    operation="step",
                    position=target,
                    quaternion=state.hand_quaternion,
                    opening=0.04,
                    phase="physics_check",
                    food="egg",
                )
            )
        result = environment.observation("final")
        return {
            "status": "physics_only",
            "simulator": "genesis",
            "ros_connected": False,
            "policy_evaluated": False,
            "serving_evaluated": False,
            "state": result.model_dump(exclude={"main_png", "wrist_png"}),
        }
    finally:
        environment.close()


def run(settings: Settings) -> Path:
    """指定役割を実行し、成功・失敗を区別して記録する。"""
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(correlation_id=uuid4().hex)
    timestamp = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%d-%H%M%S")
    directory = settings.output_dir / timestamp / settings.role
    directory.mkdir(parents=True)
    write_json(
        directory / "settings.json",
        settings.model_dump(
            mode="json",
            exclude={
                "hf_home",
                "genesis_cache_dir",
                "output_dir",
                "task_file",
                "ffmpeg_path",
            },
        ),
    )
    logger.info("run_started", mode=settings.role)
    try:
        if settings.role == "controller":
            result = controller(settings, directory)
        elif settings.role == "simulation":
            from src.ros.simulation import serve

            result = serve(settings, directory)
        elif settings.role == "probe":
            with RosBridge(
                settings.rosbridge_url, settings.ros_timeout_seconds
            ) as bridge:
                result = {
                    "status": "ros_probe",
                    "ros_connected": True,
                    "graph": bridge.service("/rosapi/topics", {}),
                }
        else:
            result = physics_check(settings)
    except Exception as error:
        write_json(
            directory / "results.json",
            {"status": "error", "error_type": type(error).__name__},
        )
        raise
    write_json(directory / "results.json", result)
    logger.info("run_finished", success=result.get("success"))
    return directory
