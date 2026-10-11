"""ROS指令でGenesisを進め、センサーと物理状態を配信する。"""

import time
from pathlib import Path

from src.config.settings import Settings
from src.environment.genesis_scene import JOINT_NAMES, GenesisEnvironment
from src.experiment.messages import Command, ObservationMessage
from src.ros.bridge import RosBridge


def publish_sensors(bridge: RosBridge, prefix: str, state: ObservationMessage) -> None:
    """関節・シミュレーション時刻・カメラを標準ROS型で配信する。"""
    stamp = {"sec": state.steps // 20, "nanosec": (state.steps % 20) * 50_000_000}
    header = {"stamp": stamp, "frame_id": "world"}
    bridge.publish("/clock", {"clock": stamp})
    bridge.publish(
        prefix + "/joint_states",
        {
            "header": header,
            "name": JOINT_NAMES,
            "position": state.joint_positions,
            "velocity": [],
            "effort": [],
        },
    )
    for name, pixels in (("main", state.main_png), ("wrist", state.wrist_png)):
        if pixels is not None:
            bridge.publish(
                prefix + f"/camera/{name}/compressed",
                {
                    "header": {"stamp": stamp, "frame_id": name + "_camera"},
                    "format": "png",
                    "data": pixels,
                },
            )


def register_topics(bridge: RosBridge, prefix: str) -> None:
    """Genesisが使用するROSトピックを用意する。"""
    bridge.subscribe(prefix + "/command")
    bridge.advertise(prefix + "/observation", "std_msgs/msg/String")
    bridge.advertise(prefix + "/joint_states", "sensor_msgs/msg/JointState")
    bridge.advertise("/clock", "rosgraph_msgs/msg/Clock")
    for name in ("main", "wrist"):
        bridge.advertise(
            prefix + f"/camera/{name}/compressed", "sensor_msgs/msg/CompressedImage"
        )


def serve(settings: Settings, directory: Path) -> dict[str, object]:
    """ROSを通った指令に限って物理演算を行う。"""
    with RosBridge(settings.rosbridge_url, settings.ros_timeout_seconds) as bridge:
        register_topics(bridge, settings.topic_prefix)
        environment = GenesisEnvironment(settings, cameras=True)
        try:
            last_command: Command | None = None
            last_observation: ObservationMessage | None = None
            with (directory / "ros_commands.jsonl").open(
                "w", encoding="utf-8"
            ) as trace:
                while True:
                    raw = bridge.receive_data(
                        settings.topic_prefix + "/command",
                        time.monotonic() + settings.ros_timeout_seconds,
                    )
                    command = Command.model_validate_json(raw)
                    if (
                        last_command is not None
                        and command.request_id == last_command.request_id
                    ):
                        if command != last_command:
                            raise ValueError("同じ要求IDに異なる指令が届きました")
                        observation = last_observation
                    else:
                        if command.operation == "step":
                            environment.advance(command)
                        observation = environment.observation(command.request_id)
                        trace.write(command.model_dump_json(by_alias=True) + "\n")
                        trace.flush()
                    if observation is None:
                        raise RuntimeError("対応するGenesis観測がありません")
                    bridge.publish(
                        settings.topic_prefix + "/observation",
                        {
                            "data": observation.model_dump_json(by_alias=True),
                        },
                    )
                    publish_sensors(bridge, settings.topic_prefix, observation)
                    last_command, last_observation = command, observation
                    if command.operation == "finish":
                        return {
                            "status": "evaluated",
                            "simulator": "genesis",
                            "ros_connected": True,
                            "steps": observation.steps,
                            "ingredients": {
                                name: value.model_dump()
                                for name, value in observation.ingredients.items()
                            },
                        }
        finally:
            environment.close()
