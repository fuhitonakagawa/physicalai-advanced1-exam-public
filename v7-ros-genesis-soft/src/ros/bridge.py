"""実ROSブリッジに対するトピックとサービスの通信。"""

import json
import time
from types import TracebackType
from typing import Any
from uuid import uuid4

from websocket import WebSocket, create_connection


class RosBridge:
    """一つの接続でROSメッセージを順に送受信する。"""

    def __init__(self, url: str, timeout: float) -> None:
        """接続先と通信待ち時間を保持する。"""
        self.url = url
        self.timeout = timeout
        self.socket: WebSocket | None = None

    def __enter__(self) -> "RosBridge":
        """ローカルのROSブリッジへ接続する。"""
        self.socket = create_connection(self.url, timeout=self.timeout)
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """WebSocketを閉じる。"""
        if self.socket is not None:
            self.socket.close()

    def send(self, packet: dict[str, object]) -> None:
        """ROSブリッジのJSONメッセージを送る。"""
        if self.socket is None:
            raise RuntimeError("ROSブリッジに接続していません")
        self.socket.send(json.dumps(packet, ensure_ascii=False, allow_nan=False))

    def receive(self, deadline: float) -> dict[str, Any]:
        """期限までに一つのメッセージを受け取る。"""
        if self.socket is None:
            raise RuntimeError("ROSブリッジに接続していません")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("ROSの応答がありません")
        self.socket.settimeout(remaining)
        raw = self.socket.recv()
        if not raw:
            raise ConnectionError("ROSブリッジが接続を閉じました")
        if len(raw) > 2_000_000:
            raise ValueError("ROSメッセージがサイズ上限を超えています")
        packet = json.loads(raw)
        if not isinstance(packet, dict):
            raise ValueError("ROSの応答形式が不正です")
        if packet.get("op") == "status" and packet.get("level") == "error":
            raise RuntimeError("ROSブリッジが要求を拒否しました")
        return packet

    def advertise(self, topic: str, message_type: str) -> None:
        """送信するROSトピックの型を登録する。"""
        self.send({"op": "advertise", "topic": topic, "type": message_type})

    def subscribe(self, topic: str) -> None:
        """JSONを格納するString型トピックを購読する。"""
        self.send({"op": "subscribe", "topic": topic, "type": "std_msgs/msg/String"})

    def publish(self, topic: str, message: dict[str, object]) -> None:
        """ROS型に対応するメッセージを配信する。"""
        self.send({"op": "publish", "topic": topic, "msg": message})

    def receive_data(self, topic: str, deadline: float) -> str:
        """指定トピックのStringメッセージだけを返す。"""
        while True:
            packet = self.receive(deadline)
            if packet.get("op") == "publish" and packet.get("topic") == topic:
                data = packet.get("msg", {}).get("data")
                if not isinstance(data, str):
                    raise ValueError("ROSのStringデータが不正です")
                return data

    def service(self, name: str, args: dict[str, object]) -> dict[str, Any]:
        """ROSサービスの応答を要求IDで対応付ける。"""
        request_id = uuid4().hex
        self.send(
            {"op": "call_service", "service": name, "args": args, "id": request_id}
        )
        deadline = time.monotonic() + self.timeout
        while True:
            packet = self.receive(deadline)
            if (
                packet.get("op") == "service_response"
                and packet.get("id") == request_id
            ):
                if packet.get("result") is not True:
                    raise RuntimeError("ROSサービスが失敗しました")
                values = packet.get("values")
                if not isinstance(values, dict):
                    raise ValueError("ROSサービスの応答形式が不正です")
                return values

    def wait_subscriber(self, topic: str) -> None:
        """シミュレータの購読開始を待つ。"""
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if self.service("/rosapi/subscribers", {"topic": topic}).get("subscribers"):
                return
            time.sleep(0.25)
        raise TimeoutError("Genesis側のROS購読が開始されませんでした")
