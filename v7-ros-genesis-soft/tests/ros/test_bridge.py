"""実ROSとの接続試験とは別のプロトコル契約テスト。"""

import json
import time
from typing import cast

import pytest
from websocket import WebSocket

from src.ros.bridge import RosBridge


class SocketStub:
    """指定したフレームだけを返すWebSocket代役。"""

    def __init__(self, frames: list[dict[str, object]]) -> None:
        """受信フレームを保持する。"""
        self.frames = iter(frames)
        self.timeout = 0.0

    def settimeout(self, seconds: float) -> None:
        """期限を保持する。"""
        self.timeout = seconds

    def recv(self) -> str:
        """次のJSONフレームを返す。"""
        return json.dumps(next(self.frames))


def test_ignores_other_topics() -> None:
    """別トピックの受信値を制御応答に使わない。"""
    bridge = RosBridge("ws://127.0.0.1:9090", 5)
    socket = SocketStub(
        [
            {"op": "publish", "topic": "/other", "msg": {"data": "wrong"}},
            {"op": "publish", "topic": "/state", "msg": {"data": "correct"}},
        ]
    )
    bridge.socket = cast(WebSocket, socket)
    assert bridge.receive_data("/state", time.monotonic() + 5) == "correct"
    assert socket.timeout > 0


def test_server_error() -> None:
    """ブリッジの拒否を成功した通信として扱わない。"""
    bridge = RosBridge("ws://127.0.0.1:9090", 5)
    bridge.socket = cast(WebSocket, SocketStub([{"op": "status", "level": "error"}]))
    with pytest.raises(RuntimeError):
        bridge.receive(time.monotonic() + 5)


def test_deadline() -> None:
    """待ち時間を超えた後は受信しない。"""
    bridge = RosBridge("ws://127.0.0.1:9090", 5)
    bridge.socket = cast(WebSocket, SocketStub([]))
    with pytest.raises(TimeoutError):
        bridge.receive(time.monotonic() - 1)
