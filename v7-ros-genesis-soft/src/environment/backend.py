"""Genesisのバックエンド初期化とCPU表示名。"""

import logging
import platform
from typing import Any

import cpuinfo


def initialize(genesis: Any, backend: str, seed: int) -> None:
    """装置検証をGenesisへ任せ、CPU名がない環境でも初期化する。"""
    get_info = cpuinfo.get_cpu_info
    details = get_info()
    if not any(
        details.get(key) for key in ("brand_raw", "hardware_raw", "vendor_id_raw")
    ):
        details["brand_raw"] = platform.machine() + " CPU"

    def cpu_details() -> dict[str, object]:
        """CPU情報と表示用のアーキテクチャ名を返す。"""
        return dict(details)

    cpuinfo.get_cpu_info = cpu_details
    try:
        genesis.init(
            backend=getattr(genesis, backend), seed=seed, logging_level=logging.WARNING
        )
    finally:
        cpuinfo.get_cpu_info = get_info
