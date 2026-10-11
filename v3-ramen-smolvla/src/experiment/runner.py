"""方策による閉ループ制御と全試行の記録。"""

import time
from collections.abc import Callable

import numpy as np

from src.experiment.types import (
    Environment,
    EpisodeResult,
    FloatArray,
    Observation,
    Policy,
)


def validate_action(action: FloatArray) -> tuple[FloatArray, int]:
    """動作の次元と有限性を検証し、環境の入力範囲に収める。"""
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("動作は有限な7次元ベクトルである必要があります")
    count = int(np.count_nonzero(np.abs(action) > 1))
    return np.clip(action, -1, 1).astype(np.float32), count


def run_episode(
    environment: Environment,
    policy: Policy,
    instruction: str,
    seed: int,
    initial_state_index: int | None,
    max_steps: int,
    record: Callable[[Observation, int], None],
) -> EpisodeResult:
    """予測した動作を逐次実行し、環境の成功判定を記録する。

    Args:
        environment: 物理環境。
        policy: 学習済み方策。
        instruction: 環境タスクの自然言語指示。
        seed: 試行の乱数シード。
        initial_state_index: 配布初期状態の番号。Noneは再サンプリング。
        max_steps: 制御回数の上限。
        record: 観測と制御番号の記録先。

    Returns:
        実際の成功判定と実行回数。
    """
    if max_steps < 1:
        raise ValueError("制御回数は1以上必要です")
    start = time.perf_counter()
    observation = environment.reset(seed, initial_state_index)
    policy.reset()
    record(observation, 0)
    success = False
    clipped_values = 0
    for step in range(1, max_steps + 1):
        action, clipped = validate_action(
            policy.select_action(observation, instruction)
        )
        clipped_values += clipped
        result = environment.step(action)
        observation = result.observation
        record(observation, step)
        success = result.success
        if success or result.terminated:
            break
    return EpisodeResult(
        seed,
        initial_state_index,
        success,
        step,
        clipped_values,
        time.perf_counter() - start,
    )
