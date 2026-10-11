"""弾性体の物理計算と保持・配置の検証。"""

import numpy as np

from src.config.settings import Settings
from src.environment.local import LocalEnvironment
from src.environment.ramen import RamenEnvironment
from src.experiment.assisted import AssistedRunner, GraspTask


def test_soft_objects_are_finite(environment: RamenEnvironment) -> None:
    """柔軟物の頂点と接触状態を実モデルから取得できる。"""
    assert environment.model.nflex == 2
    state = environment.observation("test")
    for food in state.ingredients.values():
        assert food.vertex_count > 8
        assert food.stable_steps == 0
        assert not food.finger_contacts
        assert food.contact_kind == "flex_contact_force"
    assert np.isfinite(state.state).all()
    assert state.main_png is None


def test_mechanical_soft_serving(
    environment: RamenEnvironment, settings: Settings
) -> None:
    """AIなしの機械診断として、実際の変形と両具材の配置を検証する。"""
    local = LocalEnvironment(environment, lambda state, command: None)
    runner = AssistedRunner(local, None, settings)
    results = [
        runner.run_task(GraspTask(food=food, instruction="pick"))
        for food in ("egg", "fish_cake")
    ]
    assert all(result.success for result in results)
    assert all(result.model_steps == 0 for result in results)
    assert all(item.stable_steps >= 10 for item in local.current.ingredients.values())
    assert environment.max_deformation["egg"] > 0.0003
    assert environment.max_deformation["fish_cake"] > 0.0002
    assert np.isfinite(environment.data.qpos).all()
