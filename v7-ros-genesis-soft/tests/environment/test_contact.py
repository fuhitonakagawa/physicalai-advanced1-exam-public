"""指腹への近接と、把持の成立を区別するための形状判定。"""

import numpy as np
import pytest

from src.environment.contact import bilateral_proximity


@pytest.mark.parametrize(
    "points,expected",
    [
        ([(0, 0.021, 0.103), (0, -0.021, 0.103)], True),
        ([(0, 0.021, 0.103), (0, 0, 0.103)], False),
        ([(0, 0.017, 0.103), (0, -0.017, 0.103)], False),
        ([(0, 0.021, 0.12), (0, -0.021, 0.12)], False),
        ([(0.02, 0.021, 0.103), (0.02, -0.021, 0.103)], False),
        ([(0, 0.03, 0.103), (0, -0.03, 0.103)], False),
    ],
)
def test_pad_proximity(
    points: list[tuple[float, float, float]], expected: bool
) -> None:
    """片側だけ、離れた頂点、指腹の範囲外を両側接触とみなさない。"""
    assert bilateral_proximity(np.asarray(points), 0.02, 0.02) is expected
