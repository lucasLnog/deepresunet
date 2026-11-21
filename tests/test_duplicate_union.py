import numpy as np
import pytest

from src.tcc_segmentation.data.duplicate_union import analyze_mask_pair


def test_or_union_metrics_are_explicit() -> None:
    mask_a = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
    mask_b = np.array([[0, 1, 1], [0, 0, 0]], dtype=bool)
    result = analyze_mask_pair(mask_a, mask_b)
    assert result["area_a"] == 2
    assert result["area_b"] == 2
    assert result["intersection_pixels"] == 1
    assert result["union_pixels"] == 3
    assert result["disagreement_pixels"] == 2
    assert result["dice_between_references"] == pytest.approx(0.5)
    assert result["or_added_over_larger_fraction"] == pytest.approx(0.5)


def test_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        analyze_mask_pair(np.zeros((2, 2)), np.zeros((3, 3)))
