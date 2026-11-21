import math

from src.tcc_segmentation.metric_aggregation import aggregate_global_pixel_metrics


def test_global_pixel_metrics_distinguish_micro_and_macro() -> None:
    rows = [
        {"tp": 90, "fp": 10, "fn": 10, "tn": 890, "dice": 0.9},
        {"tp": 0, "fp": 0, "fn": 10, "tn": 990, "dice": 0.0},
    ]

    result = aggregate_global_pixel_metrics(rows)

    assert result["counts"] == {"tp": 90, "fp": 10, "fn": 20, "tn": 1880}
    assert math.isclose(result["micro"]["dice"], 180 / 210)
    assert math.isclose(result["macro_dice"]["mean"], 0.45)
    assert result["zero_dice_cases"]["images"] == 1
    assert math.isclose(result["zero_dice_cases"]["image_fraction"], 0.5)
    assert math.isclose(result["zero_dice_cases"]["target_pixel_fraction"], 10 / 110)


def test_global_pixel_metrics_reject_empty_rows() -> None:
    try:
        aggregate_global_pixel_metrics([])
    except ValueError as error:
        assert "lista vazia" in str(error)
    else:
        raise AssertionError("Era esperado ValueError para lista vazia")
