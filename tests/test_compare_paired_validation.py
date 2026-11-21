from scripts.compare_paired_validation import holm_adjust, percentile


def test_percentile_interpolates_endpoints_and_middle():
    values = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert percentile(values, 0.0) == 0.0
    assert percentile(values, 0.5) == 2.0
    assert percentile(values, 1.0) == 4.0


def test_holm_adjustment_is_monotone_in_ordered_p_values():
    adjusted = holm_adjust({"P1-D0": 0.01, "P2-D0": 0.04})
    assert adjusted["P1-D0"] == 0.02
    assert adjusted["P2-D0"] == 0.04
