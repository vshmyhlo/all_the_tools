from all_the_tools.utils import weighted_sum


def test_weighted_sum():
    assert weighted_sum(10, 20, 0.8) == 12
