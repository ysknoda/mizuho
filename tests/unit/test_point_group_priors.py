"""Unit tests for mizuho.data.point_group_priors."""

from mizuho.data import IS_SYMMORPHIC_OF_SG, POINT_GROUP_OF_SG, space_group_prior_weight
from mizuho.data.point_group_priors import (
    CENTROSYMMETRIC_POINT_GROUPS, HIGH_FREQUENCY_SG_BOOST,
)


def test_all_centrosymmetric_point_groups_are_real_point_groups():
    all_point_groups = set(POINT_GROUP_OF_SG.values())
    assert CENTROSYMMETRIC_POINT_GROUPS <= all_point_groups


def test_exactly_11_centrosymmetric_point_groups():
    assert len(CENTROSYMMETRIC_POINT_GROUPS) == 11


def test_high_frequency_boost_keys_are_valid_space_groups():
    for sg in HIGH_FREQUENCY_SG_BOOST:
        assert 1 <= sg <= 230


def test_symmorphic_centrosymmetric_sg_outweighs_baseline():
    # Fm-3m (225): symmorphic and centrosymmetric (m-3m) and boosted.
    w_225 = space_group_prior_weight(225, POINT_GROUP_OF_SG[225], IS_SYMMORPHIC_OF_SG[225])
    # SG1 (P1): non-symmorphic-irrelevant (it IS symmorphic actually,
    # but point group "1" is not centrosymmetric and not boosted).
    w_1 = space_group_prior_weight(1, POINT_GROUP_OF_SG[1], IS_SYMMORPHIC_OF_SG[1])
    assert w_225 > w_1


def test_prior_weight_always_at_least_baseline():
    for sg in range(1, 231):
        w = space_group_prior_weight(sg, POINT_GROUP_OF_SG[sg], IS_SYMMORPHIC_OF_SG[sg])
        assert w >= 1.0
