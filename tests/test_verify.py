from outlet_audit.verify import MIN_MATCHES_FOR_HOMOGRAPHY, geom_stat_value


def test_geom_stat_floors_sub_ransac_counts():
    assert geom_stat_value(0) == geom_stat_value(MIN_MATCHES_FOR_HOMOGRAPHY)
    assert geom_stat_value(MIN_MATCHES_FOR_HOMOGRAPHY + 1) > geom_stat_value(MIN_MATCHES_FOR_HOMOGRAPHY)
