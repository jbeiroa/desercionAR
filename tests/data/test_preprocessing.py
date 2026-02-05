import pytest
import pandas as pd
import numpy as np

from data.preprocessing import (
    join_individuals_households,
    create_binary_feature
)


def test_join_individuals_households_success(eph_individual_data,
                                             eph_household_data,
                                             eph_merged_data):
    """Tests the join_individuals_households function to ensure it correctly merges
    individual and household DataFrames on common identifiers.

    Args:
        eph_individual_data (pytest.fixture): pytest fixture for individual EPH data
        eph_household_data (pytest.fixture): pytest fixture for household EPH data
    """
    result = join_individuals_households(eph_individual_data,
                                         eph_household_data)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, eph_merged_data)
    
def test_create_binary_feature_success():
    """Tests the create_binary_feature function to ensure it correctly generates a binary
    feature from a boolean condition.

    """
    test_df = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, np.nan]
    })
    expected_df = pd.DataFrame({
        "feature": [0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
    })
    result = create_binary_feature(test_df, "feature", test_df["feature"] > 2)
    assert isinstance(result, pd.DataFrame)
    pd.testing.assert_frame_equal(result, expected_df)