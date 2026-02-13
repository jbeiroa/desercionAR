import pytest
import pandas as pd

from data.cleaning import select_features, filter_by_columns

def test_select_features_success(eph_individual_data):
    """Tests the select_features function to ensure it correctly selects specified columns
    from the EPH individual DataFrame.
    Args:
        eph_individual_data (pytest.fixture): pytest fixture for individual EPH data
    """
    id_columns = ["CODUSU", "NRO_HOGAR", "COMPONENTE"]
    feature_columns = ["CH03", "CH04", "CH06"]

    result = select_features(eph_individual_data, feature_columns, id_columns)
    assert isinstance(result, pd.DataFrame)
    expected_columns = id_columns + feature_columns
    assert list(result.columns) == expected_columns
    assert len(result) == len(eph_individual_data)

def test_select_features_none_data():
    """Tests that select_features raises a TypeError when data is None."""
    with pytest.raises(TypeError, match="data cannot be None"):
        select_features(None, ["CH03", "CH04"])

def test_filter_by_columns_success(eph_individual_data):
    """Tests the filter_by_columns function to ensure it correctly filters the EPH individual
    DataFrame based on a query and returns specified columns.
    Args:
        eph_individual_data (pytest.fixture): pytest fixture for individual EPH data
    """
    query_filter = "CH03 == 1"
    feature_columns = ["CODUSU", "NRO_HOGAR", "CH03", "CH04"]

    result = filter_by_columns(eph_individual_data, query_filter, feature_columns)
    assert isinstance(result, pd.DataFrame)
    assert all(result["CH03"] == 1)
    assert list(result.columns) == feature_columns