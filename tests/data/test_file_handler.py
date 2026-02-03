import pytest
import pandas as pd
from data.file_handler import eph_from_indec, get_eph

def test_get_eph__individual_success(eph_individual_data, mocker):
    """Tests that get_eph successfully retrieves individual data from pyeph and does not 
    call eph_from_indec when pyeph returns data.

    Args:
        eph_individual_data (pytest.fixture): pytest fixture for individual EPH data
        mocker (pytest-mocker): pytest-mocker fixture for mocking functions
    """    
    mock_pyeph_get = mocker.patch("pyeph.get",
                                   return_value=eph_individual_data)
    mock_eph_from_indec = mocker.patch("data.file_handler.eph_from_indec")

    test_db_type = "individual"
    test_year = 2024
    test_period = 2

    result = get_eph(test_db_type, test_year, test_period)
    assert isinstance(result, pd.DataFrame)

    pd.testing.assert_frame_equal(result, eph_individual_data)
    mock_pyeph_get.assert_called_once_with(data="eph", year=test_year,
                                          period=test_period, base_type=test_db_type)
    mock_eph_from_indec.assert_not_called()

def test_get_eph__household_success(eph_household_data, mocker):
    """Tests that get_eph successfully retrieves household data from pyeph and does not 
    call eph_from_indec when pyeph returns data.

    Args:
        eph_household_data (pytest.fixture): pytest fixture for household EPH data
        mocker (pytest-mocker): pytest-mocker fixture for mocking functions
    """    
    mock_pyeph_get = mocker.patch("pyeph.get",
                                   return_value=eph_household_data)
    mock_eph_from_indec = mocker.patch("data.file_handler.eph_from_indec")

    test_db_type = "household"
    test_year = 2024
    test_period = 2

    result = get_eph(test_db_type, test_year, test_period)
    assert isinstance(result, pd.DataFrame)

    pd.testing.assert_frame_equal(result, eph_household_data)
    mock_pyeph_get.assert_called_once_with(data="eph", year=test_year,
                                          period=test_period, base_type=test_db_type)
    mock_eph_from_indec.assert_not_called()

