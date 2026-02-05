import zipfile
import csv
from io import StringIO, BytesIO
import pytest
import pyeph
import pandas as pd
from data.file_handler import eph_from_indec, get_eph


def test_get_eph_individual_success(eph_individual_data, mocker):
    """Tests that get_eph successfully retrieves individual data from pyeph and does not
    call eph_from_indec when pyeph returns data.

    Args:
        eph_individual_data (pytest.fixture): pytest fixture for individual EPH data
        mocker (pytest-mocker): pytest-mocker fixture for mocking functions
    """
    mock_pyeph_get = mocker.patch("pyeph.get", return_value=eph_individual_data)
    mock_eph_from_indec = mocker.patch("data.file_handler.eph_from_indec")

    test_db_type = "individual"
    test_year = 2024
    test_period = 2

    result = get_eph(test_db_type, test_year, test_period)
    assert isinstance(result, pd.DataFrame)

    pd.testing.assert_frame_equal(result, eph_individual_data)
    mock_pyeph_get.assert_called_once_with(
        data="eph", year=test_year, period=test_period, base_type=test_db_type
    )
    mock_eph_from_indec.assert_not_called()


def test_get_eph_household_success(eph_household_data, mocker):
    """Tests that get_eph successfully retrieves household data from pyeph and does not
    call eph_from_indec when pyeph returns data.

    Args:
        eph_household_data (pytest.fixture): pytest fixture for household EPH data
        mocker (pytest-mocker): pytest-mocker fixture for mocking functions
    """
    mock_pyeph_get = mocker.patch("pyeph.get", return_value=eph_household_data)
    mock_eph_from_indec = mocker.patch("data.file_handler.eph_from_indec")

    test_db_type = "household"
    test_year = 2024
    test_period = 2

    result = get_eph(test_db_type, test_year, test_period)
    assert isinstance(result, pd.DataFrame)

    pd.testing.assert_frame_equal(result, eph_household_data)
    mock_pyeph_get.assert_called_once_with(
        data="eph", year=test_year, period=test_period, base_type=test_db_type
    )
    mock_eph_from_indec.assert_not_called()


def test_get_eph_from_indec_success(mocker):
    """Tests that get_eph calls eph_from_indec when pyeph raises a
    NonExistentDBError, and that it returns the correct DataFrame.

    Args:
        eph_individual_data (pytest.fixture): pytest fixture for individual EPH data
        mocker (pytest-mocker): pytest-mocker fixture for mocking functions
    """
    csv_buffer = StringIO()
    writer = csv.writer(csv_buffer, delimiter=";")
    writer.writerow(["id", "age", "income"])
    writer.writerow([1, 30, 50000])
    writer.writerow([2, 45, 80000])
    csv_content = csv_buffer.getvalue().encode("utf-8")
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("eph_individual_2024_2.csv", csv_content)

    mock_requests_get = mocker.patch("requests.get")
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.content = zip_buffer.getvalue()

    result = eph_from_indec("individual", 2024, 2)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2
    assert list(result.columns) == ["id", "age", "income"]


def test_get_eph_from_indec_request_failure(mocker):
    """Tests that eph_from_indec returns None when the HTTP request fails.

    Args:
        mocker (pytest-mocker): pytest-mocker fixture for mocking functions
    """
    mock_requests_get = mocker.patch("requests.get")
    mock_requests_get.return_value.status_code = 404

    result = eph_from_indec("individual", 2024, 2)
    assert result is None


def test_get_eph_from_indec_no_matching_file(mocker):
    """Tests that eph_from_indec returns None when no matching file is found in the zip.

    Args:
        mocker (pytest-mocker): pytest-mocker fixture for mocking functions
    """
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("some_other_file.csv", "data")

    mock_requests_get = mocker.patch("requests.get")
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.content = zip_buffer.getvalue()

    result = eph_from_indec("individual", 2024, 2)
    assert result is None


def test_get_eph_successful_fallback(eph_individual_data, mocker):
    """Tests that get_eph successfully retrieves data from eph_from_indec
    when pyeph raises a NonExistentDBError.

    Args:
        eph_individual_data (pytest.fixture): pytest fixture for individual EPH data
        mocker (pytest-mocker): pytest-mocker fixture for mocking functions
    """
    mock_pyeph_get = mocker.patch(
        "pyeph.get", side_effect=pyeph.errors.NonExistentDBError
    )  # type: ignore
    mock_eph_from_indec = mocker.patch(
        "data.file_handler.eph_from_indec", return_value=eph_individual_data
    )

    test_db_type = "individual"
    test_year = 2024
    test_period = 2

    result = get_eph(test_db_type, test_year, test_period)
    assert isinstance(result, pd.DataFrame)

    pd.testing.assert_frame_equal(result, eph_individual_data)
    mock_pyeph_get.assert_called_once_with(
        data="eph", year=test_year, period=test_period, base_type=test_db_type
    )
    mock_eph_from_indec.assert_called_once_with(test_db_type, test_year, test_period)


def test_get_eph_no_db_found(mocker):
    """Tests that get_eph calls eph_from_indec when pyeph raises a
    NonExistentDBError, and that it returns None when eph_from_indec
    raises a BadZipFile error.

    Args:
        mocker (pytest-mocker): pytest-mocker fixture for mocking functions
    """
    mock_pyeph_get = mocker.patch(
        "pyeph.get", side_effect=pyeph.errors.NonExistentDBError
    )  # type: ignore
    mock_eph_from_indec = mocker.patch(
        "data.file_handler.eph_from_indec", side_effect=zipfile.BadZipFile
    )

    test_db_type = "individual"
    test_year = 2024
    test_period = 2

    result = get_eph(test_db_type, test_year, test_period)
    assert result is None

    mock_pyeph_get.assert_called_once_with(
        data="eph", year=test_year, period=test_period, base_type=test_db_type
    )
    mock_eph_from_indec.assert_called_once_with(test_db_type, test_year, test_period)
