import pytest
import pandas as pd

@pytest.fixture(scope="module")
def eph_individual_data() -> pd.DataFrame:
    return pd.read_csv("tests/fixtures/individual_fixture.csv", 
                       sep=",", header=0,
                       dtype={102: str, 105: str})

@pytest.fixture(scope="module")
def eph_household_data() -> pd.DataFrame:
    return pd.read_csv("tests/fixtures/household_fixture.csv", 
                       sep=",", header=0,
                       dtype={10: str, 13: str, 38: str})

@pytest.fixture(scope="module")
def eph_merged_data() -> pd.DataFrame:
    return pd.read_csv("tests/fixtures/eph_fixture.csv", sep=",", header=0)