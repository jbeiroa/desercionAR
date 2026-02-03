import pandas as pd
import pyeph

join_features = [
    "CODUSU",
    "NRO_HOGAR",
    "ANO4",
    "TRIMESTRE",
    "REGION",
    "MAS_500",
    "AGLOMERADO",
    "PONDERA",
]

def create_eph_fixture():
    """Fetches one quarter of the EPH and saves it as a fixture for testing.
    """    
    individual_df = pyeph.get(data="eph", 
                              year=2024, 
                              period=2, 
                              base_type="individual")
    household_df = pyeph.get(data="eph", 
                             year=2024, 
                             period=2, 
                             base_type="hogar")

    merged_df = pd.merge(individual_df, household_df, 
                         on=join_features, how="inner",
                         suffixes=("", "_r"))
    
    # Save to CSV
    individual_df.to_csv("tests/fixtures/individual_fixture.csv", index=False)
    household_df.to_csv("tests/fixtures/household_fixture.csv", index=False)
    merged_df.to_csv("tests/fixtures/eph_fixture.csv", index=False)


if __name__ == "__main__":
    create_eph_fixture()
    print("EPH fixture files created.")
