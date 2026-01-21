"""
Data preprocessing module for EPH (Encuesta Permanente de Hogares) datasets.

This module provides utilities to combine and transform EPH data by merging individual
and household records, and creating derived binary features based on conditional logic.
It includes predefined column lists for common household and individual feature sets.

Main functions:
    - join_individuals_households: Merges individual and household DataFrames on common identifiers
    - create_binary_feature: Generates binary features from boolean conditions

Predefined column lists:
    - id_house_features: Identifier columns for household data
    - house_features: Feature columns for household data
    - id_individual_features: Identifier columns for individual data
    - individual_features: Feature columns for individual data
    - join_features: Common columns used for merging individual and household data
"""

from typing import Literal
import pandas as pd
import numpy as np

id_house_features = [
    'CODUSU', 'NRO_HOGAR', 'REALIZADA', 'ANO4', 'TRIMESTRE', 'REGION', 'MAS_500', 'AGLOMERADO', 'PONDERA'
]

house_features = [
    'IV1', 'IV2', 'IV3', 'IV4', 'IV5', 'IV6', 'IV7', 'IV8', 'IV9', 'IV10', 'IV11', 'IV12_1', 'IV12_2', 'IV12_3', 'II1', 'II2', 'II3', 'II4_1', 'II4_2', 'II4_3', 'II7', 'II8', 'II9', 'V1', 'V2', 'V21', 'V22', 'V3', 'V5', 'V6', 'V7', 'V8', 'V11', 'V12', 'V13', 'V14', 'IX_TOT', 'IX_MEN10', 'IX_MAYEQ10', 'ITF', 'DECCFR'
]

id_individual_features = [
    'CODUSU', 'NRO_HOGAR', 'COMPONENTE', 'H15', 'ANO4', 'TRIMESTRE',  'REGION', 'MAS_500', 'AGLOMERADO', 'PONDERA'
]

individual_features = [
    'CH03', 'CH04', 'CH06', 'CH07', 'CH08', 'CH09', 'CH10', 'CH11', 'CH15', 'CH16', 'ESTADO', 'CAT_OCUP', 'CAT_INAC', 'PP02E', 'PP02H', 'PP07I', 'PP07H', 'PP04B1', 'T_VI', 'NIVEL_ED', 'V2_M'
]

join_features = ['CODUSU', 'NRO_HOGAR', 'ANO4', 'TRIMESTRE', 'REGION','MAS_500', 'AGLOMERADO', 'PONDERA']

def join_individuals_households(individual_df: pd.DataFrame,
                          household_df: pd.DataFrame,
                          on: list[str] | None = None,
                          how: Literal['inner'] = 'inner',
                          suffixes: tuple[str, str] = ('', '_r')) -> pd.DataFrame:
    """Joins an individual dataframe with a household dataframe on CODUSU and NRO_HOGAR columns.

    Args:
        individual_df (pd.DataFrame): DataFrame corresponding to the individual database.
        household_df (pd.DataFrame): DataFrame corresponding to the household database.
        on (list, optional): List of columns to join on. Defaults to `join_features`.

    Returns:
        pd.DataFrame: The merged dataframe.
    """
    if on is None:
        on = join_features
    merged_df = pd.merge(individual_df,
                        household_df,
                        on=on,
                        how=how,
                        suffixes=suffixes)
    return merged_df

def create_binary_feature(data: pd.DataFrame,
                          feature_name: str,
                          condition: pd.Series) -> pd.DataFrame:
    """Creates a binary categorical feature based on a condition on the columns of the dataframe passed as an argument.

    Args:
        df(pd.DataFrame): DataFrame to which the variable will be added
        feature_name (str): name of the variable
        condition (pd.Series): boolean mask

    Returns:
        pd.DataFrame: a DataFrame with the categorical column named 'feature_name'
    """
    data.loc[:, feature_name] = np.where(condition, 1.0, 0.0)
    return data

if __name__ == "__main__":
    # Example usage
    from file_handler import get_eph
    
    # Fetch individual and household data for the same period
    idf = get_eph('individual', 2020, 1)
    hdf = get_eph('hogar', 2020, 1)
    
    if idf is not None and hdf is not None:
        # Join individual and household data
        merged_df = join_individuals_households(idf, hdf)
        print("Merged DataFrame shape:", merged_df.shape)
        print(merged_df.head())
        
        # Create a binary feature example: employed (ESTADO == 1)
        merged_df = create_binary_feature(
            merged_df,
            'is_employed',
            merged_df['ESTADO'] == 1
        )
        print("\nDataFrame with binary feature:")
        print(merged_df[['CODUSU', 'NRO_HOGAR', 'ESTADO', 'is_employed']].head())
    else:
        print("Failed to fetch data from INDEC")
