"""
Data cleaning module for EPH (Encuesta Permanente de Hogares) datasets.

This module provides utilities to select, filter, and process EPH household and individual
survey data. It defines standard column lists for household and individual datasets and
offers functions to extract specific features and apply query-based filters.

Main functions:
    - select_features: Extracts specified columns from a DataFrame
    - filter_by_columns: Applies query filters to a DataFrame
"""
import pandas as pd

def select_features(data: pd.DataFrame | None,
                    column_list: list,
                    id_columns: list | None = None) -> pd.DataFrame:
    """Selects a specific list of columns from a DataFrame, including ID columns if provided.

    Args:
        data (pd.DataFrame): EPH DataFrame
        column_list (list): list of columns to select
        id_columns (list | None, optional): list of ID columns to include.
        Defaults to None.

    Returns:
        pd.DataFrame: dataframe with selected columns.
    """
    if data is None:
        raise TypeError("data cannot be None")
    if id_columns is None:
        id_columns = []
    return data[id_columns + column_list]


def filter_by_columns(data: pd.DataFrame,
                      query_filter: str,
                      column_list: list | pd.Index | None = None) -> pd.DataFrame:
    """Filters a dataframe using the query method.

    Args:
        data (pd.DataFrame): EPH dataframe
        query_filter (str): string to filter the dataframe
        column_list (list | pd.Index | None): list of columns to return.
        If None (default), returns all columns.

    Returns:
        pd.DataFrame: filtered dataframe.
    """
    if column_list is None:
        column_list = data.columns
    filtered_data = data.query(query_filter)[column_list]
    return filtered_data

if __name__ == "__main__":
    # Example usage
    from file_handler import get_eph
    id_individual_features = [
        'CODUSU', 'NRO_HOGAR', 'COMPONENTE', 'H15', 'ANO4', 'TRIMESTRE',  'REGION', 'MAS_500', 'AGLOMERADO', 'PONDERA'
    ]

    id_features = [
        'CH03', 'CH04', 'CH06', 'CH07', 'CH08', 'CH09', 'CH10', 'CH11', 'CH15', 'CH16', 'ESTADO', 'CAT_OCUP', 'CAT_INAC', 'PP02E', 'PP02H', 'PP07I', 'PP07H', 'PP04B1', 'T_VI', 'NIVEL_ED', 'V2_M'
    ]
    df = get_eph('individual', 2020, 1)
    if df is not None:
        print(df.head())
    selected_data = select_features(df, id_individual_features, id_features)
    print(selected_data.head())
    filtered_df = filter_by_columns(df, 'CH04 == 1 and ESTADO == 1', id_individual_features + id_features)
    print(filtered_df.head())