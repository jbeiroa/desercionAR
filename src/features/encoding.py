import pandas as pd
import numpy as np


def is_binary_column(values):
    # Determinar si una columna del DataFrame es binaria y cumple con la condicion {0, 1, 4.0, 9.0, 0.0, 1.0}
    unique_values = set(values)
    return unique_values.issubset({0, 1, 4.0, 9.0, 0.0, 1.0})


def convert_binary_features(df, columns):
    df_nominales_binarias = df[columns].copy()

    # Reemplazar valores categorizados con numeros (-9, 4) como faltantes por nan
    for col in columns:
        if is_binary_column(df_nominales_binarias[col]):
            replace_dict = {9.0: -999, 4.0: -999,
                            9: -999, 4: -999, np.nan: -999}

        else:
            replace_dict = {}
            if df_nominales_binarias[col].dtype == 'object':
                replace_dict.update(
                    {'1': 1, '2': 0, '9.0': -999, '4.0': -999, 'S': 0, 'N': 1, 'NO': 1, np.nan: -999})
            elif df_nominales_binarias[col].dtype == 'int64':
                replace_dict.update(
                    {1: 1, 2: 0, 9: -999, 4: -999, np.nan: -999})

            elif df_nominales_binarias[col].dtype == 'float64':
                replace_dict.update(
                    {1.0: 1.0, 2.0: 0.0, 9.0: -999, 4.0: -999, np.nan: -999})

        df_nominales_binarias[col] = df_nominales_binarias[col].replace(
            replace_dict)

    # Reemplazar los NaN en las columnas 'PP07I', 'PP07H', 'PP04B1' con -999
    df_nominales_binarias[['PP07I', 'PP07H', 'PP04B1']] = df_nominales_binarias[[
        'PP07I', 'PP07H', 'PP04B1']].fillna(-999)

    # Convertir al tipo category
    df_nominales_binarias[columns] = df_nominales_binarias[columns].astype(
        'int64')

    return df_nominales_binarias


def convert_cat_nominal_features(df, columns):
    df_nominales_no_binarias = df[columns].copy()

    for col in columns:
        # Reemplazar valores 9 por -999
        if col in ['CH16', 'CH15', 'CH07']:
            df_nominales_no_binarias[col] = df_nominales_no_binarias[col].replace(
                {9: -999, np.nan: -999})
        elif col in df_nominales_no_binarias:
            df_nominales_no_binarias[col] = df_nominales_no_binarias[col].replace(
                {np.nan: -999})

    # Convertir todas las columnas a integer fuera del bucle
    df_nominales_no_binarias[columns] = df_nominales_no_binarias[columns].astype(
        'int64')

    return df_nominales_no_binarias


def convert_cat_ordinal_features(df, columns, fill_value=None, non_numeric_code=-999):
    df_ordinales = df[columns].copy()

    # Imputar NaN con fill_value si se especifica
    if fill_value is not None:
        df_ordinales.fillna(fill_value, inplace=True)

    # Convertir a categórica y categorizar con números enteros
    for col in columns:
        df_ordinales[col] = pd.to_numeric(
            df_ordinales[col], errors='coerce')  # Convertir a numérico
        # Máscara para valores no numéricos
        non_numeric_mask = df_ordinales[col].isnull()
        # Reemplazar valores no numéricos con código específico
        df_ordinales.loc[non_numeric_mask, col] = non_numeric_code
        df_ordinales[col] = df_ordinales[col].astype(int)  # Convertir a entero

    return df_ordinales


def convert_numeric_features(df, columns):
    df_numericas = df[columns].copy()

    for col in columns:
        # Reemplazar valores -9 y 99 por NaN
        if col in ['T_VI', 'V2_M', 'IV2', 'II1', 'II2']:
            df_numericas[col] = df_numericas[col].replace(
                {-9: np.nan, 99: np.nan})

    # Imputar los valores faltantes con la mediana de cada columna (se puede elegir otro metodo depende de distribuciòn)
    df_numericas = df_numericas.fillna(df_numericas.median())

    return df_numericas
