import pandas as pd
import numpy as np
import os

def homogeneizar_binarias(df, columns):
    replace_dict = {col: {2: 0, 'S': 1, 'N': 0, 'NO': 0} for col in columns}
    data = df.replace(replace_dict)
    data[columns].astype('float64', copy=False)
    return data

def aglomerados_a_distancia(df, aglomerado='AGLOMERADO'):
    data_x_y = {
        'eph_codagl': [13, 29, 31, 25, 34, 7, 26, 15, 4, 91, 18, 23, 30, 12, 20, 93, 8, 14, 6, 5, 3, 9, 22, 36, 38, 38, 10, 19, 2, 32, 17, 33, 27],
        'eph_aglome': ['Gran Córdoba', 'Gran Tucumán - Tafi Viejo', 'Ushuaia - Rio Grande', 'La Rioja', 'Mar del Plata - Batán', 'Posadas', 'San Luis - El Chorrillo', 'Formosa', 'Gran Rosario', 'Rawson - Trelew', 'Santiago del Estero - La Banda', 'Salta', 'Santa Rosa - Toay', 'Corrientes', 'Rio Gallegos', 'Viedma - Carmen de Patagones', 'Gran Resistencia', 'Concordia', 'Gran Paraná', 'Gran Santa Fe', 'Bahia Blanca - Cerri', 'Comodoro Rivadavia - Rada Tilly', 'Gran Catamarca', 'Rio Cuarto', 'San Nicolas - Villa Constitiución', 'San Nicolas - Villa Constitiución', 'Gran Mendoza', 'Jujuy - Palpalá', 'Gran La Plata', 'CABA', 'Neuquén - Plottier', 'Partidos del GBA', 'Gran San Juan'],
        'x': [3.668196e+06, 3.575528e+06, 3.368650e+06, 3.433735e+06, 4.238702e+06, 4.500828e+06, 3.470658e+06, 4.290676e+06, 3.989413e+06, 3.561793e+06, 3.671129e+06, 3.558820e+06, 3.652175e+06, 4.215852e+06, 3.274586e+06, 3.754150e+06, 4.194276e+06, 4.258864e+06, 4.023207e+06, 4.008905e+06, 3.824880e+06, 3.382300e+06, 3.520630e+06, 3.656809e+06, 4.035477e+06, 4.029783e+06, 3.231898e+06, 3.560475e+06, 4.234043e+06, 4.193488e+06, 3.310530e+06, 4.180647e+06, 3.260047e+06],
        'y': [6.533650e+06, 7.036009e+06, 3.980855e+06, 6.726538e+06, 5.760505e+06, 6.922012e+06, 6.318389e+06, 7.090830e+06, 6.346344e+06, 5.211965e+06, 6.926946e+06, 7.258796e+06, 5.947417e+06, 6.941437e+06, 4.274342e+06, 5.478764e+06, 6.944114e+06, 6.501289e+06, 6.473109e+06, 6.487715e+06, 5.709612e+06, 4.921987e+06, 6.858955e+06, 6.336869e+06, 6.293741e+06, 6.307894e+06, 6.360832e+06, 7.329096e+06, 6.103368e+06, 6.144082e+06, 5.686581e+06, 6.148549e+06, 6.509854e+06]
    }
    df_x_y = pd.DataFrame(data_x_y)
    df_aglomerado = df.copy()
    df_aglomerado['x_temp'] = None
    df_aglomerado['y_temp'] = None
    df_aglomerado['DISTANCIA'] = None

    # Coordenadas de la capital
    capital_x = 4.193488e+06
    capital_y = 6.144082e+06
    for index, row in df_aglomerado.iterrows():
        eph_codagl = row[aglomerado]
        matching_row = df_x_y[df_x_y['eph_codagl'] == eph_codagl]
        if not matching_row.empty:
            x_temp = matching_row['x'].values[0]
            y_temp = matching_row['y'].values[0]
            distancia = np.sqrt((x_temp - capital_x)**2 + (y_temp - capital_y)**2)
            df_aglomerado.at[index, 'DISTANCIA'] = distancia

    # Eliminar columnas temporales 'x_temp' e 'y_temp'
    df_aglomerado['AGLOMERADO'] = df_aglomerado['DISTANCIA']
    df_aglomerado.drop(columns=['x_temp', 'y_temp', 'DISTANCIA'], inplace=True)
    return df_aglomerado


def preprocess_data(data: pd.DataFrame, train_test: bool = True) -> pd.DataFrame:
    if len(data) > 1:
        datos = pd.concat(data)
    if len(data) == 1:
        datos = data[0]
    drop_cols = [
        'IV8', 'IX_MAYEQ10', 'CAT_OCUP', 'CAT_INAC', 'CAT_OCUP_jefx', 'JEFE_TRABAJA', 'T_VI', 'V2_M', 'CH04_conyuge', 'CH04_jefx', 'NBI_SUBSISTENCIA', 'IV10', 'II7', 'IV12_1', 'IV12_3', 'PP07I',
        'PP07H', 'PP02E_jefx', 'REALIZADA_jefx', 'REALIZADA_conyuge',
        'H15', 'ITF', 'REALIZADA'
    ]
    if train_test is True:
        columnas_binarias = [
            'CH11', 'PP02H', 'PP04B1', 'IV5', 'IV12_2', 'II3', 'II4_1', 'II4_2', 'II4_3', 'V1', 'V2', 'V21', 'V22', 'V3', 'V5', 'V6', 'V7', 'V8', 'V11', 'V12', 'V13', 'V14', 'PP07I_jefx', 'PP07H_jefx', 'PP04B1_jefx', 'CONYUGE_TRABAJA', 'JEFA_MUJER', 'HOGAR_MONOP', 'NBI_COBERTURA_PREVISIONAL', 'NBI_DIFLABORAL', 'NBI_HACINAMIENTO', 'NBI_SANITARIA', 'NBI_TENENCIA', 'NBI_TRABAJO_PRECARIO', 'NBI_VIVIENDA', 'NBI_ZONA_VULNERABLE', 'abandono', 'MAS_500', 'CH04'
        ]
    if train_test is False:
        columnas_binarias = [
            'CH11', 'PP02H', 'PP04B1', 'IV5', 'IV12_2', 'II3', 'II4_1', 'II4_2', 'II4_3', 'V1', 'V2', 'V21', 'V22', 'V3', 'V5', 'V6', 'V7', 'V8', 'V11', 'V12', 'V13', 'V14', 'PP07I_jefx', 'PP07H_jefx', 'PP04B1_jefx', 'CONYUGE_TRABAJA', 'JEFA_MUJER', 'HOGAR_MONOP', 'NBI_COBERTURA_PREVISIONAL', 'NBI_DIFLABORAL', 'NBI_HACINAMIENTO', 'NBI_SANITARIA', 'NBI_TENENCIA', 'NBI_TRABAJO_PRECARIO', 'NBI_VIVIENDA', 'NBI_ZONA_VULNERABLE', 'MAS_500', 'CH04'
        ]
    data = datos[datos['H15'] == 1]
    data.drop(drop_cols, axis=1, inplace=True)
    data = homogeneizar_binarias(data, columnas_binarias)
    # PP04B1 --> renombre a servicio_domestico + reemplazo de valores
    data.loc[:, 'PP04B1'].replace({2: 0, np.nan: 0}, inplace=True)
    data.rename({'PP04B1': 'servicio_domestico'},
                axis=1, inplace=True)
    # PP07H_jefx renombre a apotes_jubilatorios_jefx
    data.rename({'PP07H_jefx': 'APORTES_JUBILATORIOS_jefx'},
                axis=1, inplace=True)
    # variables conyuge --> lleno NaN con ceros pues corresponden a HOGAR_MONOP==1
    cvars = data.columns.str.endswith('_conyuge')
    data[data.columns[cvars]] = data.loc[:, cvars].fillna(0)
    data = aglomerados_a_distancia(data)
    return data

def remove_duplicates(df_list):
    if len(df_list) < 2:
        raise ValueError("df_list debe ser una lista de al menos dos elementos.")
    result_df = df_list[0]
    for df in df_list[1:]:
        result_df = pd.merge(result_df, df[['CODUSU', 'NRO_HOGAR', 'COMPONENTE']],
                             on=['CODUSU', 'NRO_HOGAR', 'COMPONENTE'],
                             how='left', indicator=True)
        result_df = result_df[result_df['_merge'] == 'left_only']
        result_df = result_df.drop('_merge', axis=1)

    return result_df
