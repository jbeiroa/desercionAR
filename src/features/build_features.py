import pandas as pd
import numpy as np
from src.data import preprocessing as pr
from src.data import cleaning as l


def generate_auxiliary_dataframes(data: pd.DataFrame, households: pd.DataFrame):
    household_heads = data[data.CH03 == 1]
    spouses = data[data.CH03 == 2]
    # households by head's occupation status
    _household_heads = pd.merge(household_heads, households[l.cols_id_hogar], how='left')
    households_heads_df = pd.DataFrame(_household_heads.groupby(l.cols_id_hogar)[[
        'CH04', 'CH06', 'ESTADO', 'NIVEL_ED', 'PP02E', 'CAT_OCUP', 'PP07I', 'PP07H', 'PP04B1']].sum()).reset_index()
    # non-single-parent households by gender and occupation of the spouse
    _spouses = pd.merge(spouses, households[l.cols_id_hogar], how='left')
    households_spouses_df = pd.DataFrame(_spouses.groupby(l.cols_id_hogar)[[
        'CH04', 'ESTADO']].sum()).reset_index()
    return households_heads_df, households_spouses_df


def join_heads_spouses(data: pd.DataFrame,
                        households_heads_df: pd.DataFrame,
                        households_spouses_df: pd.DataFrame):
    _data = pr.join_individuals_households(data,
                                     households_heads_df,
                                     suffixes=('', '_jefx'))
    students = pr.join_individuals_households(_data,
                                           households_spouses_df,
                                           how='outer',
                                           suffixes=('', '_conyuge'))
    students.fillna(0)
    return students


def generate_jefe_trabaja(data: pd.DataFrame):
    cond = data.ESTADO_jefx == 1
    students = pr.create_binary_feature(data, 'JEFE_TRABAJA', cond)
    return students


def generate_jefa_mujer(data: pd.DataFrame):
    cond = data.CH04_jefx == 2
    students = pr.create_binary_feature(data, 'JEFA_MUJER', cond)
    return students


def generate_hogar_monop(data: pd.DataFrame):
    cond = data.ESTADO_conyuge.isna()
    students = pr.create_binary_feature(data, 'HOGAR_MONOP', cond)
    return students


def generate_conyuge_trabaja(data: pd.DataFrame):
    cond = data.ESTADO_conyuge == 1
    students = pr.create_binary_feature(data, 'CONYUGE_TRABAJA', cond)
    return students


def generate_nbi_vivienda_precaria(data: pd.DataFrame):
    cond = (data["IV3"] > 2) | ((data["IV4"] > 4) & (data["IV5"] == 2))
    students = pr.create_binary_feature(data, 'NBI_VIVIENDA', cond)
    return students


def generate_nbi_hacinamiento(data: pd.DataFrame):
    cond = data.IX_TOT / data.IV2 >= 3
    students = pr.create_binary_feature(data, 'NBI_HACINAMIENTO', cond)
    return students


def generate_nbi_tenencia(data: pd.DataFrame):
    cond = data['II7'].isin([3, 7])
    students = pr.create_binary_feature(data, 'NBI_TENENCIA', cond)
    return students


def generate_nbi_sanitaria(data: pd.DataFrame):
    cond = (data['IV8'] == 2) | (data['IV10'] > 1)
    students = pr.create_binary_feature(data, 'NBI_SANITARIA', cond)
    return students


def generate_nbi_zona_vulnerable(data: pd.DataFrame):
    cond = (data['IV12_1'] == 1) | (data['IV12_3'] == 1)
    students = pr.create_binary_feature(data, 'NBI_ZONA_VULNERABLE', cond)
    return students


def generate_nbi_dificultad_laboral(data: pd.DataFrame):
    male_condition = ((data.CH06_jefx.between(16, 64)) & (data.CH04_jefx == 1)) & (
        (data.ESTADO_jefx == 2) | (data.PP02E_jefx.isin([3, 5])))
    female_condition = ((data.CH06_jefx.between(16, 59)) & (data.CH04_jefx == 2)) & (
        (data.ESTADO_jefx == 2) | (data.PP02E_jefx.isin([3, 5])))
    data.loc[:, 'NBI_DIFLABORAL'] = np.nan
    data.loc[male_condition, 'NBI_DIFLABORAL'] = 1
    data.loc[female_condition, 'NBI_DIFLABORAL'] = 1
    data.loc[:, 'NBI_DIFLABORAL'].fillna(0, inplace=True)
    return data


def generate_nbi_trabajo_precario(data: pd.DataFrame):
    cond = ((data.CAT_OCUP == 2) & (data.NIVEL_ED.isin([1, 2, 3, 7]))) | (
        (data.PP07I == 2) | (data.PP07H == 2)) | (
        (data.PP04B1 == 1) & ((data.PP07I == 2) | (data.PP07H == 2)))
    students = pr.create_binary_feature(data, 'NBI_TRABAJO_PRECARIO', cond)
    return students


def generate_ratio_occupied_members(data: pd.DataFrame,
                                    individuals: pd.DataFrame,
                                    households: pd.DataFrame):
    occupied_per_household = individuals[individuals.ESTADO == 1].groupby(['CODUSU', 'NRO_HOGAR'])['ESTADO'].sum().reset_index()
    occupied_per_household.rename({'ESTADO': 'nro_ocupados'}, axis=1, inplace=True)
    occupied = pd.merge(households, occupied_per_household)
    occupied.loc[:, 'ratio_ocupados'] = occupied.nro_ocupados / occupied.IX_TOT
    cols = l.cols_id_hogar + ['ratio_ocupados']
    students = pd.merge(data, occupied[cols], how='left')
    students.ratio_ocupados.fillna(0, inplace=True)
    return students


def generate_nbi_subsistencia(data: pd.DataFrame):
    cond = data.NIVEL_ED_jefx.isin([1, 2, 3, 7]) & data.ratio_ocupados >= 4
    students = pr.create_binary_feature(data, 'NBI_SUBSISTENCIA', cond)
    return students


def generate_nbi_cobertura_previsional(data: pd.DataFrame):
    cond = ((data.CH06 >= 65) & (data.CH04 == 1) & (data.V2_M == 0)) | (
        (data.CH06 >= 60) & (data.CH04 == 2) & (data.V2_M == 0))
    students = pr.create_binary_feature(
        data, 'NBI_COBERTURA_PREVISIONAL', cond)
    return students

def generate_dropout_target(data_t: pd.DataFrame,
                    individuals_tp1: pd.DataFrame)->pd.DataFrame:
    cols_tp1 = ['CODUSU', 'NRO_HOGAR', 'COMPONENTE'] + ['CH10']
    data = pd.merge(data_t, individuals_tp1[cols_tp1],
                    on=['CODUSU', 'NRO_HOGAR', 'COMPONENTE'],
                    suffixes=('', '_fin'),
                    how='inner')
    cond = (data.CH10 == 1) & (data.CH10_fin == 2 )
    students = pr.create_binary_feature(data, 'DESERTO', cond)
    students.drop(['CH10_fin'], axis=1, inplace=True)
    return students

def homogenize_binary_columns(df, columns):
    replace_dict = {col: {2: 0, 'S': 1, 'N': 0, 'NO': 0} for col in columns}
    data = df.replace(replace_dict)
    data[columns].astype('float64', copy=False)
    return data

def distance_to_capital(df, agglomeration='AGLOMERADO'):
    agglomeration_coords = {
        'eph_codagl': [13, 29, 31, 25, 34, 7, 26, 15, 4, 91, 18, 23, 30, 12, 20, 93, 8, 14, 6, 5, 3, 9, 22, 36, 38, 38, 10, 19, 2, 32, 17, 33, 27],
        'eph_aglome': ['Gran Córdoba', 'Gran Tucumán - Tafi Viejo', 'Ushuaia - Rio Grande', 'La Rioja', 'Mar del Plata - Batán', 'Posadas', 'San Luis - El Chorrillo', 'Formosa', 'Gran Rosario', 'Rawson - Trelew', 'Santiago del Estero - La Banda', 'Salta', 'Santa Rosa - Toay', 'Corrientes', 'Rio Gallegos', 'Viedma - Carmen de Patagones', 'Gran Resistencia', 'Concordia', 'Gran Paraná', 'Gran Santa Fe', 'Bahia Blanca - Cerri', 'Comodoro Rivadavia - Rada Tilly', 'Gran Catamarca', 'Rio Cuarto', 'San Nicolas - Villa Constitiución', 'San Nicolas - Villa Constitiución', 'Gran Mendoza', 'Jujuy - Palpalá', 'Gran La Plata', 'CABA', 'Neuquén - Plottier', 'Partidos del GBA', 'Gran San Juan'],
        'x': [3.668196e+06, 3.575528e+06, 3.368650e+06, 3.433735e+06, 4.238702e+06, 4.500828e+06, 3.470658e+06, 4.290676e+06, 3.989413e+06, 3.561793e+06, 3.671129e+06, 3.558820e+06, 3.652175e+06, 4.215852e+06, 3.274586e+06, 3.754150e+06, 4.194276e+06, 4.258864e+06, 4.023207e+06, 4.008905e+06, 3.824880e+06, 3.382300e+06, 3.520630e+06, 3.656809e+06, 4.035477e+06, 4.029783e+06, 3.231898e+06, 3.560475e+06, 4.234043e+06, 4.193488e+06, 3.310530e+06, 4.180647e+06, 3.260047e+06],
        'y': [6.533650e+06, 7.036009e+06, 3.980855e+06, 6.726538e+06, 5.760505e+06, 6.922012e+06, 6.318389e+06, 7.090830e+06, 6.346344e+06, 5.211965e+06, 6.926946e+06, 7.258796e+06, 5.947417e+06, 6.941437e+06, 4.274342e+06, 5.478764e+06, 6.944114e+06, 6.501289e+06, 6.473109e+06, 6.487715e+06, 5.709612e+06, 4.921987e+06, 6.858955e+06, 6.336869e+06, 6.293741e+06, 6.307894e+06, 6.360832e+06, 7.329096e+06, 6.103368e+06, 6.144082e+06, 5.686581e+06, 6.148549e+06, 6.509854e+06]
    }
    agglomeration_coords_df = pd.DataFrame(agglomeration_coords)
    agglomeration_df = df.copy()
    agglomeration_df['x_temp'] = None
    agglomeration_df['y_temp'] = None
    agglomeration_df['distance'] = None

    # Coordenadas de la capital
    capital_x = 4.193488e+06
    capital_y = 6.144082e+06
    for index, row in agglomeration_df.iterrows():
        eph_codagl = row[agglomeration]
        matching_row = agglomeration_coords_df[agglomeration_coords_df['eph_codagl'] == eph_codagl]
        if not matching_row.empty:
            x_temp = matching_row['x'].values[0]
            y_temp = matching_row['y'].values[0]
            distance = np.sqrt((x_temp - capital_x)**2 + (y_temp - capital_y)**2)
            agglomeration_df.at[index, 'distance'] = distance

    # Eliminar columnas temporales 'x_temp' e 'y_temp'
    agglomeration_df['AGLOMERADO'] = agglomeration_df['distance']
    agglomeration_df.drop(columns=['x_temp', 'y_temp', 'distance'], inplace=True)
    return agglomeration_df

def preprocess_data(data: pd.DataFrame, train_test: bool = True) -> pd.DataFrame:
    if len(data) > 1:
        data_concat = pd.concat(data)
    if len(data) == 1:
        data_concat = data[0]
    drop_cols = [
        'IV8', 'IX_MAYEQ10', 'CAT_OCUP', 'CAT_INAC', 'CAT_OCUP_jefx', 'JEFE_TRABAJA', 'T_VI', 'V2_M', 'CH04_conyuge', 'CH04_jefx', 'NBI_SUBSISTENCIA', 'IV10', 'II7', 'IV12_1', 'IV12_3', 'PP07I',
        'PP07H', 'PP02E_jefx', 'REALIZADA_jefx', 'REALIZADA_conyuge',
        'H15', 'ITF', 'REALIZADA'
    ]
    if train_test is True:
        binary_columns = [
            'CH11', 'PP02H', 'PP04B1', 'IV5', 'IV12_2', 'II3', 'II4_1', 'II4_2', 'II4_3', 'V1', 'V2', 'V21', 'V22', 'V3', 'V5', 'V6', 'V7', 'V8', 'V11', 'V12', 'V13', 'V14', 'PP07I_jefx', 'PP07H_jefx', 'PP04B1_jefx', 'CONYUGE_TRABAJA', 'JEFA_MUJER', 'HOGAR_MONOP', 'NBI_COBERTURA_PREVISIONAL', 'NBI_DIFLABORAL', 'NBI_HACINAMIENTO', 'NBI_SANITARIA', 'NBI_TENENCIA', 'NBI_TRABAJO_PRECARIO', 'NBI_VIVIENDA', 'NBI_ZONA_VULNERABLE', 'DESERTO', 'MAS_500', 'CH04'
        ]
    if train_test is False:
        binary_columns = [
            'CH11', 'PP02H', 'PP04B1', 'IV5', 'IV12_2', 'II3', 'II4_1', 'II4_2', 'II4_3', 'V1', 'V2', 'V21', 'V22', 'V3', 'V5', 'V6', 'V7', 'V8', 'V11', 'V12', 'V13', 'V14', 'PP07I_jefx', 'PP07H_jefx', 'PP04B1_jefx', 'CONYUGE_TRABAJA', 'JEFA_MUJER', 'HOGAR_MONOP', 'NBI_COBERTURA_PREVISIONAL', 'NBI_DIFLABORAL', 'NBI_HACINAMIENTO', 'NBI_SANITARIA', 'NBI_TENENCIA', 'NBI_TRABAJO_PRECARIO', 'NBI_VIVIENDA', 'NBI_ZONA_VULNERABLE', 'MAS_500', 'CH04'
        ]
    data_concat = data_concat[data_concat['H15'] == 1]
    data_concat.drop(drop_cols, axis=1, inplace=True)
    data_concat = homogenize_binary_columns(data_concat, binary_columns)
    # PP04B1 --> renombre a servicio_domestico + reemplazo de valores
    data_concat.loc[:, 'PP04B1'].replace({2: 0, np.nan: 0}, inplace=True)
    data_concat.rename({'PP04B1': 'servicio_domestico'},
                axis=1, inplace=True)
    # PP07H_jefx renombre a apotes_jubilatorios_jefx
    data_concat.rename({'PP07H_jefx': 'APORTES_JUBILATORIOS_jefx'},
                axis=1, inplace=True)
    # variables conyuge --> lleno NaN con ceros pues corresponden a HOGAR_MONOP==1
    cvars = data_concat.columns.str.endswith('_conyuge')
    data_concat[data_concat.columns[cvars]] = data_concat.loc[:, cvars].fillna(0)
    data_concat = distance_to_capital(data_concat)
    return data_concat

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