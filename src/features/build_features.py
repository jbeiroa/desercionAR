"""
This module provides functions to generate features for the school dropout model.
"""

import pandas as pd
from ..data import preprocessing as pr


def generate_auxiliary_dataframes(data: pd.DataFrame, households: pd.DataFrame):
    """Generates auxiliary dataframes for household heads and spouses."""
    household_heads = data[data.CH03 == 1]
    spouses = data[data.CH03 == 2]
    # households by head's occupation status
    _household_heads = pd.merge(
        household_heads, households[pr.id_house_features], how="left"
    )
    households_heads_df = pd.DataFrame(
        _household_heads.groupby(pr.id_house_features)[
            [
                "CH04",
                "CH06",
                "ESTADO",
                "NIVEL_ED",
                "PP07I",
                "PP07H",
                "PP04B1",
                "PP02E",
                "CAT_OCUP",
            ]
        ].sum()
    ).reset_index()
    # non-single-parent households by gender and occupation of the spouse
    _spouses = pd.merge(spouses, households[pr.id_house_features], how="left")
    households_spouses_df = pd.DataFrame(
        _spouses.groupby(pr.id_house_features)[["ESTADO"]].sum()
    ).reset_index()
    return households_heads_df, households_spouses_df


def join_heads_spouses(
    data: pd.DataFrame,
    households_heads_df: pd.DataFrame,
    households_spouses_df: pd.DataFrame,
):
    """Joins household heads and spouses data with individual records.

    Args:
        data (pd.DataFrame): Individual data to join with.
        households_heads_df (pd.DataFrame): Aggregated household heads data.
        households_spouses_df (pd.DataFrame): Aggregated spouses data.

    Returns:
        pd.DataFrame: Joined DataFrame with household heads and spouses information.
    """
    _data = pr.join_individuals_households(
        data, households_heads_df, suffixes=("", "_jefx")
    )
    students = pr.join_individuals_households(
        _data, households_spouses_df, how="outer", suffixes=("", "_conyuge")
    )
    return students


def generate_jefa_mujer(data: pd.DataFrame) -> pd.DataFrame:
    """Generates a binary feature indicating if the household head is female."""
    cond = data.CH04_jefx == 2
    students = pr.create_binary_feature(data, "JEFA_MUJER", cond)
    return students


def generate_hogar_monop(data: pd.DataFrame) -> pd.DataFrame:
    """Generates a binary feature indicating if the household is single-parent."""
    cond = data.ESTADO_conyuge.isna()
    students = pr.create_binary_feature(data, "HOGAR_MONOP", cond)
    return students


def generate_conyuge_trabaja(data: pd.DataFrame) -> pd.DataFrame:
    """Generates a binary feature indicating if the spouse is employed."""
    cond = data.ESTADO_conyuge == 1
    students = pr.create_binary_feature(data, "CONYUGE_TRABAJA", cond)
    return students


def generate_nbi_vivienda_precaria(data: pd.DataFrame) -> pd.DataFrame:
    """Generates a binary feature for NBI 'precarious housing'."""
    cond = (data["IV3"] > 2) | ((data["IV4"] > 4) & (data["IV5"] == 2))
    students = pr.create_binary_feature(data, "NBI_VIVIENDA", cond)
    return students


def generate_nbi_hacinamiento(data: pd.DataFrame) -> pd.DataFrame:
    """Generates a binary feature for NBI 'overcrowding'."""
    cond = data.IX_TOT / data.IV2 >= 3
    students = pr.create_binary_feature(data, "NBI_HACINAMIENTO", cond)
    return students


def generate_nbi_tenencia(data: pd.DataFrame) -> pd.DataFrame:
    """Generates a binary feature for NBI 'insecure tenure'."""
    cond = data["II7"].isin([3, 7])
    students = pr.create_binary_feature(data, "NBI_TENENCIA", cond)
    return students


def generate_nbi_sanitaria(data: pd.DataFrame) -> pd.DataFrame:
    """Generates a binary feature for NBI 'inadequate sanitation'."""
    cond = (data["IV8"] == 2) | (data["IV10"] > 1)
    students = pr.create_binary_feature(data, "NBI_SANITARIA", cond)
    return students


def generate_nbi_zona_vulnerable(data: pd.DataFrame) -> pd.DataFrame:
    """Generates a binary feature for NBI 'vulnerable zone'."""
    cond = (data["IV12_1"] == 1) | (data["IV12_3"] == 1)
    students = pr.create_binary_feature(data, "NBI_ZONA_VULNERABLE", cond)
    return students


def generate_nbi_dificultad_laboral(data: pd.DataFrame) -> pd.DataFrame:
    """Generates binary feature for labor difficulty NBI."""
    male_condition = ((data.CH06_jefx.between(16, 64)) & (data.CH04_jefx == 1)) & (
        (data.ESTADO_jefx == 2) | (data.PP02E_jefx.isin([3, 5]))
    )
    female_condition = ((data.CH06_jefx.between(16, 59)) & (data.CH04_jefx == 2)) & (
        (data.ESTADO_jefx == 2) | (data.PP02E_jefx.isin([3, 5]))
    )
    data["NBI_DIFLABORAL"] = 0.0
    data.loc[male_condition, "NBI_DIFLABORAL"] = 1
    data.loc[female_condition, "NBI_DIFLABORAL"] = 1
    return data


def generate_nbi_trabajo_precario(data: pd.DataFrame) -> pd.DataFrame:
    """Generates a binary feature for NBI 'precarious work'."""
    cond = (
        ((data.CAT_OCUP == 2) & (data.NIVEL_ED.isin([1, 2, 3, 7])))
        | ((data.PP07I == 2) | (data.PP07H == 2))
        | ((data.PP04B1 == 1) & ((data.PP07I == 2) | (data.PP07H == 2)))
    )
    students = pr.create_binary_feature(data, "NBI_TRABAJO_PRECARIO", cond)
    return students


def generate_ratio_ocupados(
    data: pd.DataFrame, individuals: pd.DataFrame, households: pd.DataFrame
) -> pd.DataFrame:
    """Generates ratio of occupied household members."""
    occupied_per_household = (
        individuals[individuals.ESTADO == 1]
        .groupby(["CODUSU", "NRO_HOGAR"])["ESTADO"]
        .sum()
        .reset_index()
    )
    occupied_per_household.rename({"ESTADO": "nro_ocupados"}, axis=1, inplace=True)
    occupied = pd.merge(households, occupied_per_household)
    occupied.loc[:, "ratio_ocupados"] = occupied.nro_ocupados / occupied.IX_TOT
    cols = pr.id_house_features + ["ratio_ocupados"]
    students = pd.merge(data, occupied[cols], how="left")
    students.loc[:, "ratio_ocupados"] = students["ratio_ocupados"].fillna(0)
    return students


def generate_nbi_cobertura_previsional(data: pd.DataFrame) -> pd.DataFrame:
    """Generates a binary feature for NBI 'pension coverage'."""
    cond = ((data.CH06 >= 65) & (data.CH04 == 1) & (data.V2_M == 0)) | (
        (data.CH06 >= 60) & (data.CH04 == 2) & (data.V2_M == 0)
    )
    students = pr.create_binary_feature(data, "NBI_COBERTURA_PREVISIONAL", cond)
    return students
