import pandas as pd
import numpy as np
from features import build_features as bf
from data import preprocessing as pr

def test_generate_auxiliary_dataframes_controlled_inputs(mock_individuals_df: pd.DataFrame, mock_households_df: pd.DataFrame, expected_heads_df: pd.DataFrame, expected_spouses_df: pd.DataFrame):
    """
    Tests that auxiliary dataframes for household heads and spouses are generated
    correctly using small, controlled input DataFrames and asserts against
    explicitly defined expected outputs.
    """
    heads_df, spouses_df = bf.generate_auxiliary_dataframes(mock_individuals_df, mock_households_df)

    pd.testing.assert_frame_equal(heads_df, expected_heads_df)
    pd.testing.assert_frame_equal(spouses_df, expected_spouses_df)


def test_join_heads_spouses_controlled_inputs(mock_individuals_df: pd.DataFrame, expected_heads_df: pd.DataFrame, expected_spouses_df: pd.DataFrame, expected_joined_df: pd.DataFrame):
    """
    Tests that join_heads_spouses correctly merges individual, heads, and spouses data.
    """
    result_df = bf.join_heads_spouses(mock_individuals_df, expected_heads_df, expected_spouses_df)

    for col in expected_joined_df.columns:
        if col in result_df.columns and expected_joined_df[col].dtype != result_df[col].dtype:
            if expected_joined_df[col].dtype == 'float64' and result_df[col].dtype == 'int64':
                result_df[col] = result_df[col].astype('float64')
            elif expected_joined_df[col].dtype == 'int64' and result_df[col].dtype == 'float64' and not result_df[col].isnull().any():
                result_df[col] = result_df[col].astype('int64')

    pd.testing.assert_frame_equal(result_df, expected_joined_df, check_dtype=True)

def test_generate_jefa_mujer(expected_joined_df: pd.DataFrame, expected_jefa_mujer_df: pd.DataFrame):
    """
    Tests that generate_jefa_mujer correctly creates the 'JEFA_MUJER' binary feature.
    """
    result_df = bf.generate_jefa_mujer(expected_joined_df.copy()) # Use a copy to prevent in-place modification of fixture

    pd.testing.assert_frame_equal(result_df, expected_jefa_mujer_df, check_dtype=True)

def test_generate_hogar_monop(expected_joined_df: pd.DataFrame, expected_hogar_monop_df: pd.DataFrame):
    """
    Tests that generate_hogar_monop correctly creates the 'HOGAR_MONOP' binary feature.
    """
    result_df = bf.generate_hogar_monop(expected_joined_df.copy()) # Use a copy to prevent in-place modification of fixture

    pd.testing.assert_frame_equal(result_df, expected_hogar_monop_df, check_dtype=True)

def test_generate_conyuge_trabaja(expected_joined_df: pd.DataFrame, expected_conyuge_trabaja_df: pd.DataFrame):
    """
    Tests that generate_conyuge_trabaja correctly creates the 'CONYUGE_TRABAJA' binary feature.
    """
    result_df = bf.generate_conyuge_trabaja(expected_joined_df.copy()) # Use a copy to prevent in-place modification of fixture

    pd.testing.assert_frame_equal(result_df, expected_conyuge_trabaja_df, check_dtype=True)

def test_generate_nbi_vivienda_precaria(mock_data_for_nbi_vivienda: pd.DataFrame, expected_nbi_vivienda_df: pd.DataFrame):
    """
    Tests that generate_nbi_vivienda_precaria correctly creates the 'NBI_VIVIENDA' binary feature.
    """
    result_df = bf.generate_nbi_vivienda_precaria(mock_data_for_nbi_vivienda.copy()) # Use a copy to prevent in-place modification of fixture

    pd.testing.assert_frame_equal(result_df, expected_nbi_vivienda_df, check_dtype=True)

def test_generate_nbi_hacinamiento(mock_data_for_nbi_hacinamiento: pd.DataFrame, expected_nbi_hacinamiento_df: pd.DataFrame):
    """
    Tests that generate_nbi_hacinamiento correctly creates the 'NBI_HACINAMIENTO' binary feature.
    """
    result_df = bf.generate_nbi_hacinamiento(mock_data_for_nbi_hacinamiento.copy()) # Use a copy to prevent in-place modification of fixture

    pd.testing.assert_frame_equal(result_df, expected_nbi_hacinamiento_df, check_dtype=True)

def test_generate_nbi_tenencia(mock_data_for_nbi_tenencia: pd.DataFrame, expected_nbi_tenencia_df: pd.DataFrame):
    """
    Tests that generate_nbi_tenencia correctly creates the 'NBI_TENENCIA' binary feature.
    """
    result_df = bf.generate_nbi_tenencia(mock_data_for_nbi_tenencia.copy()) # Use a copy to prevent in-place modification of fixture

    pd.testing.assert_frame_equal(result_df, expected_nbi_tenencia_df, check_dtype=True)

def test_generate_nbi_sanitaria(mock_data_for_nbi_sanitaria: pd.DataFrame, expected_nbi_sanitaria_df: pd.DataFrame):
    """
    Tests that generate_nbi_sanitaria correctly creates the 'NBI_SANITARIA' binary feature.
    """
    result_df = bf.generate_nbi_sanitaria(mock_data_for_nbi_sanitaria.copy()) # Use a copy to prevent in-place modification of fixture

    pd.testing.assert_frame_equal(result_df, expected_nbi_sanitaria_df, check_dtype=True)

def test_generate_nbi_zona_vulnerable(mock_data_for_nbi_zona_vulnerable: pd.DataFrame, expected_nbi_zona_vulnerable_df: pd.DataFrame):
    """
    Tests that generate_nbi_zona_vulnerable correctly creates the 'NBI_ZONA_VULNERABLE' binary feature.
    """
    result_df = bf.generate_nbi_zona_vulnerable(mock_data_for_nbi_zona_vulnerable.copy()) # Use a copy to prevent in-place modification of fixture

    pd.testing.assert_frame_equal(result_df, expected_nbi_zona_vulnerable_df, check_dtype=True)

def test_generate_nbi_dificultad_laboral(mock_data_for_nbi_dificultad_laboral: pd.DataFrame, expected_nbi_dificultad_laboral_df: pd.DataFrame):
    """
    Tests that generate_nbi_dificultad_laboral correctly creates the 'NBI_DIFLABORAL' binary feature.
    """
    result_df = bf.generate_nbi_dificultad_laboral(mock_data_for_nbi_dificultad_laboral.copy()) # Use a copy to prevent in-place modification of fixture

    pd.testing.assert_frame_equal(result_df, expected_nbi_dificultad_laboral_df, check_dtype=True)

def test_generate_nbi_trabajo_precario(mock_data_for_nbi_trabajo_precario: pd.DataFrame, expected_nbi_trabajo_precario_df: pd.DataFrame):
    """
    Tests that generate_nbi_trabajo_precario correctly creates the 'NBI_TRABAJO_PRECARIO' binary feature.
    """
    result_df = bf.generate_nbi_trabajo_precario(mock_data_for_nbi_trabajo_precario.copy()) # Use a copy to prevent in-place modification of fixture

    pd.testing.assert_frame_equal(result_df, expected_nbi_trabajo_precario_df, check_dtype=True)

def test_generate_nbi_cobertura_previsional(mock_data_for_nbi_cobertura_previsional: pd.DataFrame, expected_nbi_cobertura_previsional_df: pd.DataFrame):
    """
    Tests that generate_nbi_cobertura_previsional correctly creates the 'NBI_COBERTURA_PREVISIONAL' binary feature.
    """
    result_df = bf.generate_nbi_cobertura_previsional(mock_data_for_nbi_cobertura_previsional.copy()) # Use a copy to prevent in-place modification of fixture

    pd.testing.assert_frame_equal(result_df, expected_nbi_cobertura_previsional_df, check_dtype=True)


def test_generate_ratio_ocupados(
    mock_data_for_ratio_ocupados: pd.DataFrame,
    mock_individuals_for_ratio_ocupados: pd.DataFrame,
    mock_households_for_ratio_ocupados: pd.DataFrame,
    expected_ratio_ocupados_df: pd.DataFrame,
):
    """
    Tests that generate_ratio_ocupados correctly calculates the ratio of employed household members.
    """
    # Act
    result_df = bf.generate_ratio_ocupados(
        mock_data_for_ratio_ocupados,
        mock_individuals_for_ratio_ocupados,
        mock_households_for_ratio_ocupados
    )

    # Assert
    pd.testing.assert_frame_equal(result_df, expected_ratio_ocupados_df)