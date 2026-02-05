import pytest
import pandas as pd
import numpy as np 
from data import preprocessing as pr 
from features import build_features as bf

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
    return pd.read_csv("tests/fixtures/eph_fixture.csv", sep=",", header=0,
                       dtype={102: str, 105: str, 179: str, 182: str, 187: str, 207: str, 209: str})

@pytest.fixture(scope="module")
def mock_households_df() -> pd.DataFrame:
    """A small, controlled DataFrame representing household data."""
    return pd.DataFrame({
        "CODUSU": ["H1", "H2"],
        "NRO_HOGAR": [1, 1],
        "REALIZADA": [1, 1],
        "ANO4": [2023, 2023],
        "TRIMESTRE": [1, 1],
        "REGION": [1, 1],
        "MAS_500": [1, 1],
        "AGLOMERADO": [1, 1],
        "PONDERA": [100, 100]
    })

@pytest.fixture(scope="module")
def mock_individuals_df() -> pd.DataFrame:
    """A small, controlled DataFrame representing individual data."""
    return pd.DataFrame({
        "CODUSU":       ["H1", "H1", "H1", "H2"],
        "NRO_HOGAR":    [1, 1, 1, 1],
        "COMPONENTE":   [1, 2, 3, 1],
        "CH03":         [1, 2, 3, 1],
        "CH04":         [1, 2, 1, 2],
        "CH06":         [40, 38, 15, 50],
        "ESTADO":       [1, 1, 4, 2],
        "NIVEL_ED":     [5, 5, 2, 4],
        "PP07I":        [1, 0, 0, 0],
        "PP07H":        [0, 1, 0, 0],
        "PP04B1":       [1, 1, 0, 2],
        "PP02E":        [0, 0, 0, 0],
        "CAT_OCUP":     [3, 3, 0, 2],
        "REALIZADA": [1, 1, 1, 1],
        "ANO4": [2023, 2023, 2023, 2023], "TRIMESTRE": [1, 1, 1, 1], "REGION": [1, 1, 1, 1],
        "MAS_500": [1, 1, 1, 1], "AGLOMERADO": [1, 1, 1, 1], "PONDERA": [100, 100, 100, 100]
    })

@pytest.fixture(scope="module")
def expected_heads_df() -> pd.DataFrame:
    """Expected output for household heads from generate_auxiliary_dataframes."""
    return pd.DataFrame({
        "CODUSU": ["H1", "H2"],
        "NRO_HOGAR": [1, 1],
        "REALIZADA": [1, 1],
        "ANO4": [2023, 2023],
        "TRIMESTRE": [1, 1],
        "REGION": [1, 1],
        "MAS_500": [1, 1],
        "AGLOMERADO": [1, 1],
        "PONDERA": [100, 100],
        "CH04": [1, 2],
        "CH06": [40, 50],
        "ESTADO": [1, 2],
        "NIVEL_ED": [5, 4],
        "PP07I": [1, 0],
        "PP07H": [0, 0],
        "PP04B1": [1, 2],
        "PP02E": [0, 0],
        "CAT_OCUP": [3, 2],
    })

@pytest.fixture(scope="module")
def expected_spouses_df() -> pd.DataFrame:
    """Expected output for spouses from generate_auxiliary_dataframes."""
    return pd.DataFrame({
        "CODUSU": ["H1"],
        "NRO_HOGAR": [1],
        "REALIZADA": [1],
        "ANO4": [2023],
        "TRIMESTRE": [1],
        "REGION": [1],
        "MAS_500": [1],
        "AGLOMERADO": [1],
        "PONDERA": [100],
        "ESTADO": [1],
    })

@pytest.fixture(scope="module")
def expected_joined_df(mock_individuals_df: pd.DataFrame, expected_heads_df: pd.DataFrame, expected_spouses_df: pd.DataFrame) -> pd.DataFrame:
    """Expected final output from join_heads_spouses, manually created."""
    _data_after_heads_merge = pd.merge(
        mock_individuals_df,
        expected_heads_df,
        on=pr.join_features,
        how="inner",
        suffixes=("", "_jefx")
    )

    _expected_output_df = pd.merge(
        _data_after_heads_merge,
        expected_spouses_df,
        on=pr.join_features,
        how="outer",
        suffixes=("", "_conyuge")
    )

    for col in _expected_output_df.columns:
        if _expected_output_df[col].dtype == 'int64' and _expected_output_df[col].isnull().any():
            _expected_output_df[col] = _expected_output_df[col].astype('float64')

    return _expected_output_df

@pytest.fixture(scope="module")
def expected_jefa_mujer_df(expected_joined_df: pd.DataFrame) -> pd.DataFrame:
    """Expected output after applying generate_jefa_mujer."""
    df = expected_joined_df.copy()
    # H1 rows have CH04_jefx = 1, H2 row has CH04_jefx = 2
    # JEFA_MUJER should be 0.0 for H1 components, 1.0 for H2 component
    df.loc[:, "JEFA_MUJER"] = np.where(df["CH04_jefx"] == 2, 1.0, 0.0)

    # Ensure dtypes are consistent
    for col in df.columns:
        if df[col].dtype == 'int64' and df[col].isnull().any():
            df[col] = df[col].astype('float64')

    return df

@pytest.fixture(scope="module")
def expected_hogar_monop_df(expected_joined_df: pd.DataFrame) -> pd.DataFrame:
    """Expected output after applying generate_hogar_monop."""
    df = expected_joined_df.copy()
    # H1 rows have ESTADO_conyuge = 1.0, H2 row has ESTADO_conyuge = NaN
    # HOGAR_MONOP should be 0.0 for H1 components, 1.0 for H2 component
    df.loc[:, "HOGAR_MONOP"] = np.where(df["ESTADO_conyuge"].isna(), 1.0, 0.0)

    # Ensure dtypes are consistent
    for col in df.columns:
        if df[col].dtype == 'int64' and df[col].isnull().any():
            df[col] = df[col].astype('float64')

    return df

@pytest.fixture(scope="module")
def expected_conyuge_trabaja_df(expected_joined_df: pd.DataFrame) -> pd.DataFrame:
    """Expected output after applying generate_conyuge_trabaja."""
    df = expected_joined_df.copy()
    # H1 rows have ESTADO_conyuge = 1.0, H2 row has ESTADO_conyuge = NaN
    # CONYUGE_TRABAJA should be 1.0 for H1 components, 0.0 for H2 component
    df.loc[:, "CONYUGE_TRABAJA"] = np.where(df["ESTADO_conyuge"] == 1, 1.0, 0.0)

    # Ensure dtypes are consistent
    for col in df.columns:
        if df[col].dtype == 'int64' and df[col].isnull().any():
            df[col] = df[col].astype('float64')

    return df

@pytest.fixture(scope="module")
def mock_data_for_nbi_vivienda() -> pd.DataFrame:
    """A small DataFrame to test generate_nbi_vivienda_precaria."""
    return pd.DataFrame({
        "CODUSU": ["H1", "H2", "H3", "H4", "H5", "H6"],
        "NRO_HOGAR": [1, 1, 1, 1, 1, 1],
        "IV3": [3, 1, 4, 1, 1, 2],
        "IV4": [1, 5, 6, 1, 5, np.nan], # np.nan will make this float64
        "IV5": [1, 2, 2, 1, 1, 2],
    })

@pytest.fixture(scope="module")
def expected_nbi_vivienda_df(mock_data_for_nbi_vivienda: pd.DataFrame) -> pd.DataFrame:
    """Expected output after applying generate_nbi_vivienda_precaria."""
    df = mock_data_for_nbi_vivienda.copy()
    df.loc[:, "NBI_VIVIENDA"] = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]

    # Force IV4 to be float64 in expected_df because it has np.nan in mock_data_for_nbi_vivienda
    df['IV4'] = df['IV4'].astype('float64')
    
    return df

@pytest.fixture(scope="module")
def mock_data_for_nbi_hacinamiento() -> pd.DataFrame:
    """A small DataFrame to test generate_nbi_hacinamiento."""
    return pd.DataFrame({
        "CODUSU": ["H1", "H2", "H3", "H4", "H5", "H6"],
        "NRO_HOGAR": [1, 1, 1, 1, 1, 1],
        "IX_TOT": [6, 7, 5, 10, np.nan, 6],
        "IV2": [2, 2, 2, 0, 2, np.nan], # IV2=0 to test division by zero
    })

@pytest.fixture(scope="module")
def expected_nbi_hacinamiento_df(mock_data_for_nbi_hacinamiento: pd.DataFrame) -> pd.DataFrame:
    """Expected output after applying generate_nbi_hacinamiento."""
    df = mock_data_for_nbi_hacinamiento.copy()
    df.loc[:, "NBI_HACINAMIENTO"] = [1.0, 1.0, 0.0, 1.0, 0.0, 0.0]

    # Force relevant columns to float64 as they contain np.nan or will contain inf
    df['IX_TOT'] = df['IX_TOT'].astype('float64')
    return df

@pytest.fixture(scope="module")
def mock_data_for_nbi_tenencia() -> pd.DataFrame:
    """A small DataFrame to test generate_nbi_tenencia."""
    return pd.DataFrame({
        "CODUSU": ["H1", "H2", "H3", "H4", "H5"],
        "NRO_HOGAR": [1, 1, 1, 1, 1],
        "II7": [3, 7, 1, 2, np.nan],
    })

@pytest.fixture(scope="module")
def expected_nbi_tenencia_df(mock_data_for_nbi_tenencia: pd.DataFrame) -> pd.DataFrame:
    """Expected output after applying generate_nbi_tenencia."""
    df = mock_data_for_nbi_tenencia.copy()
    df.loc[:, "NBI_TENENCIA"] = [1.0, 1.0, 0.0, 0.0, 0.0]

    # Force II7 to float64 due to np.nan in input
    return df

@pytest.fixture(scope="module")
def mock_data_for_nbi_sanitaria() -> pd.DataFrame:
    """A small DataFrame to test generate_nbi_sanitaria."""
    return pd.DataFrame({
        "CODUSU": ["H1", "H2", "H3", "H4", "H5", "H6", "H7"],
        "NRO_HOGAR": [1, 1, 1, 1, 1, 1, 1],
        "IV8": [2, 1, 2, 1, np.nan, 2, np.nan],
        "IV10": [1, 2, 2, 1, 2, np.nan, np.nan],
    })

@pytest.fixture(scope="module")
def expected_nbi_sanitaria_df(mock_data_for_nbi_sanitaria: pd.DataFrame) -> pd.DataFrame:
    """Expected output after applying generate_nbi_sanitaria."""
    df = mock_data_for_nbi_sanitaria.copy()
    df.loc[:, "NBI_SANITARIA"] = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]

    # Force relevant columns to float64 due to np.nan in input
    return df

@pytest.fixture(scope="module")
def mock_data_for_nbi_zona_vulnerable() -> pd.DataFrame:
    """A small DataFrame to test generate_nbi_zona_vulnerable."""
    return pd.DataFrame({
        "CODUSU": ["H1", "H2", "H3", "H4", "H5", "H6", "H7"],
        "NRO_HOGAR": [1, 1, 1, 1, 1, 1, 1],
        "IV12_1": [1, 0, 1, 0, np.nan, 1, np.nan],
        "IV12_3": [0, 1, 1, 0, 1, np.nan, np.nan],
    })

@pytest.fixture(scope="module")
def expected_nbi_zona_vulnerable_df(mock_data_for_nbi_zona_vulnerable: pd.DataFrame) -> pd.DataFrame:
    """Expected output after applying generate_nbi_zona_vulnerable."""
    df = mock_data_for_nbi_zona_vulnerable.copy()
    df.loc[:, "NBI_ZONA_VULNERABLE"] = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]

    # Force relevant columns to float64 due to np.nan in input
    df['IV12_1'] = df['IV12_1'].astype('float64')
    df['IV12_3'] = df['IV12_3'].astype('float64')

    return df

@pytest.fixture(scope="module")
def mock_data_for_nbi_dificultad_laboral() -> pd.DataFrame:
    """A small DataFrame to test generate_nbi_dificultad_laboral."""
    return pd.DataFrame({
        "CODUSU": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11", "H12", "H13"],
        "NRO_HOGAR": [1]*13,
        "CH06_jefx": [20, 20, 10, 70, 20, 20, 20, 10, 65, 20, np.nan, 20, 20],
        "CH04_jefx": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 1, np.nan, 1], # 1=Male, 2=Female
        "ESTADO_jefx": [2, 0, 2, 2, 1, 2, 0, 2, 2, 1, 2, 2, np.nan], # 2=Unemployed
        "PP02E_jefx": [0, 3, 0, 0, 1, 0, 3, 0, 0, 1, 0, 0, 0], # 3, 5 are target values
    })

@pytest.fixture(scope="module")
def expected_nbi_dificultad_laboral_df(mock_data_for_nbi_dificultad_laboral: pd.DataFrame) -> pd.DataFrame:
    """Expected output after applying generate_nbi_dificultad_laboral."""
    df = mock_data_for_nbi_dificultad_laboral.copy()
    # Manually calculate expected values based on the complex conditions
    expected_values = [
        1.0, # Male, age OK, ESTADO=2
        1.0, # Male, age OK, PP02E=3
        0.0, # Male, age too young
        0.0, # Male, age too old
        0.0, # Male, age OK, neither ESTADO nor PP02E
        1.0, # Female, age OK (16-59), ESTADO=2
        1.0, # Female, age OK, PP02E=3
        0.0, # Female, age too young
        0.0, # Female, age too old
        0.0, # Female, age OK, neither ESTADO nor PP02E
        0.0, # Male, age is NaN
        0.0, # Male, sex is NaN (condition fails)
        0.0, # Male, ESTADO is NaN (condition fails)
    ]
    df.loc[:, "NBI_DIFLABORAL"] = expected_values

    # Force relevant columns to float64 due to np.nan in input
    df['CH06_jefx'] = df['CH06_jefx'].astype('float64')
    df['CH04_jefx'] = df['CH04_jefx'].astype('float64')
    df['ESTADO_jefx'] = df['ESTADO_jefx'].astype('float64')
    return df

@pytest.fixture(scope="module")
def mock_data_for_nbi_trabajo_precario() -> pd.DataFrame:
    """A small DataFrame to test generate_nbi_trabajo_precario."""
    return pd.DataFrame({
        "CODUSU": ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10", "H11"],
        "NRO_HOGAR": [1]*11,
        "CAT_OCUP": [2, 1, 1, 1, 1, 2, 1, 2, np.nan, 2, 1],
        "NIVEL_ED": [1, 0, 0, 0, 0, 5, 0, 1, 1, np.nan, 0],
        "PP07I": [0, 2, 0, 2, 0, 0, 1, 2, 0, 0, np.nan],
        "PP07H": [0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0],
        "PP04B1": [0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0],
    })

@pytest.fixture(scope="module")
def expected_nbi_trabajo_precario_df(mock_data_for_nbi_trabajo_precario: pd.DataFrame) -> pd.DataFrame:
    """Expected output after applying generate_nbi_trabajo_precario."""
    df = mock_data_for_nbi_trabajo_precario.copy()
    expected_values = [
        1.0, # Cond A
        1.0, # Cond B (PP07I=2)
        1.0, # Cond B (PP07H=2)
        1.0, # Cond B (PP07I=2) & Cond C (PP04B1=1)
        0.0, # All False
        0.0, # Cond A false (NIVEL_ED not in list)
        0.0, # Cond B false (PP07I/H not 2) & Cond C false (PP07I/H not 2)
        1.0, # Cond A & B & C are all true for different parts
        0.0, # CAT_OCUP is NaN
        0.0, # NIVEL_ED is NaN
        0.0, # PP07I is NaN
    ]
    df.loc[:, "NBI_TRABAJO_PRECARIO"] = expected_values

    # Force relevant columns to float64 due to np.nan in input
    df['CAT_OCUP'] = df['CAT_OCUP'].astype('float64')
    df['NIVEL_ED'] = df['NIVEL_ED'].astype('float64')
    df['PP07I'] = df['PP07I'].astype('float64')

    return df

@pytest.fixture(scope="module")
def mock_data_for_nbi_cobertura_previsional() -> pd.DataFrame:
    """A small DataFrame to test generate_nbi_cobertura_previsional."""
    return pd.DataFrame({
        "CODUSU": ["H1", "H2", "H3", "H4", "H5"],
        "NRO_HOGAR": [1, 1, 1, 1, 1],
        "CH06": [68, 60, 45, np.nan, 65],
        "CH04": [1, 2, 2, np.nan, 1],
        "V2_M": [10000, 50000, 0, np.nan, 0]
    })

@pytest.fixture(scope="module")
def expected_nbi_cobertura_previsional_df(mock_data_for_nbi_cobertura_previsional: pd.DataFrame) -> pd.DataFrame:
    """Expected output after applying generate_nbi_cobertura_previsional."""
    df = mock_data_for_nbi_cobertura_previsional.copy()
    df.loc[:, "NBI_COBERTURA_PREVISIONAL"] = [0.0, 0.0, 0.0, 0.0, 1.0]

    # Force relevant columns to float64 due to np.nan in input
    df['CH06'] = df['CH06'].astype('float64')
    df['CH04'] = df['CH04'].astype('float64')
    df['V2_M'] = df['V2_M'].astype('float64')

    return df