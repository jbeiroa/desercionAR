import os
import pandas as pd
import pytest
from pipelines.etl import ETLPipeline

@pytest.fixture
def mock_get_eph(monkeypatch):
    """Mocks the get_eph function to read from test fixtures."""
    
    # Load the fixture data once
    individual_fixture = pd.read_csv("tests/fixtures/etl/individual_data.csv")
    household_fixture = pd.read_csv("tests/fixtures/etl/household_data.csv")

    def mock_eph(db_type: str, year: int, period: int):
        if db_type == "individual":
            data = individual_fixture
        elif db_type == "hogar":
            data = household_fixture
        else:
            return None
        
        # Filter data for the requested year and quarter
        return data[(data["ANO4"] == year) & (data["TRIMESTRE"] == period)].copy()

    monkeypatch.setattr("data.file_handler.get_eph", mock_eph)

def test_etl_pipeline_run(mock_get_eph, tmp_path):
    """
    Integration test for the ETL pipeline.
    
    This test runs the entire ETL pipeline with mocked data and asserts
    on the output files and their content.
    """
    # 1. Configure pipeline to use a temporary output directory
    output_dir = tmp_path / "processed"
    output_dir.mkdir()
    
    # Create a custom config for the test
    test_config = {
        'pipeline': {
            'years': [2023],
            'quarters': [1, 2, 3, 4],
            'output_dir': str(output_dir),
            'transformations': [
                {'function': 'generate_conyuge_trabaja'},
                {'function': 'generate_jefa_mujer'},
                {'function': 'generate_hogar_monop'},
                {'function': 'generate_ratio_ocupados'},
                {'function': 'generate_nbi_cobertura_previsional'},
                {'function': 'generate_nbi_dificultad_laboral'},
                {'function': 'generate_nbi_hacinamiento'},
                {'function': 'generate_nbi_sanitaria'},
                {'function': 'generate_nbi_tenencia'},
                {'function': 'generate_nbi_trabajo_precario'},
                {'function': 'generate_nbi_vivienda_precaria'},
                {'function': 'generate_nbi_zona_vulnerable'},
            ]
        }
    }
    
    # Use a temporary yaml file for config
    import yaml
    config_path = tmp_path / "test_etl_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(test_config, f)

    # 2. Run the ETL pipeline
    pipeline = ETLPipeline(config_path=str(config_path))
    pipeline.run(years=[2023], quarters=[1, 2, 3, 4], dataset_type='all')

    # 3. Assert file creation
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    predict_path = output_dir / "predict.csv"

    assert train_path.exists()
    assert test_path.exists()
    assert predict_path.exists()

    # 4. Load and assert content of the output files
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    predict_df = pd.read_csv(predict_path)

    # Assert predict.csv
    assert not predict_df.empty
    assert 'DESERTO' not in predict_df.columns
    # Student H2 (COMPONENTE 2) should be in predict
    assert 2 in predict_df['COMPONENTE'].values
    
    # Assert train.csv
    assert not train_df.empty
    assert 'DESERTO' in train_df.columns
    # Student H1 (COMPONENTE 1) who dropped out should be in train_df
    assert 1 in train_df['COMPONENTE'].values
    assert float(train_df[train_df['COMPONENTE'] == 1]['DESERTO'].iloc[0]) == 1.0
    assert 4 in train_df['COMPONENTE'].values
    # Student H2 (COMPONENTE 2, from predict set) should not be in train_df
    assert 2 not in train_df['COMPONENTE'].values

    # Assert test.csv
    assert not test_df.empty
    assert 'DESERTO' in test_df.columns
    # Student H3 (COMPONENTE 3) should be in test_df
    assert 3 in test_df['COMPONENTE'].values
    assert 2 not in test_df['COMPONENTE'].values
