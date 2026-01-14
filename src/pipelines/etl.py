"""ETL pipeline for processing EPH (Encuesta Permanente de Hogares) data.

This module implements an ETL (Extract, Transform, Load) pipeline that downloads
Argentine household survey data, preprocesses it, applies feature engineering
transformations, and exports the final dataset. The pipeline is configurable via
YAML configuration files.
"""

import sys
from pathlib import Path

# Add repo root to path for direct execution
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import os
import pandas as pd
from src.data import file_handler as fh
from src.data import cleaning as l
from src.data import preprocessing as pr
from src.features import build_features as bf
from src.pipelines.config import load_config

class ETLPipeline:
    """ETL pipeline for EPH data processing and feature engineering.
    
    Orchestrates the extraction of raw EPH data from INDEC, preprocessing,
    feature engineering transformations, and export of the final dataset.
    Configuration is loaded from a YAML file.
    
    Attributes:
        config (dict): Pipeline configuration dictionary loaded from YAML.
        data (list): List of merged individual-household DataFrames.
        individual_data (list): List of selected individual feature DataFrames.
        household_data (list): List of selected household feature DataFrames.
    """

    def __init__(self, config_path: str):
        """Initializes the ETL pipeline.
        
        Args:
            config_path (str): Path to the YAML configuration file.
        """
        self.config = load_config(config_path)
        self.data = None
        self.individual_data = None
        self.household_data = None

    def extract(self):
        """Extracts raw EPH data from INDEC or local cache.
        
        Downloads individual and household databases for the years and quarters
        specified in the configuration. Selects relevant features and merges
        individual and household data on common identifiers.
        """
        pipeline_config = self.config.get('pipeline', {})
        years = pipeline_config.get('years', [])
        quarters = pipeline_config.get('quarters', [])

        individual_base_dfs = [fh.get_eph(
            'individual', year, quarter) for year in years for quarter in quarters]
        household_base_dfs = [fh.get_eph('hogar', year, quarter)
                       for year in years for quarter in quarters]

        self.individual_data = [l.select_features(
            base, pr.individual_features, pr.id_individual_features) for base in individual_base_dfs]
        self.household_data = [l.select_features(
            base, pr.house_features, pr.id_house_features) for base in household_base_dfs]

        self.data = [pr.join_individuals_households(
            individuals, households) for individuals, households in zip(self.individual_data, self.household_data)]

    def transform(self):
        """Transforms and enriches the data with engineered features.
        
        Filters data to include only students (CH10 == 1, age >= 14, education level <= 3),
        generates auxiliary dataframes for household heads and spouses, and applies
        a sequence of feature engineering transformations specified in the configuration.
        
        Raises:
            ValueError: If a transformation function specified in the config
                is not found in the build_features module.
        """
        pipeline_config = self.config.get('pipeline', {})
        transformations = pipeline_config.get('transformations', [])

        # Filter students
        student_condition = 'CH10 == 1'
        age_condition = 'CH06 >= 14'
        education_level_condition = 'NIVEL_ED <= 3'
        cond = student_condition + '&' + education_level_condition + '&' + age_condition
        students = [l.filter_by_columns(base, cond) for base in self.data]

        transformed_data = []
        for base, individuals, households in zip(students, self.individual_data, self.household_data):
            household_heads, spouses = bf.generate_auxiliary_dataframes(individuals, households)
            _base_df = bf.join_heads_spouses(base, household_heads, spouses)

            for transformation in transformations:
                func_name = transformation['function']
                if hasattr(bf, func_name):
                    func = getattr(bf, func_name)
                    if func_name == 'generate_ratio_ocupados':
                        _base_df = func(_base_df, individuals, households)
                    else:
                        _base_df = func(_base_df)
                else:
                    raise ValueError(f"Transformation function '{func_name}' not found in build_features module.")
            transformed_data.append(_base_df)
        
        self.data = transformed_data

    def load(self):
        """Loads the final dataset to a CSV file.
        
        Generates the dropout target variable by comparing consecutive quarters,
        filters out the last quarter (insufficient future data), and exports
        the combined dataset to a CSV file.
        """
        pipeline_config = self.config.get('pipeline', {})
        output_dir = pipeline_config.get('output_dir', 'data/processed')
        output_filename = pipeline_config.get('output_filename', 'output.csv')
        
        repo_path = fh.get_repo_path()
        output_path = os.path.join(repo_path, output_dir)
        os.makedirs(output_path, exist_ok=True)
        
        # Build the target variable
        last_quarter = pipeline_config.get('quarters', [])[-1]
        students = []
        for base, base_p1 in zip(self.data[:-1], self.individual_data[1:]):
            _base_df = bf.generate_dropout_target(base, base_p1)
            students.append(_base_df[_base_df.TRIMESTRE != last_quarter])
        
        final_data = pd.concat(students)

        final_data.to_csv(os.path.join(output_path, output_filename), index=False)

    def run(self):
        """Executes the complete ETL pipeline.
        
        Runs the extract, transform, and load steps in sequence.
        """
        self.extract()
        self.transform()
        self.load()

if __name__ == "__main__":
    import tempfile
    import yaml
    
    config_dict = {
        'pipeline': {
            'years': [2020, 2021],
            'quarters': [2, 3],
            'output_dir': 'data/processed',
            'output_filename': 'example_dropout_data.csv',
            'transformations': [
                {'function': 'generate_jefe_trabaja'},
                {'function': 'generate_conyuge_trabaja'},
                {'function': 'generate_jefa_mujer'},
                {'function': 'generate_hogar_monop'},
                {'function': 'generate_ratio_ocupados'},
                {'function': 'generate_nbi_subsistencia'},
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
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f)
        config_path = f.name
    
    try:
        pipeline = ETLPipeline(config_path)
        print("Running ETL pipeline with years [2020, 2021] and quarters [2, 3]...")
        pipeline.run()
        print("ETL pipeline completed successfully!")
    except Exception as e:
        print(f"Error running ETL pipeline: {e}")
    finally:
        import os as _os
        if _os.path.exists(config_path):
            _os.remove(config_path)