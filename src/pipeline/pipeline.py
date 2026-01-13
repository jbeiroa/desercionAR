import pandas as pd
from src.data import file_handler as fh
from src.data import cleaning as l
from src.data import preprocessing as pr
from src.features import build_features as bf
from src.pipeline.config import load_config
import os


class ETLPipeline:
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
        self.data = None
        self.individual_data = None
        self.household_data = None

    def extract(self):
        """
        Extracts data from the EPH.
        """
        pipeline_config = self.config.get('pipeline', {})
        years = pipeline_config.get('years', [])
        quarters = pipeline_config.get('quarters', [])

        individual_base_dfs = [fh.get_eph(
            'individual', anio, quarter) for anio in years for quarter in quarters]
        household_base_dfs = [fh.get_eph('hogar', anio, quarter)
                       for anio in years for quarter in quarters]

        self.individual_data = [l.select_features(
            base, l.cols_individual, l.cols_id_individual) for base in individual_base_dfs]
        self.household_data = [l.select_features(
            base, l.cols_hogar, l.cols_id_hogar) for base in household_base_dfs]

        self.data = [pr.join_individuals_households(
            individuals, households) for individuals, households in zip(self.individual_data, self.household_data)]

    def transform(self):
        """
        Transforms the data using the functions specified in the config.
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
                    if func_name == 'generar_ratio_ocupados_miembros':
                        _base_df = func(_base_df, individuals, households)
                    else:
                        _base_df = func(_base_df)
                else:
                    raise ValueError(f"Transformation function '{func_name}' not found in build_features module.")
            transformed_data.append(_base_df)
        
        self.data = transformed_data

    def load(self):
        """
        Loads the transformed data to a CSV file.
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
        """
        Runs the ETL pipeline.
        """
        self.extract()
        self.transform()
        self.load()
line.
        """
        self.extract()
        self.transform()
        self.load()
