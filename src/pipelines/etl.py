"""ETL pipeline for processing EPH (Encuesta Permanente de Hogares) data.

This module implements an ETL (Extract, Transform, Load) pipeline that downloads
Argentine household survey data, preprocesses it, applies feature engineering
transformations, and exports the final dataset. The pipeline is configurable via
YAML configuration files.
"""
import os
import numpy as np
import pandas as pd
from ..data import file_handler as fh
from ..data import cleaning as l
from ..data import preprocessing as pr
from ..features import build_features as bf
from .config import load_config


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
        self.data: list[pd.DataFrame] = []
        self.individual_data: list[pd.DataFrame] | None = None
        self.household_data: list[pd.DataFrame] | None= None

    def _get_raw_data(
        self, years: list[int], quarters: list[int]
    ) -> tuple[list[pd.DataFrame | None], list[pd.DataFrame | None]]:
        """Fetches raw EPH data for specified years and quarters."""
        individual_base_dfs = [
            fh.get_eph("individual", year, quarter)
            for year in years
            for quarter in quarters
        ]
        household_base_dfs = [
            fh.get_eph("hogar", year, quarter) for year in years for quarter in quarters
        ]
        return individual_base_dfs, household_base_dfs

    def _select_and_join_features(
        self,
        individual_base_dfs: list[pd.DataFrame | None],
        household_base_dfs: list[pd.DataFrame | None],
    ) -> list[pd.DataFrame]:
        """Selects features and joins individual and household data."""
        self.individual_data = [
            l.select_features(base, pr.individual_features, pr.id_individual_features)
            for base in individual_base_dfs
        ]
        self.household_data = [
            l.select_features(base, pr.house_features, pr.id_house_features)
            for base in household_base_dfs
        ]

        return [
            pr.join_individuals_households(individuals, households)
            for individuals, households in zip(
                self.individual_data, self.household_data
            )
        ]

    def extract(self):
        """Extracts raw EPH data from INDEC or local cache.

        Downloads individual and household databases for the years and quarters
        specified in the configuration. Selects relevant features and merges
        individual and household data on common identifiers.
        """
        pipeline_config = self.config.get("pipeline", {})
        years = pipeline_config.get("years", [])
        quarters = pipeline_config.get("quarters", [])

        individual_base_dfs, household_base_dfs = self._get_raw_data(years, quarters)
        self.data = self._select_and_join_features(
            individual_base_dfs, household_base_dfs
        )

    def _apply_transformations(
        self, base_df: pd.DataFrame, individuals: pd.DataFrame, households: pd.DataFrame
    ) -> pd.DataFrame:
        """Applies a series of feature engineering transformations."""
        pipeline_config = self.config.get("pipeline", {})
        transformations = pipeline_config.get("transformations", [])
        household_heads, spouses = bf.generate_auxiliary_dataframes(
            individuals, households
        )
        _base_df = bf.join_heads_spouses(base_df, household_heads, spouses)

        for transformation in transformations:
            func_name = transformation["function"]
            if hasattr(bf, func_name):
                func = getattr(bf, func_name)
                if func_name == "generate_ratio_ocupados":
                    _base_df = func(_base_df, individuals, households)
                else:
                    _base_df = func(_base_df)
            else:
                raise ValueError(
                    f"Transformation function '{func_name}' not found in build_features module."
                )
        return _base_df

    def transform(self):
        """Transforms and enriches the data with engineered features.

        Filters data to include only students (CH10 == 1, age >= 14, education level <= 3),
        generates auxiliary dataframes for household heads and spouses, and applies
        a sequence of feature engineering transformations specified in the configuration.

        Raises:
            ValueError: If a transformation function specified in the config
                is not found in the build_features module.
        """
        # Filter students
        student_condition = "CH10 == 1"
        age_condition = "CH06 >= 14"
        education_level_condition = "NIVEL_ED <= 3"
        cond = student_condition + "&" + education_level_condition + "&" + age_condition
        students = [l.filter_by_columns(base, cond) for base in self.data]

        transformed_data = [
            self._apply_transformations(base, individuals, households)
            for base, individuals, households in zip(
                students, self.individual_data, self.household_data
            )
        ]

        self.data = transformed_data
    def _generate_dropout_target(
        self, data_t: pd.DataFrame, individuals_tp1: pd.DataFrame
    ) -> pd.DataFrame:
        cols_tp1 = ["CODUSU", "NRO_HOGAR", "COMPONENTE", "CH10"]
        data = pd.merge(
            data_t,
            individuals_tp1[cols_tp1],
            on=["CODUSU", "NRO_HOGAR", "COMPONENTE"],
            suffixes=("", "_fin"),
            how="inner",
        )
        cond = (data.CH10 == 1) & (data.CH10_fin == 2)
        students = pr.create_binary_feature(data, "DESERTO", cond)
        students.drop(["CH10_fin"], axis=1, inplace=True)
        return students

    def _homogenize_binary_columns(
        self, df: pd.DataFrame, columns: list[str]
    ) -> pd.DataFrame:
        replace_dict = {col: {2: 0, "S": 1, "SI": 1, "N": 0, "NO": 0} for col in columns}
        data = df.replace(replace_dict)
        data[columns] = data[columns].astype("float64", copy=False)
        return data

    def _distance_to_capital(
        self, df: pd.DataFrame, agglomeration="AGLOMERADO"
    ) -> pd.DataFrame:
        agglomeration_coords = {
            "eph_codagl": [
                13, 29, 31, 25, 34, 7, 26, 15, 4, 91, 18, 23, 30, 12, 20, 93, 8,
                14, 6, 5, 3, 9, 22, 36, 38, 38, 10, 19, 2, 32, 17, 33, 27],
            "eph_aglome": [
                "Gran Córdoba", "Gran Tucumán - Tafi Viejo", "Ushuaia - Rio Grande",
                "La Rioja", "Mar del Plata - Batán", "Posadas", "San Luis - El Chorrillo",
                "Formosa", "Gran Rosario", "Rawson - Trelew", "Santiago del Estero - La Banda",
                "Salta", "Santa Rosa - Toay", "Corrientes", "Rio Gallegos",
                "Viedma - Carmen de Patagones", "Gran Resistencia", "Concordia",
                "Gran Paraná", "Gran Santa Fe", "Bahia Blanca - Cerri",
                "Comodoro Rivadavia - Rada Tilly", "Gran Catamarca", "Rio Cuarto",
                "San Nicolas - Villa Constitiución", "San Nicolas - Villa Constitiución",
                "Gran Mendoza", "Jujuy - Palpalá", "Gran La Plata", "CABA",
                "Neuquén - Plottier", "Partidos del GBA", "Gran San Juan"
            ],
            "x": [
                3.668196e06, 3.575528e06, 3.368650e06, 3.433735e06, 4.238702e06,
                4.500828e06, 3.470658e06, 4.290676e06, 3.989413e06, 3.561793e06,
                3.671129e06, 3.558820e06, 3.652175e06, 4.215852e06, 3.274586e06,
                3.754150e06, 4.194276e06, 4.258864e06, 4.023207e06, 4.008905e06,
                3.824880e06, 3.382300e06, 3.520630e06, 3.656809e06, 4.035477e06,
                4.029783e06, 3.231898e06, 3.560475e06, 4.234043e06, 4.193488e06,
                3.310530e06, 4.180647e06, 3.260047e06
            ],
            "y": [
                6.533650e06, 7.036009e06, 3.980855e06, 6.726538e06, 5.760505e06,
                6.922012e06, 6.318389e06, 7.090830e06, 6.346344e06, 5.211965e06,
                6.926946e06, 7.258796e06, 5.947417e06, 6.941437e06, 4.274342e06,
                5.478764e06, 6.944114e06, 6.501289e06, 6.473109e06, 6.487715e06,
                5.709612e06, 4.921987e06, 6.858955e06, 6.336869e06, 6.293741e06,
                6.307894e06, 6.360832e06, 7.329096e06, 6.103368e06, 6.144082e06,
                5.686581e06, 6.148549e06, 6.509854e06
            ],
        }
        agglomeration_coords_df = pd.DataFrame(agglomeration_coords)
        agglomeration_df = df.copy()
        agglomeration_df["x_temp"] = None
        agglomeration_df["y_temp"] = None
        agglomeration_df["distance"] = None

        capital_x = 4.193488e06
        capital_y = 6.144082e06
        for index, row in agglomeration_df.iterrows():
            eph_codagl = row[agglomeration]
            matching_row = agglomeration_coords_df[
                agglomeration_coords_df["eph_codagl"] == eph_codagl
            ]
            if not matching_row.empty:
                x_temp = matching_row["x"].values[0]
                y_temp = matching_row["y"].values[0]
                distance = np.sqrt(
                    (x_temp - capital_x) ** 2 + (y_temp - capital_y) ** 2
                )
                agglomeration_df.at[index, "distance"] = distance

        agglomeration_df["AGLOMERADO"] = agglomeration_df["distance"]
        agglomeration_df.drop(columns=["x_temp", "y_temp", "distance"], inplace=True)
        return agglomeration_df

    def _create_datetime_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Creates a datetime column from ANO4 and TRIMESTRE."""
        df["period"] = pd.to_datetime(
            df["ANO4"].astype(str) + "Q" + df["TRIMESTRE"].astype(str)
        ).dt.to_period("Q")
        return df

    def _postprocess_data(
        self, data: list[pd.DataFrame], train_test: bool = True
    ) -> pd.DataFrame:
        if len(data) > 1:
            data_concat = pd.concat(data)
        elif len(data) == 1:
            data_concat = data[0]
        else:
            return pd.DataFrame()

        drop_cols = [
            "IV8", "IX_MAYEQ10", "CAT_OCUP", "CAT_INAC", "T_VI", "V2_M",
            "IV10", "II7", "IV12_1", "IV12_3", "PP07I", "CH04_jefx",
            "CAT_OCUP_jefx", "PP02E_jefx", "PP07H", "H15",
            "ITF", "REALIZADA", "REALIZADA_jefx", 'REALIZADA_conyuge'
        ]
        binary_columns = [
            "CH11", "PP02H", "PP04B1", "IV5", "IV12_2", "II3", "II4_1",
            "II4_2", "II4_3", "V1", "V2", "V21", "V22", "V3", "V5", "V6",
            "V7", "V8", "V11", "V12", "V13", "V14", "PP07I_jefx",
            "PP07H_jefx", "PP04B1_jefx", "CONYUGE_TRABAJA", "JEFA_MUJER",
            "HOGAR_MONOP", "NBI_COBERTURA_PREVISIONAL", "NBI_DIFLABORAL",
            "NBI_HACINAMIENTO", "NBI_SANITARIA", "NBI_TENENCIA",
            "NBI_TRABAJO_PRECARIO", "NBI_VIVIENDA", "NBI_ZONA_VULNERABLE",
            "MAS_500", "CH04",
        ]
        if train_test:
            binary_columns.append("DESERTO")

        data_concat = data_concat[data_concat["H15"] == 1].copy()
        data_concat.drop(drop_cols, axis=1, inplace=True)
        data_concat = self._homogenize_binary_columns(data_concat, binary_columns)

        data_concat.loc[:, "PP04B1"].replace({2: 0, np.nan: 0}, inplace=True)
        data_concat.rename({"PP04B1": "servicio_domestico"}, axis=1, inplace=True)
        data_concat.rename(
            {"PP07H_jefx": "APORTES_JUBILATORIOS_jefx"}, axis=1, inplace=True
        )

        cvars = data_concat.columns.str.endswith("_conyuge")
        data_concat.loc[:, cvars] = data_concat.loc[:, cvars].fillna(0)

        data_concat = self._distance_to_capital(data_concat)
        return data_concat

    def _remove_duplicates(self, df_list: list[pd.DataFrame]) -> pd.DataFrame:
        if len(df_list) < 2:
            raise ValueError("df_list must be a list of at least two DataFrames.")

        result_df = df_list[0]
        for df in df_list[1:]:
            result_df = pd.merge(
                result_df,
                df[["CODUSU", "NRO_HOGAR", "COMPONENTE"]],
                on=["CODUSU", "NRO_HOGAR", "COMPONENTE"],
                how="left",
                indicator=True,
            )
            result_df = result_df[result_df["_merge"] == "left_only"]
            result_df = result_df.drop("_merge", axis=1)
        return result_df

    def load(self):
        """
        Generates the dropout target variable, preprocesses the data, splits it into
        train, test, and predict sets, and saves them to CSV files.
        This method correctly handles the panel data nature of the EPH by ensuring
        that individuals in the prediction set do not appear in the train or test sets.
        """
        pipeline_config = self.config.get("pipeline", {})
        output_dir = pipeline_config.get("output_dir", "data/processed")
        repo_path = fh.get_repo_path()
        output_path = os.path.join(repo_path, output_dir)
        os.makedirs(output_path, exist_ok=True)

        train_path = os.path.join(output_path, "train.csv")
        test_path = os.path.join(output_path, "test.csv")
        predict_path = os.path.join(output_path, "predict.csv")

        # 1. Isolate data for prediction (last quarter) from training/testing data
        predict_source_df = self.data[-1]
        training_sources = self.data[:-1]
        individual_training_sources = self.individual_data[1:]

        # 2. Generate target variable only for the training/testing data
        labeled_data_list = [
            self._generate_dropout_target(base, base_p1)
            for base, base_p1 in zip(training_sources, individual_training_sources)
        ]

        # 3. Post-process both labeled and prediction dataframes
        labeled_df = self._postprocess_data(labeled_data_list, train_test=True)
        predict_df = self._postprocess_data([predict_source_df], train_test=False)

        # 4. Remove individuals from labeled_df that are in the prediction set
        labeled_df = self._remove_duplicates([labeled_df, predict_df])

        # 5. Create a time-series column to split chronologically
        labeled_df = self._create_datetime_column(labeled_df)

        # 6. Split labeled data into train and test sets based on time
        if not labeled_df.empty:
            test_period = labeled_df["period"].max()
            test_df = labeled_df[labeled_df["period"] == test_period].copy()
            train_df = labeled_df[labeled_df["period"] < test_period].copy()
        else:
            test_df = pd.DataFrame(columns=labeled_df.columns)
            train_df = pd.DataFrame(columns=labeled_df.columns)

        # 7. Drop the temporary period column
        for df in [train_df, test_df, predict_df]:
            if "period" in df.columns:
                df.drop(columns=["period"], inplace=True)

        # 8. Save the final datasets
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        predict_df.to_csv(predict_path, index=False)

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
        "pipeline": {
            "years": [2021, 2022],
            "quarters": [2, 3, 4],
            "output_dir": "data/processed/v0.2.0/",
            "transformations": [
                {"function": "generate_conyuge_trabaja"},
                {"function": "generate_jefa_mujer"},
                {"function": "generate_hogar_monop"},
                {"function": "generate_ratio_ocupados"},
                {"function": "generate_nbi_cobertura_previsional"},
                {"function": "generate_nbi_dificultad_laboral"},
                {"function": "generate_nbi_hacinamiento"},
                {"function": "generate_nbi_sanitaria"},
                {"function": "generate_nbi_tenencia"},
                {"function": "generate_nbi_trabajo_precario"},
                {"function": "generate_nbi_vivienda_precaria"},
                {"function": "generate_nbi_zona_vulnerable"},
            ],
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        temp_config_path = f.name

    try:
        pipeline = ETLPipeline(temp_config_path)
        print(
            "Running ETL pipeline with years [2020, 2021] and quarters [1, 2, 3, 4]..."
        )
        pipeline.run()
        print("ETL pipeline completed successfully!")
    except Exception as e:
        print(f"Error running ETL pipeline: {e}")
    finally:
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
