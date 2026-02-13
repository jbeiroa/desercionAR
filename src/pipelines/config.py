"""Configuration loading utilities for ETL pipelines."""

import yaml


def load_config(config_path: str) -> dict:
    """Loads a YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary parsed from YAML.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    # Example usage
    import os

    file_path = os.path.dirname(__file__)
    pipelines_path = os.path.dirname(file_path)
    repo_path = os.path.dirname(pipelines_path)
    config_path = os.path.join(repo_path, "configs/etl_pipeline.yaml")
    config = load_config(config_path)
    print(config)
