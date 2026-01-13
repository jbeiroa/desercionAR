import yaml

def load_config(config_path: str) -> dict:
    """Loads a YAML configuration file.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: The configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
