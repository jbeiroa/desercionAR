"""
File handler module for downloading EPH (Encuesta Permanente de Hogares) data.

This module provides utilities to fetch Argentine household and individual survey data
from INDEC (Instituto Nacional de Estadística y Censos) or from the pyeph library cache.
It handles downloading zip files, extracting the appropriate database files, and parsing
them into pandas DataFrames.

Main functions:
    - eph_from_indec: Downloads EPH data directly from INDEC servers
    - get_eph: Fetches EPH data with fallback from local cache to INDEC
    - get_repo_path: Returns the root repository path
"""

import os
from io import BytesIO
import zipfile

import requests
import pandas as pd
import pyeph

BASE_INDEC_URL = "https://www.indec.gob.ar/ftp/cuadros/menusuperior/eph/"


def eph_from_indec(
    db_type: str, year: int, period: int, base_url=BASE_INDEC_URL
) -> pd.DataFrame | None:
    """Downloads an EPH database from a given year, type (individuals or
    household) and period (quarter).

    Args:
        db_type (str): db type, either 'individual' or 'hogar'
        year (int): calendar year, YYYY
        period (int): quarter, 1 to 4
        base_url (str, optional): INDEC url. Defaults to BASE_INDEC_URL.

    Returns:
        pd.DataFrame | None: a pandas DataFrame with the requested database, or None if not found
    """
    url = f"{base_url}EPH_usu_{period}_Trim_{year}_txt.zip"
    response = requests.get(url, timeout=100)
    if response.status_code != 200:
        print("File not found on INDEC servers.")
        return None
    try:
        with zipfile.ZipFile(BytesIO(response.content)) as zip_file:
            file_names = zip_file.namelist()
            matching_files = [name for name in file_names if f"{db_type}" in name]
            if not matching_files:
                print(f"No file matching type '{db_type}' found in zip")
                return None
            file_name = matching_files[0]
            with zip_file.open(file_name) as file:
                df = pd.read_csv(file, sep=";", header=0)
        return df
    except (zipfile.BadZipFile, pd.errors.ParserError) as e:
        print(f"Error reading database: {e}")
        return None


def get_eph(db_type: str, year: int, period: int) -> pd.DataFrame | None:
    """Fetches an EPH database using pyeph, or downloads it from INDEC
    if not found.

    Args:
        db_type (str): db type, either 'individual' or 'hogar'
        year (int): calendar year, YYYY
        period (int): quarter, 1 to 4

    Returns:
        pd.DataFrame | None: a pandas DataFrame with the requested database, or None if not found
    """
    try:
        df = pyeph.get(data="eph", year=year, period=period, base_type=db_type)
        return df
    except pyeph.errors.NonExistentDBError:  # type: ignore
        try:
            df = eph_from_indec(db_type, year, period)
            return df
        except zipfile.BadZipFile:
            print("No existe la base solicitada.")
            return None


def get_repo_path() -> str:
    """Returns the path of the root repository.

    Returns:
        str: the path of the root repository
    """
    file_path = os.path.dirname(__file__)
    src_path = os.path.dirname(file_path)
    repo_path = os.path.dirname(src_path)
    return repo_path


if __name__ == "__main__":
    # Example usage
    data = get_eph("individual", 2020, 1)
    if data is not None:
        print(data.head())
