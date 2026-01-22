"""
Command Line Interface for the ETL pipeline.
"""

import typer
from enum import Enum
from typing import List, Optional
from src.pipelines.etl import ETLPipeline

app = typer.Typer()


class DatasetType(str, Enum):
    all = "all"
    train_test = "train-test"
    predict = "predict"


@app.command()
def etl(
    years: Optional[List[int]] = typer.Option(
        [2021, 2022], help="List of years to process."
    ),
    quarters: Optional[List[int]] = typer.Option(
        [2, 3, 4], help="List of quarters to process."
    ),
    dataset_type: DatasetType = typer.Option(
        DatasetType.all, help="Type of dataset to create."
    ),
):
    """
    Run the ETL pipeline with specified years, quarters, and dataset type.
    """
    typer.echo(f"Running ETL pipeline for years: {years}, quarters: {quarters}")
    typer.echo(f"Dataset type: {dataset_type.value}")

    try:
        pipeline = ETLPipeline()
        pipeline.run(
            years=years,
            quarters=quarters,
            dataset_type=dataset_type.value,
        )
        typer.echo("ETL pipeline completed successfully!")
    except Exception as e:
        typer.echo(f"Error running ETL pipeline: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
