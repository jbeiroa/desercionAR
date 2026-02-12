"""
Command Line Interface for the ETL pipeline.
"""

import typer
from pipelines.etl import ETLPipeline

app = typer.Typer()


@app.command()
def etl(
    config_path: str = typer.Option(
        "configs/etl_pipeline.yaml", help="Path to the ETL configuration YAML file."
    ),
):
    """
    Run the ETL pipeline with a specified configuration file.
    """
    typer.echo(f"Running ETL pipeline with config: {config_path}")
    try:
        pipeline = ETLPipeline(config_path=config_path)
        pipeline.run()  # No arguments needed
        typer.echo("ETL pipeline completed successfully!")
    except Exception as e:
        typer.echo(f"Error running ETL pipeline: {e}", err=True)
        raise typer.Exit(code=1)



if __name__ == "__main__":
    app()
