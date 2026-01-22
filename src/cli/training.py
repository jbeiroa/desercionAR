"""
Command Line Interface for the Training pipeline.
"""

import typer
import yaml
import os
from typing import Optional
from src.pipelines.training import TrainingPipeline

app = typer.Typer()


@app.command()
def training(
    config: Optional[str] = typer.Option(
        "configs/training.yaml", help="Path to the training configuration YAML file."
    ),
):
    """
    Run the Training pipeline with a specified configuration file.
    """
    typer.echo(f"Running Training pipeline with config: {config}")

    # Load configuration from YAML file
    try:
        config_path = os.path.expanduser(config)
        with open(config_path, "r") as f:
            pipeline_config = yaml.safe_load(f)
    except FileNotFoundError:
        typer.echo(f"Error: Configuration file not found at {config_path}", err=True)
        raise typer.Exit(code=1)
    except yaml.YAMLError as e:
        typer.echo(f"Error loading YAML config: {e}", err=True)
        raise typer.Exit(code=1)

    try:
        pipeline = TrainingPipeline(config=pipeline_config)
        pipeline.run()
        typer.echo("Training pipeline completed successfully!")
    except Exception as e:
        typer.echo(f"Error running Training pipeline: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
