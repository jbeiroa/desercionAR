FROM python:3.12.4

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy only files needed for dependency installation
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root --no-dev

# Copy the application code. Note: the `api` service in docker-compose.yml has its context set to the project root,
# so paths like `src/api/` are relative to the build context.
COPY src/api/ ./src/api/
COPY src/data/ ./src/data/
COPY src/features/ ./src/features/
COPY src/models/ ./src/models/
COPY src/pipelines/ ./src/pipelines/
COPY configs/ ./configs/
COPY tests/ ./tests/ # If needed for testing within the container, otherwise omit
COPY data/processed/ ./data/processed/ # Copy processed data if it's not pulled via DVC inside the container

# Expose the port and run the app
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
