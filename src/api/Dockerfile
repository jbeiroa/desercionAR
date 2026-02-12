FROM python:3.12.4

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy only files needed for dependency installation
COPY ../../poetry.lock ../../pyproject.toml ./

# Install dependencies
RUN poetry config virtualenvs.create false && poetry install --no-root

# Copy the application code
COPY . .

# Expose the port and run the app
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]