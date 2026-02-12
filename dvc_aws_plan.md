Excellent idea. Integrating DVC and setting up a proper deployment on AWS are great next steps for this project. Here is a comprehensive plan designed for manual implementation, focusing on cost optimization and learning.

---

## Plan: DVC Integration and AWS Deployment

This plan is divided into two main parts. First, we'll integrate DVC to manage your data. Second, we'll set up the AWS infrastructure to host your MLflow server, data, and applications.

### **Part 1: DVC (Data Version Control) Integration**

DVC allows you to version your data and models, keeping your Git repository lightweight.

**Step 1.1: Install DVC**

If you haven't already, add DVC to your poetry project:

```bash
poetry add dvc[s3]
```
*(The `[s3]` part installs the necessary libraries to communicate with AWS S3)*

**Step 1.2: Initialize DVC**

In your project root, initialize DVC. This will create a `.dvc` directory.

```bash
dvc init
```
Commit this change to Git:
```bash
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

**Step 1.3: Configure DVC Remote Storage**

We'll use an S3 bucket for remote storage.

1.  **Create an S3 Bucket**: Go to the AWS S3 console and create a new bucket. Choose a unique name (e.g., `desercionar-dvc-storage`). It's recommended to create it in a region close to you.
2.  **Configure DVC**: Tell DVC to use this bucket as its remote storage.

    ```bash
    dvc remote add -d myremote s3://desercionar-dvc-storage
    ```
    *(This command sets `myremote` as the default remote)*

3.  **Commit the configuration**:
    ```bash
    git add .dvc/config
    git commit -m "Configure DVC remote storage"
    ```

**Step 1.4: Start Tracking Data**

Let's track the processed data directory.

```bash
dvc add data/processed
```
This command creates a `data/processed.dvc` file, which is a small text file containing a hash of the data. This is what you'll commit to Git instead of the data itself.

**Step 1.5: Add Data to `.gitignore`**

To prevent the actual data from being committed to Git, DVC automatically adds it to `.gitignore`. Verify that `data/processed` is in your `.gitignore` file.

**Step 1.6: Push Data to Remote Storage**

Now, push the data to your S3 bucket:

```bash
dvc push
```
This will upload the contents of `data/processed` to your `desercionar-dvc-storage` S3 bucket.

**Step 1.7: Commit to Git**

Finally, commit the `.dvc` file to Git:
```bash
git add data/processed.dvc .gitignore
git commit -m "Track processed data with DVC"
```
Your data is now version-controlled! If you or a collaborator needs to retrieve this version of the data, they can simply run `dvc pull`.

---

### **Part 2: AWS Deployment (Cost-Optimized)**

Here's how to deploy the different parts of your project to AWS.

**Step 2.1: Set up the MLflow Server on EC2**

For cost-effectiveness, we'll use a small EC2 instance (within the free tier if possible) to run MLflow using Docker Compose.

1.  **Create an S3 Bucket for MLflow Artifacts**: Create a new S3 bucket (e.g., `desercionar-mlflow-artifacts`) to store your model artifacts.

2.  **Create a `docker-compose.yml` for MLflow**: In the root of your project, create a file named `docker-compose.yml`:

    ```yaml
    version: '3.8'
    services:
      mlflow:
        image: ghcr.io/mlflow/mlflow:v2.11.1
        ports:
          - "5000:5000"
        environment:
          - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
          - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
        command: >
          mlflow server
          --host 0.0.0.0
          --port 5000
          --backend-store-uri sqlite:///mlflow.db
          --default-artifact-root s3://desercionar-mlflow-artifacts/
    ```
    *(Note: This uses SQLite for the backend store, which is simple and has no extra cost. The artifacts are stored in S3.)*

3.  **Launch an EC2 Instance**:
    *   Go to the EC2 console and launch a new instance.
    *   **AMI**: Choose an Amazon Linux 2 or Ubuntu AMI.
    *   **Instance Type**: Select `t2.micro` or `t3.micro` (these are usually in the free tier).
    *   **Security Group**: Create a new security group and add an inbound rule to allow TCP traffic on port 5000 from your IP address (for security). You'll also need to allow SSH (port 22).
    *   Launch the instance and connect to it via SSH.

4.  **Set up the EC2 Instance**:
    *   Install Docker and Docker Compose.
    *   Create a directory for your MLflow setup.
    *   Copy the `docker-compose.yml` file to this directory.
    *   **Important**: Set your AWS credentials as environment variables (`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`) or, even better, attach an IAM role to the EC2 instance with permissions to read/write to the `desercionar-mlflow-artifacts` S3 bucket.

5.  **Run MLflow**:
    ```bash
    docker-compose up -d
    ```
    Your MLflow server is now running! You can access it at `http://<your-ec2-public-ip>:5000`.

**Step 2.2: Deploy the FastAPI App and Dashboard on EC2**

To minimize costs and leverage the existing EC2 instance running MLflow, we will deploy both the FastAPI API and the Dash dashboard on the same instance using Docker Compose.

1.  **Create a `Dockerfile` for the API**: In your `src/api/` directory, create a `Dockerfile`. This Dockerfile prepares the environment and runs the FastAPI application.

    ```dockerfile
    FROM python:3.11

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
    ```
    *(Note: The `COPY` commands for source directories are relative to the docker-compose `context` which will be the project root. The `CMD` also needs to reflect the path within the container.)*

2.  **Create a `Dockerfile` for the Dashboard**: In your `src/dashboard/` directory, create a `Dockerfile`. This Dockerfile prepares the environment and runs the Plotly Dash application.

    ```dockerfile
    FROM python:3.11

    WORKDIR /app

    # Install poetry
    RUN pip install poetry

    # Copy only files needed for dependency installation
    COPY pyproject.toml poetry.lock ./

    # Install dependencies
    RUN poetry config virtualenvs.create false && poetry install --no-root --no-dev

    # Copy the application code.
    COPY src/dashboard/ ./src/dashboard/
    COPY src/data/ ./src/data/
    COPY src/features/ ./src/features/
    COPY src/models/ ./src/models/
    COPY src/pipelines/ ./src/pipelines/
    COPY configs/ ./configs/
    COPY tests/ ./tests/ # If needed for testing within the container, otherwise omit
    COPY data/processed/ ./data/processed/ # Copy processed data if it's not pulled via DVC inside the container


    # Expose the port and run the app
    EXPOSE 8050
    CMD ["python", "src/dashboard/app.py"]
    ```
    *(Note: Similar to the API, `COPY` commands are relative to the build context, and `CMD` reflects the path within the container.)*

3.  **Update `docker-compose.yml`**: Modify the `docker-compose.yml` file in your project root to include services for both the FastAPI API and the Dash Dashboard, alongside the MLflow server.

    ```yaml
    version: '3.8'
    services:
      mlflow:
        image: ghcr.io/mlflow/mlflow:v2.11.1
        ports:
          - "5000:5000"
        environment:
          - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} # Only if not using IAM role
          - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} # Only if not using IAM role
          - MLFLOW_SERVER_ALLOWED_HOSTS=* # Required for some MLflow versions
        command: >
          mlflow server
          --host 0.0.0.0
          --port 5000
          --backend-store-uri sqlite:///mlflow.db
          --default-artifact-root s3://desercionar-mlflow-artifacts/
      api:
        build:
          context: . # Build context is the project root
          dockerfile: src/api/Dockerfile # Path to the API's Dockerfile
        ports:
          - "8000:8000"
        environment:
          - MLFLOW_TRACKING_URI=http://mlflow:5000 # Point to the MLflow service within the Docker network
          - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} # Only if API needs direct S3 access
          - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} # Only if API needs direct S3 access
          - PYTHONPATH=/app # Ensure Python can find src modules
        depends_on:
          - mlflow
      dashboard:
        build:
          context: . # Build context is the project root
          dockerfile: src/dashboard/Dockerfile # Path to the Dashboard's Dockerfile
        ports:
          - "8050:8050"
        environment:
          - MLFLOW_TRACKING_URI=http://mlflow:5000 # Point to the MLflow service within the Docker network
          - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} # Only if Dashboard needs direct S3 access
          - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} # Only if Dashboard needs direct S3 access
          - PYTHONPATH=/app # Ensure Python can find src modules
        depends_on:
          - mlflow
          - api # Dashboard might depend on API being up, if it calls it directly
    ```
    *(Note: Ensure your EC2 instance's IAM role has permissions to access the MLflow artifact S3 bucket and any other S3 buckets the API or Dashboard might need.)*

4.  **Open Security Group Ports on EC2**:
    *   Go to the AWS EC2 console, select your MLflow EC2 instance, and modify its associated Security Group.
    *   Add inbound rules to allow TCP traffic on port `8000` (for FastAPI) and port `8050` (for Dash Dashboard) from your IP address or a wider range if accessible publicly. (Remember to also have SSH (port 22) and MLflow (port 5000) open).

5.  **Copy Project to EC2, Build, and Run**:
    *   **SSH into your EC2 instance**.
    *   **Clone your Git repository** onto the EC2 instance, or `rsync` your project files including `pyproject.toml`, `poetry.lock`, `src/`, `configs/`, `data/processed/` (if not using DVC pull inside container), and the updated `docker-compose.yml`.
    *   **Navigate to your project root directory** on the EC2 instance.
    *   **If you are using DVC**, ensure you have DVC installed on the EC2 instance and run `dvc pull` to retrieve the `data/processed` directory.
    *   **Build the Docker images**:
        ```bash
        docker-compose build
        ```
    *   **Start all services**:
        ```bash
        docker-compose up -d
        ```
        Your MLflow server, FastAPI API, and Dash Dashboard should now be running on the same EC2 instance! You can access them at:
        *   MLflow UI: `http://<your-ec2-public-ip>:5000`
        *   FastAPI: `http://<your-ec2-public-ip>:8000`
        *   Dashboard: `http://<your-ec2-public-ip>:8050`

---
This plan provides a solid, cost-effective foundation. Once you're comfortable with this setup, you can explore more advanced options like using AWS RDS for the MLflow backend, setting up a CI/CD pipeline with GitHub Actions to automate deployments, and using a custom domain for your services.

Let me know if you have any questions before you begin!
