# Customer Segmentation MLOps

This project provides an end-to-end MLOps pipeline for customer segmentation and market basket analysis using machine learning. It covers data ingestion, processing, feature engineering, model training, and deployment, with a Streamlit web app for interactive predictions and recommendations. The project is designed for extensibility and MLOps best practices, with support for monitoring, CI/CD, and cloud deployment.

---

## Project Structure

- **dags/**: Airflow DAGs for orchestrating data extraction from Google Cloud Storage to a Postgres database.
- **pipeline/**: Python scripts for running the training and prediction pipelines.
- **src/**: Core modules for data ingestion, processing, feature engineering, and model training.
- **utils/**: Utility scripts for clustering, feature engineering, and text processing.
- **artifacts/**: Stores trained models, scalers, and processed data.
- **streamlit_app.py**: Streamlit web app for customer segmentation and recommendations.
- **docker-compose.yml**: For running Prometheus and Grafana for monitoring.
- **.gitlab-ci.yml**: GitLab CI/CD pipeline for automated build and deployment to Google Cloud Run via Artifact Registry (GCR).
- **Dockerfile**: Docker image definition for the app.
- **requirements.txt**: Python dependencies.

---

## General Workflow

1. **Data Extraction**:  
   Use the Airflow DAG (`extract_data_from_gcp.py`) to extract raw e-commerce data from Google Cloud Storage and load it into a Postgres database.

2. **Data Ingestion**:  
   The training pipeline (`pipeline/training_pipeline.py`) ingests data from the database and saves it as a CSV file.

3. **Data Processing & Feature Engineering**:  
   The pipeline cleans the data, performs clustering on products and customers, and stores features in Redis.

4. **Model Training**:  
   The pipeline trains multiple models (Random Forest, XGBoost, LightGBM) for customer segmentation, logs metrics to MLflow, and saves the best models.

5. **Prediction & Recommendation**:  
   The Streamlit app allows users to upload their own transaction data, select a customer, and receive segment predictions and product recommendations.

6. **Monitoring**:  
   Prometheus and Grafana are available for monitoring metrics from the Streamlit app.

7. **CI/CD & Cloud Deployment**:  
   GitLab CI/CD automates building the Docker image, pushing it to Google Artifact Registry (GCR), and deploying to Google Cloud Run.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd customer_segmentation_mlops
```

### 2. Install Dependencies

It is recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Set Up Environment

- Ensure you have access to a Postgres database and update `config/database_config.py` with your credentials.
- (Optional) Set up Google Cloud credentials if using the Airflow DAG for GCS extraction.
- For CI/CD and deployment, set the following environment variables in your GitLab project:
  - `GCP_SERVICE_ACCOUNT_KEY`: Base64-encoded or raw JSON key for a GCP service account with permissions for Artifact Registry and Cloud Run.

### 4. Run Data Extraction (Optional)

If you want to extract data from GCS, start Airflow and trigger the `extract_ecommerce_data` DAG:

```bash
astro dev start
# Access Airflow UI at http://localhost:8080 and trigger the DAG
```

### 5. Run the Training Pipeline

This will ingest data, process it, engineer features, and train models:

```bash
python pipeline/training_pipeline.py
```

Artifacts (models, scalers, processed data) will be saved in the `artifacts/` directory.

### 6. Run the Streamlit App

Start the web app for interactive predictions and recommendations:

```bash
streamlit run streamlit_app.py
```

- Upload your own CSV transaction data.
- Select a customer to view their segment and recommendations.

### 7. (Optional) Monitoring

To enable monitoring with Prometheus and Grafana:

```bash
docker-compose up -d
```

- Prometheus: [http://localhost:9090](http://localhost:9090)
- Grafana: [http://localhost:3000](http://localhost:3000) (default user/pass: admin/admin)

---

## CI/CD, GCR, and Cloud Deployment

### GitLab CI/CD Pipeline

The project includes a `.gitlab-ci.yml` file for automated CI/CD with the following stages:

- **Build**: Builds a Docker image for the app, authenticates with Google Cloud using a service account, and pushes the image to Google Artifact Registry (GCR).
- **Deploy**: Deploys the pushed image to Google Cloud Run, making the app available as a managed, scalable web service.

#### Key Variables
- `GCP_SERVICE_ACCOUNT_KEY`: Service account key for GCP authentication (set as a GitLab CI/CD variable).
- `GCP_REGION`, `APP_NAME`, `REPO_NAME`, `IMAGE_NAME`, `DOCKER_VERSION`: Used for configuring the build and deployment process.

#### Pipeline Overview
1. **Authenticate** to Google Cloud using the service account key.
2. **Configure Docker** to use GCP credentials for Artifact Registry.
3. **Build** the Docker image locally on the CI runner.
4. **Push** the image to Artifact Registry (GCR).
5. **Deploy** the image to Google Cloud Run using `gcloud run deploy`.

See `.gitlab-ci.yml` for full details and customization.

### Google Artifact Registry (GCR)
- The Docker image is stored in Artifact Registry (formerly GCR) under your GCP project.
- The image is referenced in the deploy step and used by Cloud Run.

### Google Cloud Run
- The app is deployed as a fully managed service on Cloud Run.
- Deployment is automated via the GitLab pipeline.
- You can access the deployed app via the URL provided by Cloud Run after deployment.

---

## Customization

- **Data Source**: Modify the Airflow DAG or `src/data_ingestion.py` to connect to your own data sources.
- **Modeling**: Update `src/model_training.py` to experiment with different models or hyperparameters.
- **Feature Engineering**: Extend `utils/feature_engineering.py` and related scripts for new features.
- **CI/CD**: Adjust `.gitlab-ci.yml` for your own GCP project, region, or deployment preferences.

---

## Requirements

See `requirements.txt` for all Python dependencies.

---

## License

MIT License

---

## Contact

For questions or support, please contact [Duc Tran](mailto:your-email@example.com).
