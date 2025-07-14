# Customer Segmentation MLOps

This project provides an end-to-end pipeline for customer segmentation and market basket analysis using machine learning. It includes data ingestion, processing, feature engineering, model training, and a Streamlit web app for interactive predictions and recommendations. The project is designed for extensibility and MLOps best practices, with support for monitoring and deployment.

---

## Project Structure

- **dags/**: Airflow DAGs for orchestrating data extraction from Google Cloud Storage to a Postgres database.
- **pipeline/**: Python scripts for running the training and prediction pipelines.
- **src/**: Core modules for data ingestion, processing, feature engineering, and model training.
- **utils/**: Utility scripts for clustering, feature engineering, and text processing.
- **artifacts/**: Stores trained models, scalers, and processed data.
- **streamlit_app.py**: Streamlit web app for customer segmentation and recommendations.
- **docker-compose.yml**: For running Prometheus and Grafana for monitoring.
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

## Customization

- **Data Source**: Modify the Airflow DAG or `src/data_ingestion.py` to connect to your own data sources.
- **Modeling**: Update `src/model_training.py` to experiment with different models or hyperparameters.
- **Feature Engineering**: Extend `utils/feature_engineering.py` and related scripts for new features.

---

## Requirements

See `requirements.txt` for all Python dependencies.

---

## License

MIT License

---

## Contact

For questions or support, please contact [Duc Tran](mailto:your-email@example.com).
