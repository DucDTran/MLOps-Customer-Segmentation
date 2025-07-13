from airflow import DAG
from airflow.providers.google.cloud.transfers.gcs_to_local import GCSToLocalFilesystemOperator
from airflow.providers.google.cloud.operators.gcs import GCSListObjectsOperator
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime
import pandas as pd
import sqlalchemy
import io

def load_to_sql(file_name:str, bucket_name:str):
    gcs_hook = GCSHook()
    conn = BaseHook.get_connection("postgres_default")
    engine = sqlalchemy.create_engine(f"postgresql+psycopg2://{conn.login}:{conn.password}@{conn.host}:{conn.port}/{conn.schema}")
   
    file_bytes = gcs_hook.download(bucket_name=bucket_name, object_name=file_name)
    with pd.read_csv(io.BytesIO(file_bytes), chunksize=10000, encoding="latin1") as reader:
        for chunk in reader:
            chunk.to_sql("ecommerce_data", engine, if_exists="append", index=False)

with DAG(
    dag_id="extract_ecommerce_data",
    schedule=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    render_template_as_native_obj=True,
    tags=["gcs", "postgres"]
) as dag:

    list_files = GCSListObjectsOperator(
        task_id="list_files_in_bucket",
        bucket="ecommerce-data-bucket-001",
    )

    load_file = PythonOperator.partial(
        task_id="load_to_sql",
        python_callable=load_to_sql,
        op_kwargs={
            "bucket_name": "ecommerce-data-bucket-001"
        }
    ).expand(op_args=list_files.output.map(lambda x: [x]))
