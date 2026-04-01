from datetime import datetime
import json
import os

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.filesystem import FileSensor

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV = os.path.join(PROJECT_DIR, ".venv", "Scripts", "python")

default_args = {
    "owner": "airflow",
    "start_date": datetime(2026, 1, 1),
    "retries": 1,
}

with DAG(
    dag_id="ml_training_pipeline",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    description="ML Training Pipeline: prepare -> train -> evaluate -> register",
    tags=["mlops", "lab5"],
) as dag:

    check_data = FileSensor(
        task_id="check_data_available",
        filepath=os.path.join(PROJECT_DIR, "data", "raw", "heart.csv"),
        timeout=30,
        poke_interval=5,
    )

    prepare_data = BashOperator(
        task_id="prepare_data",
        bash_command=f"cd {PROJECT_DIR} && python src/prepare.py data/raw/heart.csv data/prepared",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT_DIR} && python train_ci.py",
    )

    def check_quality(**kwargs):
        metrics_path = os.path.join(PROJECT_DIR, "metrics.json")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        accuracy = float(metrics["accuracy"])
        if accuracy > 0.85:
            return "register_model"
        return "stop_pipeline"

    evaluate = BranchPythonOperator(
        task_id="evaluate_model",
        python_callable=check_quality,
    )

    register_model = BashOperator(
        task_id="register_model",
        bash_command=f"echo 'Model registered with accuracy > 0.85'",
    )

    stop_pipeline = BashOperator(
        task_id="stop_pipeline",
        bash_command=f"echo 'Model quality below threshold, stopping'",
    )

    check_data >> prepare_data >> train_model >> evaluate
    evaluate >> register_model
    evaluate >> stop_pipeline