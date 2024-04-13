from datetime import datetime, timedelta
import pandas as pd
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import mlflow
import os
import tempfile
from pathlib import Path
import boto3
import joblib

from botocore.exceptions import NoCredentialsError
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 4, 6),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def load_and_transform_data(**kwargs):
    # Ruta del archivo JSON
    json_file_path = '/opt/airflow/data/collected_data.json'

    # Cargar el JSON como diccionario
    with open(json_file_path, 'r') as f:
        data_dict = json.load(f)

    # Obtener los datos de interés
    data = data_dict['data']

    column_names = [
    'Elevation', 
    'Aspect', 
    'Slope', 
    'Horizontal_Distance_To_Hydrology', 
    'Vertical_Distance_To_Hydrology', 
    'Horizontal_Distance_To_Roadways', 
    'Hillshade_9am', 
    'Hillshade_Noon', 
    'Hillshade_3pm', 
    'Horizontal_Distance_To_Fire_Points', 
    'Wilderness_Area', 
    'Soil_Type',
    'Cover_Type']

    # Crear un DataFrame con los datos
    df = pd.DataFrame(data, columns=column_names)

    #one hot
    df = pd.get_dummies(df, columns = ['Wilderness_Area','Soil_Type'], drop_first=True)

    # Guardar el DataFrame como una variable en el contexto de Airflow
    kwargs['ti'].xcom_push(key='dataframe', value=df)

def train_model(**kwargs):
    # Preparación del entorno
    ti = kwargs['ti']
    df = ti.xcom_pull(task_ids='load_and_transform_task', key='dataframe')

    X = df.drop(columns=['Cover_Type'])
    y = df['Cover_Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configuración de MLflow
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:1880')
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.start_run(run_name="RandomForestClassifier")

    # Entrenamiento del modelo
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Evaluación del modelo
    predictions = rf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(rf, "RandomForestClassifier")

    print(4)
    s3_client = boto3.client('s3',
                             endpoint_url='http://localhost:9001',
                             aws_access_key_id='minio',
                             aws_secret_access_key='minio123',
                             )

    # Guardar el modelo entrenado en un directorio temporal
    with tempfile.TemporaryDirectory() as tmpdirname:
        model_filename = Path(tmpdirname) / "RandomForestClassifier.pkl"
        joblib.dump(rf, model_filename)

        # Intenta subir el modelo a MinIO
        try:
            response = s3_client.upload_file(str(model_filename), 'bucket', 'RandomForestClassifier.pkl')
            print("Modelo subido exitosamente a MinIO.")
        except NoCredentialsError:
            print("Error de credenciales al subir el archivo a MinIO.")

    mlflow.end_run()



# Definir el DAG
dag = DAG(
    'json_to_dataframe_and_train_model',
    default_args=default_args,
    description='DAG to load JSON, transform to DataFrame, and train model',
    schedule_interval=timedelta(days=1),  # Frecuencia de ejecución
)

# Tareas del DAG
start_task = DummyOperator(task_id='start_task', dag=dag)

load_and_transform_task = PythonOperator(
    task_id='load_and_transform_task',
    python_callable=load_and_transform_data,
    provide_context=True,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model,
    provide_context=True,
    dag=dag,
)

end_task = DummyOperator(task_id='end_task', dag=dag)

# Definir el orden de las tareas
start_task >> load_and_transform_task >> train_model_task >> end_task
