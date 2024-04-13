import requests
import json
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

# Configuración predeterminada de las tareas
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 4, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def obtener_datos():
    url = "http://10.43.101.149/data?group_number=4"  # URL de la API de datos
    response = requests.get(url)

    if response.status_code == 200:
        datos = response.json()
        # Ruta al archivo donde se guardarán los datos
        file_path = "/opt/airflow/data/collected_data.json"#

        # Guardar los datos en un archivo JSON
        with open(file_path, "w") as file:
            json.dump(datos, file)
            print(f"Datos guardados en {file_path}")
    else:
        print(f"Error al obtener los datos: {response.status_code}")

# Definición del DAG
dag = DAG('recolectar_datos_api_grupo4', default_args=default_args, schedule_interval='*/5 * * * *')

def tarea_recolectar_datos():
    # Llamada a la función ajustada para el grupo 4
    obtener_datos()  # Asegúrate de que esta función esté definida o importada correctamente

# Creación y configuración de la tarea en Airflow
recolectar_datos = PythonOperator(
    task_id='recolectar_datos_grupo4',
    python_callable=tarea_recolectar_datos,
    dag=dag,
)


