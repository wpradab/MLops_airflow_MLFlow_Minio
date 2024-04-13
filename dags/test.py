from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 4, 1),
}

dag = DAG(
    'mlflow_example',
    default_args=default_args,
    description='A simple DAG to demonstrate MLflow integration in Airflow',
    schedule_interval=timedelta(days=1),
)

def train_and_log_mlflow():
    # Cargar datos de ejemplo (iris dataset)
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)
    
    # Crear y entrenar un modelo (Random Forest Classifier)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predecir con el modelo
    y_pred = model.predict(X_test)
    
    # Calcular precisión
    accuracy = accuracy_score(y_test, y_pred)
    
    # Inicializar un experimento en MLflow
    mlflow.set_experiment('iris_classification')
    
    with mlflow.start_run():
        # Registrar parámetros y métricas en MLflow
        mlflow.log_param('n_estimators', 100)
        mlflow.log_metric('accuracy', accuracy)
        
        # Guardar el modelo en MLflow
        mlflow.sklearn.log_model(model, 'random_forest_model')

run_mlflow_task = PythonOperator(
    task_id='train_and_log_mlflow',
    python_callable=train_and_log_mlflow,
    dag=dag,
)

run_mlflow_task
