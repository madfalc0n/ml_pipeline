## Airflow module list
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models.baseoperator import chain

## Common module list
import numpy as np
import sys
sys.path.append("/home/madfalcon/model_trainer")

## Custom module list
from data_load import dataset
from modeling import model_
save_path = "/home/madfalcon/data"

default_args = {
    'start_date': days_ago(0),
}

dag = DAG(
    dag_id='DAG_ML_pipline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)

def T_dataset_loading():
    print("Task 1")
    dset = dataset()
    print("Dataset Loading Process")
    dset.load_dataset()
    dset.split_data()
    print("Dataset Save Process")
    result = dset.save_data(path=save_path)
    return result

def T_model_training(**context):
    print("Task 2")
    print("model training")
    result = context['ti'].xcom_pull(task_ids='T_dataset_loading', key='return_value')
    x_train = np.load(result['x_train'])
    y_train = np.load(result['y_train'])
    x_test = np.load(result['x_test'])
    y_test = np.load(result['y_test'])
    print("Train Set:", x_train.shape, y_train.shape)
    print("Test Set:", x_test.shape, y_test.shape)
    model = model_()
    model.fit([x_train,y_train])
    model.predict([x_test,y_test])
    return model

def T_model_serving(**context):
    print("Model Save Process")
    model = context['ti'].xcom_pull(task_ids='T_model_training', key='return_value')
    model.save_()

T_dataset_loading = PythonOperator(
    task_id='T_dataset_loading',
    python_callable=T_dataset_loading,
    dag=dag
)
T_model_training = PythonOperator(
    task_id='T_model_training',
    python_callable=T_model_training,
    dag=dag
)
T_model_serving = PythonOperator(
    task_id='T_model_serving',
    python_callable=T_model_serving,
    dag=dag
)


chain(T_dataset_loading, T_model_training, T_model_serving)