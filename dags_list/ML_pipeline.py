## Airflow module list
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models.baseoperator import chain
from airflow.operators.bash_operator import BashOperator

## Common module list
import numpy as np
import sys
sys.path.append('/home/madfalcon/git/ml_pipeline/PIPELINE')

## Custom module list
from dataset_module.data_load_MNIST import c_dataset
from model_module.modeling_DL import modeling_main

default_args = {
    'start_date': days_ago(0),
}

dag = DAG(
    dag_id='DAG_ML_pipline',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)

bash_cmd = """
        cd /home/madfalcon/git/ml_pipeline || exit 1
        git pull origin main
    """
    
def T_dataset_loading():
    """
        학습가능한 Dataset으로 변환 후 저장하는 Task
    """
    save_path = "/home/madfalcon/data/MNIST_trainable"
    print("Task 1")
    # iglabel=[1,2,3,4]
    iglabel=[]
    dset = c_dataset(ignore_label=iglabel)
    print("Dataset Loading Process")
    dset.load_dataset()
    dset.split_data()
    print("Dataset Save Process")
    result = dset.save_data(path=save_path)
    return result

def T_model_training(**context):
    """
        학습가능한 데이터 저장경로를 xcom을 통해 수신 후 모델학습코드 실행 Task
    """
    print("Task 2")
    print("model training")
    save_path_dict = context['ti'].xcom_pull(task_ids='T_dataset_loading', key='return_value')
    model_save_path = "/home/madfalcon/madfalcon_lab/model"
    print("model save path:", model_save_path)
    modeling_main(save_path_dict,save_path=model_save_path)

T_code_update = BashOperator(
        task_id='T_code_update',
        bash_command=bash_cmd
    )

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


chain(T_code_update, T_dataset_loading, T_model_training)