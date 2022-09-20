import lib.config as config
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from lib.dataset import Dataset
from lib.model import NeuralNetwork
from lib.utils import plot_results

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import ModelSignature

from tensorflow.python.saved_model import signature_constants
import tensorflow as tf

import logging

logging.basicConfig(
    filename=f'{config.LOG_PATH}/mlops.log',
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def dataset_generator():
    args = {
        "input_points": config.INPUT_POINTS,
        "predict_points": config.PREDICT_POINTS,
        "scaler": StandardScaler(),
        "train_test_split": config.TRAIN_TEST_SPLIT,
        "summer_data": config.DATASET_SUMMER,
        "winter_data": config.DATASET_WINTER
    }
    data = Dataset()
    data.download_data(read_pickle_data=config.READ_DATASET_FROM_PICKLE)
    return data.preprocessing_data(args)

def model_generator(train_dataset, test_dataset):
    args = {
        "input_shape": (train_dataset.x.shape[1], len(config._FEATURE_LIST)),
        "max_epocs": config.MAX_EPOCS,
        "patience": config.PATIENCE,
        "learning_rate": config.LEARNING_RATE,
        "batch_size": config.BATCH_SIZE,
        "train_dataset": train_dataset, 
        "test_dataset": test_dataset
    }

    model = NeuralNetwork()
    param_and_metrics = model.train_model(args)
    return model, param_and_metrics


def pipeline_ml():
    logging.info("Start Execution")
    mlflow.set_tracking_uri("http://localhost:5000")
    client = MlflowClient()
    try:
        # Creates a new experiment
        experiment_id = client.create_experiment(f"demanda_sin_py_{'winter' if config.DATASET_WINTER else 'summer'}") # logging.info(f"The experiment {data_name} was created with id={experiment_id} ")
    except:
        experiment_id = client.get_experiment_by_name(f"demanda_sin_py_{'winter' if config.DATASET_WINTER else 'summer'}").experiment_id # Retrieves the experiment id from the already created project

    with mlflow.start_run(experiment_id=experiment_id, run_name=f"cnn_lstm_parallel_{'winter' if config.DATASET_WINTER else 'summer'}"):
        logging.info("Generate and Preprocession Dataset")
        train_dataset, test_dataset = dataset_generator()
        logging.info("Generate and Train Model")
        model, param_and_metrics = model_generator(train_dataset, test_dataset)
        # logging
        for k, v in  param_and_metrics["metrics"].items():
            mlflow.log_metric(k, v)
        for k, v in  param_and_metrics["params"].items():
            mlflow.log_param(k, v)

        logging.info("Save model")
        model.save_model(config.MODEL_PATH)
        # log artifacts
        plot_results(model.model, test_dataset.x, test_dataset.y, test_dataset.timestamp, f"{config.IMAGES_PATH}/result.svg")
        mlflow.log_artifact(f"{config.IMAGES_PATH}/result.svg")
        # model evaluation
        signature = ModelSignature(inputs=config.input_schema, outputs=config.output_schema)

        model_info = mlflow.tensorflow.log_model(tf_saved_model_dir=config.MODEL_PATH,
                                tf_meta_graph_tags=[tf.compat.v1.saved_model.tag_constants.SERVING],
                                tf_signature_def_key=signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
                                artifact_path="tf-models",
                                registered_model_name=f"CNN-LSTM-Parellel-{'winter' if config.DATASET_WINTER else 'summer'}",
                                signature=signature)

        # result = mlflow.evaluate(
        #     model_info.model_uri,
        #     data = test_dataset.x.reshape((test_dataset.x.shape[0], test_dataset.x.shape[1])),
        #     targets = test_dataset.y.reshape((test_dataset.y.shape[0], test_dataset.y.shape[1])),
        #     model_type="regressor",
        #     dataset_name="demanda_sin_py_2022",
        #     evaluators=["default"],
        # )

        # copy artifacts to MLFlow dir
        from distutils.dir_util import copy_tree, remove_tree
        copy_tree("./artifacts","./MLFlow/artifacts")
        remove_tree("./artifacts")

if __name__ == "__main__":
    pipeline_ml()