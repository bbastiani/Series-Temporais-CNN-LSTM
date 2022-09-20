import tensorflow as tf
import lib.config as config
import pandas as pd
from lib.utils import inv_transform, plot_results
from lib.dataset import Dataset
import logging

logging.basicConfig(
    filename=f'{config.LOG_PATH}/log.log',
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

def dataset_generator():
    data = Dataset()
    data.download_data()
    return data.preprocessing_data(), data

def model_generator():
    return tf.keras.models.load_model(config.MODEL_PATH)

def main():
    dataset, dataset_obj = dataset_generator()
    model = model_generator()
    
    prediction = model.predict(dataset)[0]
    y_hat = inv_transform(prediction, config._LABEL_LISTS[0], config._FEATURE_LIST)
    y = dataset_obj.raw_dataset.power[-24:]
    timestamp = dataset_obj.raw_dataset.index[-24:]
    plot_results(y, y_hat, timestamp, "previsao.svg")
    
    df = pd.DataFrame(y_hat, index=timestamp, columns=["ANDE"])
    df.to_csv("previsao.csv")
    
if __name__ == "__main__":
    main()