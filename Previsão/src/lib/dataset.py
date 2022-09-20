import pandas as pd
import numpy as np
import tempfile
from datetime import datetime, timedelta
from meteostat import Hourly
import lib.config as config
import PIconnect as PI
import pickle


class Dataset():
    def __init__(self, dataset_path = None, scaler_path = None):
        self._temp_path = tempfile.mkdtemp()
        self._dataset_path = dataset_path if dataset_path else self._temp_path
        self._scaler_path = scaler_path if dataset_path else self._temp_path
        self.raw_dataset = None

    def _get_pi_data(self, start_date, end_date):
        with PI.PIServer() as server:
            points = server.search(config.TAG)
            data = points[0].recorded_values(start_date, end_date)
        # resample data
        data = pd.DataFrame(data)
        data = data.resample('H').mean()
        data = data.interpolate('linear')
        data.columns = ["power"]
        data.index = data.index - timedelta(hours=3) 
        return data
        
    def _get_weather_data(self, start, end):
        data = Hourly(config.WEATHER_STATION_ID, start, end)
        data = data.fetch()
        data = data.resample('H').mean()
        data = data.interpolate('linear')
        return data
    
    def _get_dataset(self, current_date, window_size):
        start_date = current_date - timedelta(days=window_size) + timedelta(hours=1)
        end_date = current_date 
        weather_data = self._get_weather_data(start_date + timedelta(days=1), end_date + timedelta(days=1)) # shitf 1 day
        
        end_date = end_date + timedelta(hours=1)
        power_data = self._get_pi_data(start_date.strftime(config.DATE_FORMAT), end_date.strftime(config.DATE_FORMAT))
        
        power_data.index = weather_data.index
        df = pd.concat([power_data,weather_data],axis=1)
        return df
    
    def _add_features(self, dataset):
        dataset["weekday"] = dataset.index.dayofweek
        weekend = [1 if day > 4 else 0 for day in dataset.index.dayofweek]
        dataset["weekend"] = weekend
        return dataset        
    
    def _dataset_order(self, dataset):
        # ['power', 'wspd', 'weekday', 'weekend', 'temp']
        df = pd.DataFrame()
        df["power"] = dataset.power
        df["wspd"] = dataset.wspd
        df["weekday"] = dataset.weekday
        df["weekend"] = dataset.weekend
        df["temp"] = dataset.temp
        df.index = dataset.index
        return df
    
    def _filter_data(self, dataset, feature_list):
        return dataset.drop(dataset.columns.difference(feature_list), 1)

    def _scale_data(self, dataset, scaler = None):
        scaler = pickle.load(open(config.SCALER_PATH, 'rb'))
        normalized_dataset = scaler.transform(dataset)
        df = pd.DataFrame(normalized_dataset, columns = dataset.columns, index = dataset.index)
        return df

    def preprocessing_data(self):
        dataset = self._filter_data(self.raw_dataset, config._FEATURE_LIST)
        dataset = self._add_features(dataset)
        dataset = self._dataset_order(dataset)
        dataset = self._scale_data(dataset)
        return np.array([dataset.to_numpy()])

    def download_data(self):
        date = datetime.now().replace(minute=0, second=0, microsecond=0)
        self.raw_dataset = self._get_dataset(date, window_size=config.WINDOW_WIDTH_DAYS)
        return self.raw_dataset