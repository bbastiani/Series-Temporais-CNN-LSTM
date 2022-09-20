import pickle
import pandas as pd
import numpy as np
import lib.config as config
from matplotlib import pyplot as plt
from datetime import timedelta

def inv_transform(data, colName, colNames):
    scaler = pickle.load(open(config.SCALER_PATH, 'rb')) 
    dummy = pd.DataFrame(np.zeros((len(data), len(colNames))), columns=colNames)
    dummy[colName] = data
    dummy = pd.DataFrame(scaler.inverse_transform(dummy), columns=colNames)
    return dummy[colName].values

def plot_results(y, y_hat, timestamp, filename):

    font = {
        "family": "Calibri",
        "weight": "normal",
        "size": 16
    }

    plt.rc('font',**font)

    fig = plt.figure(figsize=(9,6))
    plt.plot(timestamp-timedelta(hours=24),y, color="#218ED3", linewidth=2)
    plt.step(timestamp,y_hat, color="#ff7f0e", linewidth=2)
    plt.xlabel("Amostras")
    plt.ylabel("Intercâmbio SIN-PY [MW]")
    plt.legend(["Valor Observado", "Valor Previsto"], frameon=False)
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()
    fig.savefig(filename,bbox_inches='tight')