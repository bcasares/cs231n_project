# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np
import pandas as pd
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from preprocess_data import logTotalValue

def logTotalValue(data, column_id="TotalValue"):
  data["log_total_value"] = data[column_id].apply(lambda x: np.log(x))


def readCsvFiles():
  train = pd.read_csv("data/train.csv")
  test = pd.read_csv("data/test.csv")
  val = pd.read_csv("data/val.csv")
  logTotalValue(train)
  logTotalValue(test)
  logTotalValue(val)
  return train, test, val


def ligthData(x, y, x_test, y_test):
  train_data = lightgbm.Dataset(x, label=y)
  test_data = lightgbm.Dataset(x_test, label=y_test)
  return train_data, test_data

def train_test(train_data, test_data):
  parameters = {
      'application': 'huber',
      'objective': 'regression',
      'metric': 'huber',
      'boosting': 'gbdt',
      'learning_rate': 0.05,
      'verbose': 0
  }

  model = lightgbm.train(parameters,
                         train_data,
                         valid_sets=test_data,
                         num_boost_round=5000,
                         early_stopping_rounds=100)

  return model

def makePred(model, x, y, x_test, y_test):
  y_hat = model.predict(x)
  y_hat_test = model.predict(x_test)
  pred = {"train" : y_hat, "test" : y_hat_test}
  residual = {"train" : (y - y_hat), "test" : (y_test - y_hat_test)}
  return pred, residual





