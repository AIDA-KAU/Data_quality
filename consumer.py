from kafka import KafkaConsumer
import json
import numpy as np
from sklearn.ensemble import IsolationForest
import pandas as pd
from scipy.spatial import distance
from threading import Thread
import time
import pickle
import yaml
from multiprocessing import Process


range_ = (0, 1500)
bins = 30


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)

        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return


def check_constraints(data_point):
    s = 0
    key, value = zip(*data_point.items())
    values = value[0]
    if not config['min_negative']:
        if values[0] < config['min_value'] or any(t < 0 for t in values):
            s = 1
        return s
    else :
        if values[0] < config['min_value']:
            s = 1
        return s


def check_null(data_point):
    s = 0
    key, value = zip(*data_point.items())
    values = value[0]
    if not config['replace_null']:
        if np.isnan(values).any():
            s = 1
    return s


def check_anomaly(data_point):
    key, value = zip(*data_point.items())
    values = np.array(value[0][:-1])[~np.isnan(np.array(value[0][:-1]))]
    model = pickle.load(open('anamoly_detection.sav', 'rb'))
    # model.fit(values.reshape(-1, 1))
    # score = model.decision_function(values.reshape(-1, 1))
    s = model.predict(values.reshape(-1, 1))
    if not config['remove_anomaly']:
        if -1 in s[5:]:
            r = 1
        else:
            r = 0
        if r == 1:
            print(key)
    return r


def check_drift(data_point):
    density = np.load('density.npy')
    key, value = zip(*data_point.items())
    values = value[0]
    hist, bin_edges = np.histogram(values, range=range_, bins=bins, density=True)
    pdf = hist * np.diff(bin_edges)
    if not config['drift_threshold']:
        div = distance.jensenshannon(pdf, density)** 2
    return div


if __name__ == '__main__':
    with open(r'config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    consumer = KafkaConsumer(config['topic'], group_id=config['group_id'],
                             bootstrap_servers=config['bootstrap_servers'],
                             auto_offset_reset=config['auto_offset_reset'])
    data_points = {}
    score_df = pd.DataFrame(columns={'values', 'score'})
    for message in consumer:
        # buffer = message.append()

        msg = json.loads(message.value)[0]
        index = str(msg[0])
        values = msg[1:]
        data_points = {index: values}
        # print(data_points)
        # print('seq')
        start = time.time()
        con_score = check_constraints(data_points)
        null_score = check_null(data_points)
        anom_score = check_anomaly(data_points)
        drift_score = check_drift(data_points)
        a = config['dcw']
        b = config['nvw']
        c = config['amw']
        d = config['dfw']
        #   Normalize the score between 0 and 1
        norm = a+b+c+d
        score = 1/norm*( a*con_score + b*null_score + c*anom_score + d*drift_score)
        # print(score)
        end = time.time()
        # print(end - start)
        score_df.loc[index, 'values'] = values
        score_df.loc[index, 'score'] = score
        df = pd.DataFrame(score_df['values'].to_list(), index=score_df.index)
        df['score'] = score_df['score']
        df.to_csv('score.csv', index=True)
