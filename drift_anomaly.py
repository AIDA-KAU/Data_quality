import numpy.fft
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest


data = pd.read_csv('ESR_M_min.csv', index_col=0).iloc[:, :-1]
densities = []

range_ = (0, 1500)
bins = 30


#   Estimate the density function using histograms
def hist(ts):
    hist, bin_edges = np.histogram(ts, range=range_, bins=bins, density=True)
    return hist * np.diff(bin_edges)


#   Calcuate pdf and save into numpy array
for index, row in data.iterrows():
    pdf = hist(row)
    densities.append(pdf)

density = np.mean(densities, axis=0)
np.save('density.npy', density)


#   Train anamoly detction algorithm and save it into a file
model = IsolationForest(n_jobs=-1, contamination=0.001, warm_start=False)
model.fit(np.array(data.iloc[2]).reshape(-1, 1))
filename = 'anamoly_detection.sav'
pickle.dump(model, open(filename, 'wb'))
