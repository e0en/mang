import cPickle as pickle
import gzip
import time
import os

import numpy as np


_time = None

def tic():
    global _time
    _time = time.time()

def toc():
    global _time
    return time.time() - _time

def loadDefault(given, default):
    result = dict(default)
    for key in given:
        result[key] = given[key]
    return result

def save_pickle(data, filename):
    # fp = gzip.open(filename, 'wb')
    fp = open(filename, 'wb')
    pickle.dump(data, fp)
    fp.close()

def load_pickle(filename):
    # fp = gzip.open(filename, 'rb')
    fp = open(filename, 'rb')
    data = pickle.load(fp)
    fp.close()
    return data

# use it to save models
def save_pkl_gz(data, filename):
    fp = gzip.open(filename, 'wb')
    pickle.dump(data, fp)
    fp.close()

def load_pkl_gz(filename):
    fp = gzip.open(filename, 'rb')
    data = pickle.load(fp)
    return data

# use it to save data
def save_npz(data, filename):
    if type(data) == dict:
        np.savez(filename, **data)
    else:
        np.savez(filename, data)

def load_npz(filename):
    data = np.load(filename)
    return data
