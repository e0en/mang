import cPickle as pickle
import gzip
import time

import numpy as np
import scipy.io as sio
import msgpack
import msgpack_numpy as m
m.patch()


_TIME = None


def tic():
    global _TIME
    _TIME = time.time()


def toc():
    return time.time() - _TIME


def loadDefault(given, default):
    result = dict(default)
    for key in given:
        result[key] = given[key]
    return result


def save_pickle(filename, data):
    with open(filename, 'wb') as fp:
        pickle.dump(data, fp)
        fp.close()


def load_pickle(filename):
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
    return data


# use it to save models
def save_pkl_gz(filename, data):
    with gzip.open(filename, 'wb') as fp:
        pickle.dump(data, fp)
        fp.close()


def load_pkl_gz(filename):
    with gzip.open(filename, 'rb') as fp:
        data = pickle.load(fp)
    return data


# use it to save data
def save_npz(filename, data):
    if type(data) == dict:
        np.savez(filename, **data)
    else:
        np.save(filename, data)


def load_npz(filename):
    data = np.load(filename)
    return data


def save_msgpack(filename, data):
    with open(filename, "wb") as fp:
        fp.write(msgpack.packb(data))


def load_msgpack(filename):
    with open(filename, "rb") as fp:
        data = msgpack.unpackb(fp.read())
        return data


def load_file(filename):
    ext_table = {
        ".pkl": load_pickle,
        ".pkl.gz": load_pkl_gz,
        ".npy": np.load,
        ".npz": np.load,
        ".msgpack": load_msgpack,
        ".mat": sio.loadmat,
        }

    for ext in ext_table:
        if filename.endswith(ext):
            return ext_table[ext](filename)

    raise ValueError("Unknown file format")
