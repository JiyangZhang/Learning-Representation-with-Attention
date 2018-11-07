import numpy as np
import numpy.random as npr
import pickle

if __name__=="__main__":
    from rng import np_rng, py_rng
else:
    from lib.rng import np_rng, py_rng
from sklearn import utils as skutils

def load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    return data

def list_shuffle(*data):
    idxs = np_rng.permutation(np.arange(len(data[0])))
    if len(data) == 1:
        return [data[0][idx] for idx in idxs]
    else:
        return [[d[idx] for idx in idxs] for d in data]

def shuffle(*arrays, **options):
    if isinstance(arrays[0][0], str):
        return list_shuffle(*arrays)
    else:
        return skutils.shuffle(*arrays, random_state=np_rng)

def pad_list(vec, max_len, padding_id=0):
    '''padding id vector with padding id'''
    if len(vec) <= max_len:
        return vec + [padding_id for i in range(max_len - len(vec))]
    else:  # if vec too long just cut
        return vec[:max_len]

def pad_batch_list(batch, max_len, padding_id=0):
    padded_batch = []
    for vec in batch:
        padded_batch.append(pad_list(vec, max_len, padding_id))
    return padded_batch

def pad_vector(L, seq, max_len, padding_id=0):
    '''padding a list of vectors with [0,0,0,0,..]'''
    zero_vec = [0] * seq
    if len(L) <= max_len:
        return L + [zero_vec for i in range(max_len - len(L))]
    else:  # if vec too long just cut
        return L[:max_len]

def pad_nparray(vec, max_len, padding_id=0):
    '''padding id vector with padding id'''
    if vec.shape[0] <= max_len:
        padding = np.zeros(max_len - vec.shape[0])
        return np.concatenate([vec, padding])
    else:  # if vec too long just cut
        return vec[:max_len]


if __name__ == '__main__':
    print("test!OK")
