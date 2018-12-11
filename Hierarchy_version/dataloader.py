import numpy as np
import glob, os, sys
import pickle
import math
import signal
import configparser, argparse
import ast
import re
from itertools import combinations


def load_dataset(path=None):
    '''load the train and test datasets'''
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    return data


def load_dataset_pool(data_id, poolsize, path_list=None):
    '''load the train dataset in a pool of poolsize'''
    data_pool = {}
    for data_path in path_list[data_id: data_id + poolsize]:
        with open(data_path, 'rb') as f:
            data = pickle.load(f, encoding="latin1")
            data_pool.update(data)
        print("loaded data {} into data pool".format(data_path))
    return data_pool


def point_data_generator(data, max_q_len):
    """ make use of all queries and all docs
        generate (q, d) in point-wise scheme
    """
    qid = []
    Q = []
    D = []
    label = []
    q_list = list(data.keys())
    for q_id in q_list:
        query = data[q_id]['query']
        if (1 not in query and len(query) <= max_q_len):  # no OOV token in query
            docs = data[q_id]['docs']
            scores = data[q_id]['scores']
            Q += [query] * len(docs)
            D += docs
            qid += [q_id] * len(docs)
            label += scores
    return [qid, Q, D, label]


def pair_data_generator(data, max_q_len):
    '''
    make use of all queries and all docs
    for a given query q, list all possible document pair combinations, and assign to D_pos, D_neg
    according to its score
    '''
    Q = []
    D_pos = []
    D_neg = []
    label = []
    q_list = list(data.keys())
    for q_id in q_list:
        query = data[q_id]['query']
        q_list = data[q_id]['q_list']
        if (1 not in query and len(query) <= max_q_len):  # no OOV token in query
            docs = data[q_id]['docs'] #[ [[1,2,...],[1,2,3...]] , [[1,2,4..],[3,4,5]], ... ]
            scores = data[q_id]['scores']
            if len(docs) >=2:
                idx = []
                idx.extend(combinations(range(len(docs)), 2))
                for i in range(len(idx)):
                    if scores[idx[i][0]][0] - scores[idx[i][1]][0] > 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][0]]) #[[1,2,3...],[1,2,3..]]
                        D_neg.append(docs[idx[i][1]])
                        label.append([scores[idx[i][0]], scores[idx[i][1]]]) # [[1,1,1,1..], [2,2,2,2...]]
                    if scores[idx[i][0]][0] - scores[idx[i][1]][0] < 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][1]])
                        D_neg.append(docs[idx[i][0]])
                        label.append([scores[idx[i][1]], scores[idx[i][0]]])
    return [Q, D_pos, D_neg, label]