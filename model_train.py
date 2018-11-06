import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm

import time
import numpy as np
import glob, os, sys
import pickle
import math
import signal
import configparser
import ast

from multidep_match_4L_1d_builder import MultiMatch
from lib.torch_utils import non_neg_normalize, np_softmax
from lib.data_utils import list_shuffle, pad_batch_list
from lib.eval import write_run, compute_ndcg, compute_map
from lib.loss import hinge_loss
from functools import partial

def load_dataset(path=None):
    '''load the train and test datasets'''
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    return data

def max_len(D):
    maxlen = 0
    for doc in D:
        current_len = len(doc)
        if current_len > maxlen:
            maxlen = current_len
    return maxlen

def prepare_data(data, block_size, max_q_len):
    """randomly sample a Q and then from its docs, randomly sample D+, D-,
    repeat this sampling until blocksize reached"""
    Q = []
    D_pos = []
    D_neg = []
    label = []
    q_list = list(data.keys())
    while len(Q) < block_size:
        # random sampling one topic, one D+, one D- and accumulate until block_size
        q_id = np.random.choice(range(len(q_list)), size=(1,), replace=False)
        q_id = q_id[0]
        this_topic = q_list[q_id]
        if (1 not in data[this_topic]['query'] and len(data[this_topic]['query']) <= max_q_len):  # eliminate OOV
            query = data[this_topic]['query']
            docs = data[this_topic]['docs']
            scores = data[this_topic]['scores']
            if len(docs) >= 2:  # more than 2 docs in this group
                idx = np.random.choice(range(len(docs)), size=(2,), replace=False)
                if scores[idx[0]] != scores[idx[1]]:
                    Q.append(query)
                    if scores[idx[0]] > scores[idx[1]]:  # idx0 is pos doc
                        D_pos.append(docs[idx[0]])
                        D_neg.append(docs[idx[1]])
                        label.append([scores[idx[0]], scores[idx[1]]])
                    else:  # idx1 is pos doc
                        D_pos.append(docs[idx[1]])
                        D_neg.append(docs[idx[0]])
                        label.append([scores[idx[1]], scores[idx[0]]])
    return [Q, D_pos, D_neg, label]

def prepare_data_w_diff(data, block_size, max_q_len):
    '''choose D+ D- with a BM25 score difference greater than a threshold'''
    Q = []
    D_pos = []
    D_neg = []
    label = []
    q_list = list(data.keys())
    while len(Q) < block_size:
        # random sampling one topic, one D+, one D- and accumulate until block_size
        q_id = np.random.choice(range(len(q_list)), size=(1,), replace=False)
        q_id = q_id[0]
        this_topic = q_list[q_id]
        if (1 not in data[this_topic]['query'] and len(data[this_topic]['query']) <= max_q_len):   # eliminate OOV
            query = data[this_topic]['query']
            docs = data[this_topic]['docs']
            scores = data[this_topic]['scores']
            if len(docs) >= 2:  # more than 2 docs in this group
                idx = np.random.choice(range(len(docs)), size=(2,), replace=False)
                if abs(scores[idx[0]] - scores[idx[1]]) >= 0.5:
                    Q.append(query)
                    if scores[idx[0]] > scores[idx[1]]:  # idx0 is pos doc
                        D_pos.append(docs[idx[0]])
                        D_neg.append(docs[idx[1]])
                        label.append([scores[idx[0]], scores[idx[1]]])
                    if scores[idx[1]] > scores[idx[0]]:  # idx1 is pos doc
                        D_pos.append(docs[idx[1]])
                        D_neg.append(docs[idx[0]])
                        label.append([scores[idx[1]], scores[idx[0]]])
    return [Q, D_pos, D_neg, label]

def prepare_data_all_Q(data, block_size, max_q_len):
    '''make use of all queries instead of randomly sampling queries
    for a given query, still sample docs uniformly
    (Q1, D1+, D1-)....(Q1, Dn+, Dn-), (Q2,D1+,D1-)...(Q2, Dn+, Dn-)
    block_size: for a given q, how many pairs of D+, D- generated
    '''
    Q = []
    D_pos = []
    D_neg = []
    label = []
    q_list = list(data.keys())
    for q_id in q_list:
        query = data[q_id]['query']
        if (1 not in query and len(query) <= max_q_len):  # no OOV token in query
            docs = data[q_id]['docs']
            scores = data[q_id]['scores']
            if len(docs) >=2:
                idx = np.random.choice(range(len(docs)), size=(block_size, 2), replace=True)
                for i in range(idx.shape[0]):
                    if scores[idx[i][0]] - scores[idx[i][1]] > 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][0]])
                        D_neg.append(docs[idx[i][1]])
                        label.append([scores[idx[i][0]], scores[idx[i][1]]])
                    if scores[idx[i][0]] - scores[idx[i][1]] < 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][1]])
                        D_neg.append(docs[idx[i][0]])
                        label.append([scores[idx[i][1]], scores[idx[i][0]]])
    return [Q, D_pos, D_neg, label]

def prepare_data_sampleQ_BM25distro(data, q_sample_size, docpair_sample_size, max_q_len):
    """sample randomly a query
    for a given query, sample doc according to the distro softmax(BM25_scores)
    q_sample_size: num of queries sampled from one data pkl file
    docpair_sample_size: for each q, how many pairs of (D+, D-) sampled
    """
    Q = []
    D_pos = []
    D_neg = []
    label = []
    q_list = list(data.keys())
    Q_counter = 0
    while Q_counter < q_sample_size:
        # random sampling one topic, one D+, one D- and accumulate until block_size
        q_idx = np.random.choice(range(len(q_list)), size=(1,), replace=False)
        q_idx = q_idx[0]  # idx from 0 to len(q_list)
        topic_num = q_list[q_idx]
        query = data[topic_num]['query']
        if (1 not in query and len(query) <= max_q_len):  # no OOV token in query
            docs = data[topic_num]['docs']
            scores = data[topic_num]['scores']
            if len(docs) >=2:
                # calcuate BM25 score softmax distribution
                np_scores = np.asarray(scores)
                BM25_distro = np_softmax(np_scores)
                idx = np.random.choice(range(len(docs)), size=(docpair_sample_size, 2), replace=True, p=BM25_distro)
                for i in range(idx.shape[0]):
                    if scores[idx[i][0]] - scores[idx[i][1]] > 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][0]])
                        D_neg.append(docs[idx[i][1]])
                        label.append([scores[idx[i][0]], scores[idx[i][1]]])
                    if scores[idx[i][0]] - scores[idx[i][1]] < 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][1]])
                        D_neg.append(docs[idx[i][0]])
                        label.append([scores[idx[i][1]], scores[idx[i][0]]])
                Q_counter += 1
    return [Q, D_pos, D_neg, label]

def prepare_data_all_Q_BM25distro(data, block_size, max_q_len):
    """make use of all queries instead of randomly sampling queries
    for a given query, sample doc according to the distro softmax(BM25_scores)
    block_size: for a given q, how many pairs of D+, D- sampled
    """
    Q = []
    D_pos = []
    D_neg = []
    label = []
    q_list = list(data.keys())
    for q_id in q_list:
        query = data[q_id]['query']
        if (1 not in query and len(query) <= max_q_len):  # no OOV token in query
            docs = data[q_id]['docs']
            scores = data[q_id]['scores']
            if len(docs) >=2:
                # calcuate BM25 score softmax distribution
                np_scores = np.asarray(scores)
                BM25_distro = np_softmax(np_scores)
                idx = np.random.choice(range(len(docs)), size=(block_size, 2), replace=True, p=BM25_distro)
                for i in range(idx.shape[0]):
                    if scores[idx[i][0]] - scores[idx[i][1]] > 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][0]])
                        D_neg.append(docs[idx[i][1]])
                        label.append([scores[idx[i][0]], scores[idx[i][1]]])
                    if scores[idx[i][0]] - scores[idx[i][1]] < 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][1]])
                        D_neg.append(docs[idx[i][0]])
                        label.append([scores[idx[i][1]], scores[idx[i][0]]])
    return [Q, D_pos, D_neg, label]


def prepare_data_allQ_allD(data, max_q_len):
    '''make use of all queries and all docs
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
        if (1 not in query and len(query) <= max_q_len):  # no OOV token in query
            docs = data[q_id]['docs']
            scores = data[q_id]['scores']
            if len(docs) >=2:
                idx = []
                idx.extend(combinations(range(len(docs)), 2))
                for i in range(len(idx)):
                    if scores[idx[i][0]] - scores[idx[i][1]] > 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][0]])
                        D_neg.append(docs[idx[i][1]])
                        label.append([scores[idx[i][0]], scores[idx[i][1]]])
                    if scores[idx[i][0]] - scores[idx[i][1]] < 0.0:
                        Q.append(query)
                        D_pos.append(docs[idx[i][1]])
                        D_neg.append(docs[idx[i][0]])
                        label.append([scores[idx[i][1]], scores[idx[i][0]]])
    return [Q, D_pos, D_neg, label]

def int_handler(sess, model_path, saver, signal, frame):
    '''ctrl+C interrupt handler'''
    print('You pressed Ctrl+C!, model will be saved and stopping training now')
    save_path = saver.save(sess, model_path)
    print('successfully saved model to {}'.format(model_path))
    sys.exit(0)




def train(config_path, resume_training=False):
    """
    Load the parameters
    """
    config = configparser.ConfigParser()
    config.read(config_path)

    batch_sampling_method = config['params']['batch_sampling_method']
    model_name_str = config['hyperparams']['model_name_str']
    
    '''common parameters'''
    batch_size = ast.literal_eval(config['hyperparams']['batch_size'])  # batch_size
    vocab_size = ast.literal_eval(config['hyperparams']['vocab_size'])  # vocab_size
    emb_size = ast.literal_eval(config['hyperparams']['emb_size'])  # embedding dimension
    hidden_size = ast.literal_eval(config['hyperparams']['hidden_size'])
    dropout = ast.literal_eval(config['hyperparams']['dropout'])
    preemb = ast.literal_eval(config['hyperparams']['preemb'])
    sim_type = config['hyperparams']['sim_type']
    hinge_margin = ast.literal_eval(config['hyperparams']['hinge_margin'])
    emb_tune = ast.literal_eval(config['commons']['emb_tune'])
    n_epoch = ast.literal_eval(config['hyperparams']['n_epoch']) # num of epochs
    # q and doc cuts
    q_len = ast.literal_eval(config['hyperparams']['q_len'])
    d_len = ast.literal_eval(config['hyperparams']['d_len'])
    # base_path
    data_base_path = config['hyperparams']['data_base_path']  #'/scratch/nyfbb'
    model_base_path = config['hyperparams']['model_base_path']  #'/home/nyfbb/exp'
    # dataset and fold
    dataset = config['commons']['dataset']
    fold = ast.literal_eval(config['commons']['fold'])
    
    '''representation hyperparams'''
    filt_size_rep = ast.literal_eval(config['rep']['filt_size'])
    kernel_size_rep = ast.literal_eval(config['rep']['kernel_size'])
    output_dim_rep = ast.literal_eval(config['rep']['output_dim'])

    # for sampleQ sampleD prepare_data()
    q_sample_size = ast.literal_eval(config['hyperparams']['q_sample_size'])
    docpair_sample_size = ast.literal_eval(config['hyperparams']['docpair_sample_size'])
    
    alpha = ast.literal_eval(config['hyperparams']['alpha'])  # weight decay
    learning_rate = ast.literal_eval(config['hyperparams']['learning_rate'])

    # new hyper parameters
    num_heads = ast.literal_eval(config['hyperparams']['num_heads'])
    emb_path = ast.literal_eval(config['hyperparams']['emb_path'])
    # parameters that maybe useless
    pool_mode = config['hyperparams']['pool_mode']
    Mtopk = ast.literal_eval(config['hyperparams']['Mtopk'])

    # use cuda flag
    use_cuda = True
    

    """
    the tranining directory
    """
    # load data
    TRAIN_DIR01 = "{}/MQ2007/S1/".format(data_base_path)
    TRAIN_DIR02 = "{}/MQ2007/S2/".format(data_base_path)
    TRAIN_DIR03 = "{}/MQ2007/S3/".format(data_base_path)
    TRAIN_DIR04 = "{}/MQ2007/S4/".format(data_base_path)
    TRAIN_DIR05 = "{}/MQ2007/S5/".format(data_base_path)

    TEST_DIR01 = '{}/MQ2007/S1/'.format(data_base_path)
    TEST_DIR02 = '{}/MQ2007/S2/'.format(data_base_path)
    TEST_DIR03 = '{}/MQ2007/S3/'.format(data_base_path)
    TEST_DIR04 = '{}/MQ2007/S4/'.format(data_base_path)
    TEST_DIR05 = '{}/MQ2007/S5/'.format(data_base_path)

    train_files01 = glob.glob("{}/data0.pkl".format(TRAIN_DIR01))
    train_files02 = glob.glob("{}/data0.pkl".format(TRAIN_DIR02))
    train_files03 = glob.glob("{}/data0.pkl".format(TRAIN_DIR03))
    train_files04 = glob.glob("{}/data0.pkl".format(TRAIN_DIR04))
    train_files05 = glob.glob("{}/data0.pkl".format(TRAIN_DIR05))

    test_files01 = glob.glob("{}/testdata0.pkl".format(TEST_DIR01))
    test_files02 = glob.glob("{}/testdata0.pkl".format(TEST_DIR02))
    test_files03 = glob.glob("{}/testdata0.pkl".format(TEST_DIR03))
    test_files04 = glob.glob("{}/testdata0.pkl".format(TEST_DIR04))
    test_files05 = glob.glob("{}/testdata0.pkl".format(TEST_DIR05))

    if fold == 1:
        train_files = train_files01 + train_files02 + train_files03
        test_files = test_files04[0]  # a path list ['/...'] only take the str
        rel_path = '{}/{}/tmp/test/S4.qrels'.format(model_base_path, model_name_str)
    elif fold == 2:
        train_files = train_files02 + train_files03 + train_files04
        test_files = test_files05[0]
        rel_path = '{}/{}/tmp/test/S5.qrels'.format(model_base_path, model_name_str)
    elif fold == 3:
        train_files = train_files03 + train_files04 + train_files05
        test_files = test_files01[0]
        rel_path = '{}/{}/tmp/test/S1.qrels'.format(model_base_path, model_name_str)
    elif fold == 4:
        train_files = train_files04 + train_files05 + train_files01
        test_files = test_files02[0]
        rel_path = '{}/{}/tmp/test/S2.qrels'.format(model_base_path, model_name_str)
    elif fold == 5:
        train_files = train_files05 + train_files01 + train_files02
        test_files = test_files03[0]
        rel_path = '{}/{}/tmp/test/S3.qrels'.format(model_base_path, model_name_str)
    else:
        raise ValueError("wrong fold num {}".format(fold))
    
    """!!!!!!!!!!"""    
    X = prepare_train_data(0, 3, q_len, d_len, train_files)
    valid_data = load_dataset(test_files)


    """
    Build the model
    """
    model = Attention(word_dim=emb_size, query_length=q_len, doc_length=d_len, num_heads=num_heads, 
        kernel_size=kernel_size_rep, filter_size=filt_size, vocab_size=vocab_size, bias_mask=None,
        dropout=0.0, qrep_dim=output_dim_rep, hidden_size=hidden_size,  batch_size=batch_size, preemb=preemb,
        preemb_path=emb_path)

    if use_cuda:
        model.cuda()
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=alpha)
    # loss func
    loss = nn.MarginRankingLoss(margin=hinge_margin, size_average=True)
    # experiment
    print("Experiment")
    if resume_training == False:
        f_log = open('{}/{}/logs/training_log.txt'.format(model_base_path, model_name_str), 'w+', 1)
        valid_log = open('{}/{}/logs/valid_log.txt'.format(model_base_path, model_name_str), 'w+', 1)
    else:
        f_log = open('{}/{}/logs/training_log.txt'.format(model_base_path, model_name_str), 'a+', 1)
        valid_log = open('{}/{}/logs/valid_log.txt'.format(model_base_path, model_name_str), 'a+', 1)
    # model_file
    model_file = '{}/{}/saves/model_file'.format(model_base_path, model_name_str)
    

    """
    TRAINING
    """

    # init best validation MAP value
    best_MAP = 0.0
    best_NDCG1 = 0.0
    # restore saved parameter if resume_training is true
    if resume_training == True:
        model_file = '{}/{}/saves/model_file'.format(model_base_path, model_name_str)
        model.load_state_dict(torch.load(model_file))
        with open('{}/{}/saves/best_MAP.pkl'.format(model_base_path, model_name_str), 'rb') as f_MAP:
            best_MAP = pickle.load(f_MAP)
        print("loaded model, and resume training now")
    # preparing batch data
    file_count = 0
    # gen Hingeloss Y labels always 1 (S+ > S-)
    if use_cuda:
        Y = Variable(torch.ones(batch_size), requires_grad=False).cuda()
    else:
        Y = Variable(torch.ones(batch_size), requires_grad=False)
    for epoch in range(1, n_epoch + 1):
        '''load_data'''
        data = load_dataset(train_files)
        print("loaded {}".format(train_files))
        '''prepare_data'''
        [Q, D_pos, D_neg, L] = prepare_data_sampleQ_BM25distro(data, q_sample_size, docpair_sample_size, q_len)
        ''' shuffle data'''
        train_data = list_shuffle(Q, D_pos, D_neg, L)
        '''training func'''
        batch_count_tr = 0
        num_batch = len(train_data[0]) // batch_size
        for batch_count in range(num_batch):
            Q = train_data[0][batch_size * batch_count: batch_size * (batch_count + 1)]
            D_pos = train_data[1][batch_size * batch_count: batch_size * (batch_count + 1)]
            D_neg = train_data[2][batch_size * batch_count: batch_size * (batch_count + 1)]
            L = train_data[3][batch_size * batch_count: batch_size * (batch_count + 1)]
            if use_cuda:
                Q = Variable(torch.LongTensor(pad_batch_list(Q, max_len=q_len, padding_id=0)), requires_grad=False).cuda()
                D_pos = Variable(torch.LongTensor(pad_batch_list(D_pos, max_len=d_len, padding_id=0)), requires_grad=False).cuda()
                D_neg = Variable(torch.LongTensor(pad_batch_list(D_neg, max_len=d_len, padding_id=0)), requires_grad=False).cuda()
                L = Variable(torch.FloatTensor(L), requires_grad=False).cuda()
            else:
                Q = Variable(torch.LongTensor(pad_batch_list(Q, max_len=q_len, padding_id=0)), requires_grad=False)
                D_pos = Variable(torch.LongTensor(pad_batch_list(D_pos, max_len=d_len, padding_id=0)), requires_grad=False)
                D_neg = Variable(torch.LongTensor(pad_batch_list(D_neg, max_len=d_len, padding_id=0)), requires_grad=False)
                L = Variable(torch.FloatTensor(L), requires_grad=False)
            
            # run on this batch
            optimizer.zero_grad()
            t1 = time.time()
            q_mask, d_pos_mask, d_neg_mask = model.generate_mask(Q, D_pos, D_neg)
            S_pos, S_neg = model(Q, D_pos, D_neg, q_mask, d_pos_mask, d_neg_mask)
            Loss = hinge_loss(S_pos, S_neg, 1.0)
            Loss.backward()
            optimizer.step()
            t2 = time.time()
            batch_count_tr += 1
            print("epoch {} batch {} training cost: {} using {}s" \
            .format(epoch, batch_count_tr, Loss.data[0], t2-t1))
            f_log.write("epoch {} batch {} training cost: {}, using {}s".format(epoch, batch_count_tr, Loss.data[0], t2 - t1) + '\n')
        file_count += 1
        if file_count % 4 == 0:
            # do rapid validation
            ndcg_list, mapvalue = validation(model, model_name_str,
                                             data_base_path, model_base_path)  # pass the training compuational graph placeholder to valid function to evaluate with the same set of parameters
            print("epoch :{}, pkl count: {}, NDCG".format(epoch, file_count), ndcg_list)
            print("MAP: {}".format(mapvalue))
            # check if this valid period is the best and update best_MAP, save model to disk
            if mapvalue > best_MAP:
                best_MAP = mapvalue
                with open('{}/{}/saves/best_MAP.pkl'.format(model_base_path, model_name_str), 'wb') as f_MAP:
                    pickle.dump(best_MAP, f_MAP)
                # save model params after several epoch
                model_file = '{}/{}/saves/model_file'.format(model_base_path, model_name_str)
                torch.save(model.state_dict(), model_file)
                print("successfully saved model to the path {}".format(model_file))
            valid_log.write("{} {} {} {}".format(ndcg_list[1][0], ndcg_list[1][1], ndcg_list[1][2], ndcg_list[1][3]))
            valid_log.write(" MAP: {}".format(mapvalue))
            valid_log.write('\n')
    f_log.close()
    valid_log.close()