import pickle
import glob, os, sys
import argparse
from lib.torch_utils import Dot, Gaussian_ker, GenDotM, cossim, cossim1, MaxM_fromBatch


def load_dataset(path=None):
    '''load the train and test datasets'''
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding="latin1")
    return data

def dump_dataset(data, path=None):
    with open(path, 'wb') as f:
        pickle.dump(data, f, protocol=2)

def find_exact_match_idx(q, doc, OOV=1):
    """
       q = [5, 6, 7, 8]
       doc = [7, 7, 88, 9, 102]
       OOV: OOV idx to exclude
       return [0, 1]
       return: the document exact match index
    """
    q = set(q) - set([OOV])
    idx = [(i, item) for i, item in enumerate(doc) if item in q]
    q_id = [i[1] for i in idx]
    idx = [i[0] for i in idx]
    return idx, q_id


def gen_range_lists(exact_match_idx, window_size):
    """ generate range lists for each exact match centric word
        return: range lists
    """
    range_tuples = []
    for i in exact_match_idx:
        r = []
        if i - window_size < 0:
            r.append(0)
            r.append(2*window_size)
        else:
            r.append(i - window_size)
            r.append(i + window_size)
        range_tuples.append(r)
    return range_tuples

def merge_range_lists(range_lists):
    """ merge range lists if they overlaps
    """
    sorted_by_lower_bound = sorted(range_lists, key=lambda tup: tup[0])
    merged = []
    for higher in sorted_by_lower_bound:
        if not merged:
            merged.append(higher)
        else:
            lower = merged[-1]
            # test for intersection between lower and higher:
            # we know via sorting that lower[0] <= higher[0]
            if higher[0] <= lower[1]:
                upper_bound = max(lower[1], higher[1])
                merged[-1] = (lower[0], upper_bound)  # replace by merged interval
            else:
                merged.append(higher)
    return merged

def gen_range_lists(exact_match_idx, window_size):
    """ generate range lists for each exact match centric word
        return: range lists
    """
    range_tuples = []
    for i in exact_match_idx:
        r = []
        if i - window_size < 0:
            r.append(0)
            r.append(2*window_size)
        else:
            r.append(i - window_size)
            r.append(i + window_size)
        range_tuples.append(r)
    return range_tuples

def get_exact_match_window_data(data, window_size):
    """ 
        find exact match idx
        data: data pkl = {topic_num: {'query': ..., 'docs': [[4,6,7], [2,5,6,8,88]...]
                                      'scores': [25.6, 27.1, ...]}}
    """
    new_data = {}
    topic_counter = 0
    for topic_num in data.keys():
        q = data[topic_num]['query']
        docs = data[topic_num]['docs']
        scores = data[topic_num]['scores']
        new_docs = []
        new_scores = []
        new_qlist = []
        for i in range(len(docs)):
            new_doc = []
            new_score = []
            idx, q_id = find_exact_match_idx(q, docs[i], OOV=1)
            if len(idx) == 0:
                continue
            ranges = gen_range_lists(idx, window_size)
            for r in ranges:
                window = docs[i][r[0]: r[1] + 1]
                if len(window) < 11:
                    window += [0]*(11-len(window))
                new_doc.append(window) #[[.....], [.....], ... ]
                new_score.append(scores[i]) # [1,1,1,1...]
            new_docs.append(new_doc) # [ [[1,2,3,4..], [1,23,4,3,..]], [[1,2,3,4..], [1,23,4,3,..]]... ]
            new_scores.append(new_score)# [[1,1,1,1,1], [2,2,2,2,2]]
            new_qlist.append(q_id) # [[1,2,2,1,1], [1,21,2,2,3,2]]
        new_data[topic_num] = {'query': q, 'q_list': new_qlist, 'docs': new_docs, 'scores': new_scores}
        topic_counter += 1
    return new_data

def get_exact_match_window_testdata(data, window_size):
    """ 
    find exact match idx
    valid_data: data pkl = {topic_num: {'query': ..., 'docs': [[4,6,7], [2,5,6,8,88]...]
                                      'docno': [1, 2, 3, 4...]}}
    """
    new_data = {}
    topic_counter = 0
    for topic_num in data.keys():
        q = data[topic_num]['query']
        docs = data[topic_num]['docs']
        docno = data[topic_num]['docno']
        new_docs = []
        new_qlist = []
        new_docno = []
        for i in range(len(docs)):
            new_doc = []
            idx, q_id = find_exact_match_idx(q, docs[i], OOV=1)
            if len(idx) == 0 and len(docs[i]) >= 11:
                num = len(docs[i]) // 11
                for j in range(min(num, 50)):
                    new_doc.append(docs[i][11 * j: 11 * (j + 1)])
            elif len(idx) == 0 and len(docs[i]) < 11:
                window = docs[i][0:11]
                window += [0]*(11-len(window))
                assert len(window) == 11
                new_doc.append(window)
            else:
                ranges = gen_range_lists(idx, window_size)
                for r in ranges:
                    window = docs[i][r[0]: r[1] + 1]
                    if len(window) < 11:
                        window += [0]*(11-len(window))
                    new_doc.append(window)
            new_docno.append(docno[i])
            new_docs.append(new_doc)
        new_data[topic_num] = {'query': q, 'docs': new_docs, 'docno': new_docno}
        topic_counter += 1
    return new_data

def make_data(input_base_path, output_base_path, window_size):
    f_list = glob.glob("{}/data*.pkl".format(input_base_path))
    for f_path in f_list:
        id_str = f_path.split('/')[-1]
        data = load_dataset(f_path)
        new_data = get_exact_match_window_data(data, window_size)
        output_path = output_base_path + '/' + id_str
        dump_dataset(new_data, output_path)
        print("successfully dumped {}".format(output_path))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str)
    parser.add_argument('--outpath', type=str)
    parser.add_argument('--window', type=int)
    args = parser.parse_args()
    make_data(args.inpath, args.outpath, args.window)

if __name__ == '__main__':
    main()
