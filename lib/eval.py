'''
Basic functions to compute IR metrics with standard IR tools.
IR eval tools such as:
    gdeval.pl, trec_eval
    should be on the PATH
'''

import os


def write_run(run_list, out_path):
    '''
    Params:
        run_list: list of tuples representing this run (qry_no, doc_no, score)
        out_path: output file path (output will be in trecFormat)
    '''
    out_fmt = "{} Q0 {} {} {:.6f} mod"  # trecFormat output template
    run_dict = {}
    for (qry_no, doc_no, score) in run_list:
        if qry_no not in run_dict:
            run_dict[qry_no] = []
        run_dict[qry_no].append((score, doc_no))

    # write results
    res_strs = []
    for (qry_no, res_list) in run_dict.items():
        res_list = sorted(res_list, reverse=True)
        for rnk, (score, doc_no) in enumerate(res_list):
            res_strs.append(out_fmt.format(qry_no, doc_no, rnk + 1, score))

    out_file = open(out_path, "w")
    out_file.write("\n".join(res_strs))
    out_file.write("\n")  # add newline at the end of last line
    out_file.close()


def compute_ndcg(run_path, rel_path, tmp_path):
    '''
    Params:
        run_path: run file (trecFormat) of the model.
        rel_path: relevance file (to compute the judgments).
    '''
    K = [1, 3, 10, 20]
    ndcgs = []
    for k in K:
        os.system("gdeval.pl -k {} {} {} > {}".format(k, rel_path, run_path, tmp_path))
        with open(tmp_path) as rf:
            lines = rf.readlines()
            means = lines[-1]         # indri,amean,0.14550,0.05194
            means = means.split(',')
            ndcgs.append(float(means[2]))
    return (K, ndcgs)

def compute_map(run_path, rel_path, tmp_path):
    '''
    params:
    run_path: runfile path (trecFormat)
    rel_path: relevance file
    tmp_path: generated temp result file
    '''
    os.system("trec_eval {} {} > {}".format(rel_path, run_path, tmp_path))
    with open(tmp_path) as rf:
        lines = rf.readlines()
        mapline = lines[5]         # map all 0.0213
        mapvaluelist = mapline.split()
        mapvalue = float(mapvaluelist[2])
    return mapvalue
