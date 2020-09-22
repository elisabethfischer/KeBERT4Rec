import json
import os

from datasets import dataset_factory
from options import args
from collections import Counter
import numpy as np

from utils import create_experiment_export_folder


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


def store_results_json(args, results):
    export_root = create_experiment_export_folder(args)
    logs_dir = os.path.join(export_root, 'logs')

    if not os.path.exists(logs_dir):
        os.mkdir(logs_dir)
    with open(os.path.join(logs_dir, 'test_metrics.json'), 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    args.mode = "train"
    dataset_code = args.dataset_code
    args.dataset_code = dataset_code if dataset_code else input('ml-1m, ml-20m: ')
    args.min_rating = 0 if args.dataset_code == 'ml-1m' else 4
    args.min_uc = 5
    args.min_sc = 0
    args.split = 'leave_one_out'

    dataset = dataset_factory(args)
    dataset._get_preprocessed_folder_path()
    dataset = dataset.load_dataset()
    train = dataset['train']
    # val = dataset['val'] # ignore valid
    test = dataset['test']
    smaps = dataset['smap']

    c = Counter()
    for i in train:
        c.update(train[i][0])

    results = {}
    ks = [1, 5, 10, 20, 50, 100]
    for k in ks:
        print('working on k={}'.format(k))
        common_k = [x[0] for x in c.most_common(k)]
        num_examples = 0
        hits_at_k = 0
        ranks_at_k = 0
        ndcg_sum = 0
        for i in range(len(train)):
            num_examples += 1
            seq = train[i][0]
            answer = test[i][0][0]
            binary_gain = np.zeros(len(common_k))
            try:
                value_index = common_k.index(answer)
                hits_at_k += 1
                ranks_at_k += 1 / (value_index + 1)
                binary_gain[value_index] = 1
                ndcg_sum += ndcg_at_k(binary_gain, k)
            except:
                value_index = -1
                ndcg_sum += ndcg_at_k(binary_gain, k)

        results["Recall@{}".format(k)] = hits_at_k / num_examples
        results["MRR@{}".format(k)] = ranks_at_k / num_examples
        results["NDCG@{}".format(k)] = ndcg_sum / num_examples

    print(results)
    store_results_json(args, results)
