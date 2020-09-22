from baselines.popular import ndcg_at_k, store_results_json
from datasets import dataset_factory
from options import args
from collections import Counter
import numpy as np


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
    val = dataset['val']
    test = dataset['test']
    smaps = dataset['smap']

    c = Counter()
    for i in train:
        c.update(train[i][0])

    results = {}

    num_examples = 0
    hits_at_k = 0
    ranks_at_k = 0
    ndcg_sum = 0
    for i in range(len(train)):
        num_examples += 1
        last_item = val[i][0][0]
        answer = test[i][0][0]
        binary_gain = np.zeros(1)
        if answer == last_item:
            hits_at_k += 1
            ranks_at_k += 1
            binary_gain[0] = 1
            ndcg_sum += ndcg_at_k(binary_gain, 1)
        else:
            ndcg_sum += ndcg_at_k(binary_gain, 1)
    results["Recall@1"] = hits_at_k / num_examples
    results["MRR@1"] = ranks_at_k / num_examples
    results["NDCG@1"] = ndcg_sum / num_examples

    print(results)
    store_results_json(args, results)
