import argparse
import json
from scipy import stats
import numpy as np


def significance(first_file, second_file):
    with open(first_file) as first_json, open(second_file) as second_json:
        h1_data = json.load(first_json)
        h2_data = json.load(second_json)

        for key in h1_data:
            print("-----"+key+"-----")
            h1_d = np.asarray(h1_data[key])
            h2_d = np.asarray(h2_data[key])
            print("Mean h1", np.mean(h1_d))
            print("Mean h2", np.mean(h2_d))
            t, p = stats.ttest_rel(h1_d, h2_d)

            print("t-value", '{0:1.6f}'.format(t))
            print("p-value", '{0:1.6f}'.format(p))
            print("p<0.05: ", p < 0.05)
            print("p<0.01: ", p < 0.01)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RecPlay')
    parser.add_argument('first_result_file', type=str)
    parser.add_argument('second_result_file', type=str)

    args = parser.parse_args()
    significance(args.first_result_file, args.second_result_file)
