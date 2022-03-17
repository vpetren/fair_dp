import csv
import argparse
import json

import pandas as pd

from sklearn.model_selection import train_test_split
from collections import defaultdict
from pprint import pprint


def to_jsonl(lines, path):
    with open(path, 'w') as out_file:
        for line in lines:
            out_file.write(json.dumps(line) + '\n')


def read_jsonl(path):
    lines = [json.loads(line) for line in open(path)]
    return lines


def binarize(data):
    updated = []
    for line in data:
        if line['rating'] == 3:
            continue
        line['rating'] = 1 if line['rating'] > 3 else 0
        updated.append(line)
    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    train = read_jsonl(args.path + '/train.json')
    dev = read_jsonl(args.path + '/dev.json')
    test = read_jsonl(args.path + '/test.json')

    train = binarize(train)
    dev = binarize(dev)
    test = binarize(test)

    #import pdb; pdb.set_trace()
    to_jsonl(train, args.path + '/train_bin.jsonl')
    to_jsonl(dev, args.path + '/dev_bin.jsonl')
    to_jsonl(test, args.path + '/test_bin.jsonl')
