import csv
import argparse
import json

import pandas as pd

from sklearn.model_selection import train_test_split
from collections import defaultdict
from pprint import pprint


def clean(text):
    text = text.replace('urlLink', '')
    text = text.replace('&nbsp;', '')
    text = text.strip()
    return text


def to_jsonl(lines, path):
    with open(path, 'w') as out_file:
        for line in lines:
            out_file.write(json.dumps(line) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    args = parser.parse_args()

    # Seed for splits
    seed = 1234

    # Read data file
    df = pd.read_csv(args.path)

    # Filter unknown topics and basic preprocessing
    df = df[df['topic'] != 'indUnk']
    df['text'] = df['text'].apply(lambda x: clean(x))
    df = df[['text', 'age', 'sign', 'gender', 'topic']]

    # Gender statistics
    counts = df.groupby(['gender', 'topic']).size().reset_index().rename(columns={0: 'count'})
    counts = counts.to_dict('records')

    gender_dist = defaultdict(dict)
    for c in counts:
        gender_dist[c['topic']][c['gender']] = c['count']

    for topic, dist in gender_dist.items():
        n_fem = dist['female']
        n_mal = dist['male']
        gender_dist[topic]['total'] = n_mal + n_fem
        gender_dist[topic]['m/f ratio'] = n_mal / n_fem

    #pprint(gender_dist)
    #import pdb; pdb.set_trace()

    # Topics with heavy gender imbalance
    #topics = ['Technology', 'Chemicals', 'Museums-Libraries', 'Marketing']
    topics = ['Technology', 'Arts']
    df = df[df['topic'].isin(topics)]

    # Generate splits
    test_frac = 0.2
    train, test = train_test_split(df, test_size=test_frac, random_state=seed)

    dev_frac = 0.25  # Same size as test_frac when splitting train
    train, dev = train_test_split(train, test_size=dev_frac, random_state=seed)

    # Save splits as jsonl
    train, dev, test = train.to_dict('records'), dev.to_dict('records'), test.to_dict('records')
    base_path = '/'.join(args.path.split('/')[:-1])

    #import pdb; pdb.set_trace()
    to_jsonl(train, base_path + '/blog_train.jsonl')
    to_jsonl(dev, base_path + '/blog_dev.jsonl')
    to_jsonl(test, base_path + '/blog_test.jsonl')
