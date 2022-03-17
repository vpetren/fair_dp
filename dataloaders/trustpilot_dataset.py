import os
import json
import torch
import pandas as pd
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy
#from utils import F1
from sklearn.metrics import f1_score
from wilds.common.metrics.all_metrics import F1

class TrustPilotDataset(WILDSDataset):
    """
    The CivilComments-wilds toxicity classification dataset.
    This is a modified version of the original CivilComments dataset.

    Supported `split_scheme`:
        'denmark', 'germany', 'france', 'us', 'uk'

    Input (x):
        A comment on an online article, comprising one or more sentences of text.

    Label (y):
        y is binary. It is 1 if the comment was been rated as toxic by a majority of the crowdworkers who saw that comment, and 0 otherwise.


    Website:
        https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification

    Original publication:
        @inproceedings{borkan2019nuanced,
          title={Nuanced metrics for measuring unintended bias with real data for text classification},
          author={Borkan, Daniel and Dixon, Lucas and Sorensen, Jeffrey and Thain, Nithum and Vasserman, Lucy},
          booktitle={Companion Proceedings of The 2019 World Wide Web Conference},
          pages={491--500},
          year={2019}
        }

    License:
        This dataset is in the public domain and is distributed under CC0.
        https://creativecommons.org/publicdomain/zero/1.0/
    """

    def __init__(self, root_dir='data', download=False, split_scheme='germany', logger=None, binary=True):
        self._dataset_name = 'trustpilot'
        self._version = '1.0'

        # Use binary ratings
        self.binary = binary

        if "aug" in split_scheme:
            self._data_dir = os.path.join(root_dir, 'trustpilot_aug')
        else:
            self._data_dir = os.path.join(root_dir, 'trustpilot')

        # Extract splits
        self._split_scheme = split_scheme

        self._metadata_df = self.read_data(root_path=self._data_dir, language=split_scheme)
        print(self._metadata_df.head())
        self.logger = logger


        # Get the y values
        self._y_array = torch.LongTensor(self._metadata_df['rating'])
        self._y_size = 1
        self._n_classes = 2 if self.binary else 5

        # Extract text
        self._text_array = list(self._metadata_df['text'])

        # DEFAULT_SPLITS = {'train': 0, 'val': 1, 'test': 2}
        self._split_dict = {'train': 0, 'val': 1, 'test': 2}
        self._split_names = {'train': 'train', 'val': 'dev', 'test': "test"}
        #cnt = 2
        #for mode in ['dev', 'test']:
        #    for lang in ['denmark', 'germany', 'france', 'us', 'uk']:
        #        cnt += 1
        #        self._split_dict["%s_%s"%(mode, lang)] = cnt
        #        self._split_names["%s_%s"%(mode, lang)] = "%s_%s"%(mode, lang)

        for split in self.split_dict:
            split_indices = self._metadata_df['data_type'] == split
            self._metadata_df.loc[split_indices, 'data_type'] = self.split_dict[split]
        self._split_array = self._metadata_df['data_type'].values

        # Extract metadata
        self._identity_vars = [
            'male',
            'female',
            'young',
            'elder',
            'M-Y',
            'M-E',
            'F-Y',
            'F-E'
        ]

        self._metadata_array = torch.cat(
            (
                torch.LongTensor(self._metadata_df.loc[:, self._identity_vars].values),
                self._y_array.reshape((-1, 1))
            ),
            dim=1
        )
        self._metadata_fields = self._identity_vars + ['y']

        self._eval_groupers = [
            CombinatorialGrouper(
                dataset=self,
                groupby_fields=['male', 'young'])
            ]

        # self._eval_groupers = [
        #     CombinatorialGrouper(
        #         dataset=self,
        #         groupby_fields=[identity_var, 'y'])
        #     for identity_var in self._identity_vars[-4:]]

        self._metric = F1(average='macro')

        super().__init__(root_dir, download, split_scheme)

    def read_jsonl(self, filename, data_type=None):
        data = []
        gender_dict = {'M':0, "F":1}
        generation_dict = {"Y":0, "E":1}
        with open(filename) as fh:
            for line in fh:
                #import pdb; pdb.set_trace()
                example = json.loads(line)
                rating = int(example['rating'])
                if rating == 3 and self.binary:
                    continue
                elif self.binary:
                    example['rating'] = 1 if rating > 3 else 0
                else:
                    example['rating'] = int(rating) - 1
                example['gender'] = gender_dict[example['group'][0]]
                example['generation'] = generation_dict[example['group'][2]]
                example['male'] = 1 if "M" in example['group'] else 0
                example['female'] = 1 if "F" in example['group'] else 0
                example['young'] = 1 if "Y" in example['group'] else 0
                example['elder'] = 1 if "E" in example['group'] else 0

                for name in ['M-Y', 'M-E', 'F-Y', 'F-E']:
                    if example['group'] == name:
                        example[name] = 1
                    else:
                        example[name] = 0

                if example['title'] is not None and len(example['title']) > 0:
                    example['text'] = example['title'] + ' [SEP] ' + example['text']
                if data_type is not None:
                    example['data_type'] = data_type
                data.append(example)
        return data

    def read_data(self, root_path, language='germany'):
        #data_dirs = ['denmark', 'germany', 'france', 'us', 'uk']
        if "trustpilot" not in root_path:
            if "aug" in language:
                root_path = os.path.join(root_path, "trustpilot_aug")
            else:
                root_path = os.path.join(root_path, "trustpilot")
        train_path = os.path.join(root_path, language, "train.json")
        train_data = self.read_jsonl(train_path, data_type="train")
        data = train_data
        dev_path = os.path.join(root_path, language, 'dev.json')
        test_path = os.path.join(root_path, language, 'test.json')
        dev_data = self.read_jsonl(dev_path, data_type="val")
        test_data = self.read_jsonl(test_path, data_type="test")
        data += dev_data
        data += test_data
        """
        for data_dir in data_dirs:
            new_data_dir = data_dir
            if "aug" in language:
                new_data_dir = data_dir + "_aug"
            dev_path = os.path.join(root_path, new_data_dir, 'dev.json')
            test_path = os.path.join(root_path, new_data_dir, 'test.json')
            dev_data = self.read_jsonl(dev_path, data_type="dev_%s"%data_dir)
            test_data = self.read_jsonl(test_path, data_type="test_%s"%data_dir)
            data += dev_data
            data += test_data
        """
        # print(data[0])
        # print(data[-1])
        df = pd.DataFrame(data)
        df = df.fillna("")
        return df

    def get_input(self, idx):
        return self._text_array[idx]


    def eval(self, y_pred, y_true, metadata):
        #y_pred = torch.argmax(y_pred, dim=1)
        y_pred = torch.argmax(y_pred, dim=1)
        metric = F1(average='macro')
        eval_grouper = self._eval_groupers[0]

        return self.standard_group_eval(
            metric,
            eval_grouper,
            y_pred, y_true, metadata)


    """
    def eval(self, y_pred, y_true, metadata):
        results = {
            **self._metric.compute(y_pred, y_true),
        }

        # print()
        # print(y_pred.size())
        # print(y_true.size())
        # print(metadata.size())

        new_y_pred = y_pred.cpu().detach()
        new_y_true = y_true.cpu().detach()
        new_y_pred = torch.argmax(new_y_pred, dim=-1)
        # print(new_y_pred)
        acc = torch.mean((new_y_pred == new_y_true).float())
        f1 = f1_score(new_y_true, new_y_pred, average='macro')
        # print(acc)
        # print(f1)

        results_str = f"TAverage {self._metric.name}: {results[self._metric.agg_metric_field]:.4f}\n"
        # Each eval_grouper is over label + a single identity
        # We only want to keep the groups where the identity is positive
        # The groups are:
        #   Group 0: identity = 0, y = 0
        #   Group 1: identity = 1, y = 0
        #   Group 2: identity = 0, y = 1
        #   Group 3: identity = 1, y = 1
        # so this means we want only groups 1 and 3.
        worst_group_metric = None
        for identity_var, eval_grouper in zip(self._identity_vars, self._eval_groupers):
            g = eval_grouper.metadata_to_group(metadata)
            group_results = {
                **self._metric.compute_group_wise(y_pred, y_true, g, eval_grouper.n_groups)
            }
            # print(identity_var)
            # print(group_results)

            results_str += f"{identity_var:8s}, #group-{eval_grouper.n_groups}"
            results_str += f"\n{group_results}\n"
            results.update(group_results)
            # for group_idx in range(eval_grouper.n_groups):
            #     group_str = eval_grouper.group_field_str(group_idx)
            #     print(group_str)
            #     if f'{identity_var}:1' in group_str:
            #         group_metric = group_results[self._metric.group_metric_field(group_idx)]
            #         group_counts = group_results[self._metric.group_count_field(group_idx)]
            #         results[f'{self._metric.name}_{group_str}'] = group_metric
            #         results[f'count_{group_str}'] = group_counts
            #         # if f'y:0' in group_str:
            #         #     label_str = 'non_toxic'
            #         # else:
            #         #     label_str = 'toxic'
            #         print(group_str)
            #         label_str = '*' * (int(group_str.split('_')[1][-1])+1)
            #         results_str += (
            #             f"   {label_str}: {group_metric:.4f}"
            #             f" (n={results[f'count_{group_str}']:.0f}) "
            #         )
            #         if worst_group_metric is None:
            #             worst_group_metric = group_metric
            #         else:
            #             worst_group_metric = self._metric.worst(
            #                 [worst_group_metric, group_metric])
            results_str += f"\n"
        # results[f'Worst case: {self._metric.worst_group_metric_field}'] = worst_group_metric
        # results_str += f"Worst-group {self._metric.name}: {worst_group_metric:.4f}\n"

        return results, results_str
    """
