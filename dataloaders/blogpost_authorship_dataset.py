import os
import json
import torch
import pandas as pd
import numpy as np
from wilds.datasets.wilds_dataset import WILDSDataset
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.all_metrics import Accuracy
from wilds.common.metrics.all_metrics import F1
#from utils import F1
from sklearn.metrics import f1_score

class BlogAuthorDataset(WILDSDataset):
    """
    Supported `split_scheme`:


    Input (x):
        A blog test consisting of informal diary-style text entries.

    Label (y):
        y is binary. It is 1 if the comment was been rated as toxic by a majority of the crowdworkers who saw that comment, and 0 otherwise.


    Website:
        https://www.kaggle.com/rtatman/blog-authorship-corpus

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

    def __init__(self, root_dir='data', download=False, split_scheme=None, logger=None):
        self._dataset_name = 'blog_author'
        self._version = '1.0'
        self._data_dir = os.path.join(root_dir, 'blog-authorship')
        self.logger = logger

        # Read in metadata
        self._metadata_df = self.read_data(self._data_dir)

        # Get the y values
        self._y_array = torch.LongTensor(self._metadata_df['topic'])
        self._y_size = 1
        self._n_classes = 4

        # Extract text
        self._text_array = list(self._metadata_df['text'])

        # Extract splits
        self._split_scheme = split_scheme

        # DEFAULT_SPLITS = {'train': 0, 'val': 1, 'test': 2}
        # DEFAULT_SPLIT_NAMES = {'train': 'Train', 'val': 'Validation', 'test': 'Test'}
        # self._split_dict = {'train': 0, 'val': 1, 'id_val': 2, 'test': 3, 'id_test': 4}
        # self._split_names = {'train': 'Train', 'val': 'Validation (OOD)', 'id_val': 'Validation (ID)', 'test':'Test (OOD)', 'id_test': 'Test (ID)'}
        self._split_dict = {'train': 0, 'val': 1, 'test': 2}
        self._split_names = {'train': 'train', 'val': 'dev', 'test': 'test'}

        for split in self.split_dict:
            split_indices = self._metadata_df['data_type'] == split
            self._metadata_df.loc[split_indices, 'data_type'] = self.split_dict[split]
        self._split_array = self._metadata_df['data_type'].values

        # Extract metadata
        self._identity_vars = [
            'gender',
            'age',
            'sign'
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
                groupby_fields=[identity_var, 'y'])
            for identity_var in self._identity_vars
        ]
        self._eval_grouper = CombinatorialGrouper(
                dataset=self,
                groupby_fields=['gender', 'age'])

        self._metric = F1(average='macro')

        super().__init__(root_dir, download, split_scheme)

    def read_jsonl(self, filename, data_type):
        #topic_dict = {'Technology': 0, 'Chemicals': 1, 'Museums-Libraries': 2, 'Marketing': 3}
        topic_dict = {'Technology': 0, 'Arts': 1}
        sign_dict = {
            'Leo': 0, 'Aquarius': 1, 'Gemini': 2, 'Aries': 3, 'Cancer': 4, 'Libra': 5,
            'Sagittarius': 6, 'Capricorn': 7, 'Scorpio': 8, 'Pisces': 9, 'Virgo': 10, 'Taurus': 11
        }
        gender_dict = {'male': 0, 'female': 1}

        data = []
        with open(filename) as fh:
            for line in fh:
                example = json.loads(line)
                example['topic'] = topic_dict[example['topic']]
                example['gender'] = gender_dict[example['gender']]
                example['sign'] = sign_dict[example['sign']]
                example['age'] = 1 if example['age'] > 35 else 0
                example['data_type'] = data_type
                data.append(example)
        return data

    def read_data(self, root_path):
        train_path = os.path.join(root_path, "blog_train.jsonl")
        train_data = self.read_jsonl(train_path, data_type="train")

        data = train_data
        dev_path = os.path.join(root_path, 'blog_dev.jsonl')
        test_path = os.path.join(root_path, 'blog_test.jsonl')
        dev_data = self.read_jsonl(dev_path, data_type="val")
        test_data = self.read_jsonl(test_path, data_type="test")
        data += dev_data
        data += test_data

        df = pd.DataFrame(data)
        df = df.fillna("")
        return df

    def get_input(self, idx):
        return self._text_array[idx]

    def eval(self, y_pred, y_true, metadata, prediction_fn=None):
        #y_pred = torch.argmax(y_pred, dim=1)
        #metric = Accuracy(prediction_fn=prediction_fn)
        y_pred = torch.argmax(y_pred, dim=1)
        metric = F1(prediction_fn=prediction_fn, average='macro')
        #metric = self._metric
        eval_grouper = self._eval_grouper
        #eval_grouper = self._eval_groupers[0]
        return self.standard_group_eval(
            metric,
            eval_grouper,
            y_pred, y_true, metadata)

        """
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
        # print(f1)

        results_str = f"Average {self._metric.name}: {results[self._metric.agg_metric_field]:.4f}\n"
        worst_group_metric = None
        for identity_var, eval_grouper in zip(self._identity_vars, self._eval_groupers):
            g = eval_grouper.metadata_to_group(metadata)
            group_results = {
                **self._metric.compute_group_wise(y_pred, y_true, g, eval_grouper.n_groups)
            }

            results_str += f"{identity_var:8s}, #group-{eval_grouper.n_groups}"
            results_str += f"\n{group_results}\n"
            results.update(group_results)

            for group_idx in range(eval_grouper.n_groups):
                group_str = eval_grouper.group_field_str(group_idx)
                group_metric = group_results[self._metric.group_metric_field(group_idx)]
                group_counts = group_results[self._metric.group_count_field(group_idx)]
                results[f'{self._metric.name}_{group_str}'] = group_metric
                results[f'count_{group_str}'] = group_counts
                #import pdb; pdb.set_trace()
                label_str = int(group_str.split('_')[0][-1])
                results_str += (
                    f"   {label_str}: {group_metric:.4f}"
                    f" (n={results[f'count_{group_str}']:.0f}) "
                )
                worst_group_metric = group_metric
                #worst_group_metric = self._metric.worst([worst_group_metric, group_metric])
            results_str += "\n"
        results[f'Worst case: {self._metric.worst_group_metric_field}'] = worst_group_metric
        results_str += f"Worst-group {self._metric.name}: {worst_group_metric:.4f}\n"
        #import pdb; pdb.set_trace()
        return results, results_str
        """