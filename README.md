# The Impact of Differential Privacy on Group Disparity Mitigation
This repository includes the code used to reproduce the experiments for the paper "The Impact of Differential Privacy on Group Disparity Mitigation".

It directly extends the [WILDS repository](https://github.com/p-lambda/wilds) with functionality for training models with differential privacy using the [Opacus framework](https://opacus.ai/).

## Datasets
All the datasets should be placed in the `data/` folder. If it doesn't exists, simply create it with the following command from the root of the repository:

```bash
mkdir data
```

In addition to the built-in dataset that WILDS offers we also run experiments on the following datasets:

### Blog Authorship Corpus

The Blog Authorship Corpus can be downloaded from [kaggle](https://www.kaggle.com/rtatman/blog-authorship-corpus).

To preprocess it the same way as we do for our experiments, download the dataset, extract it to `data/blog-authorship/` and run the preprocessing command:
```bash
python data_preprocessing/blog_author_preprocess.py --path data/blog-authorship/blogtext.csv
```

We include a compressed file with our processed data already in the `data/` folder.

### Trustpilot Corpus

The full processed dataset (all languages) can be download [here](https://drive.google.com/file/d/1_BEZQXp38BeiuLMJxKydTlrOj7Ne9gP8/view?usp=sharing).

Then put the decompressed folder to the data folder and run the corresponding preprocessing command.

### CelebA

The CelebA is already included in the WILDS repository, if you haven't downloaded the data yet, remember to include the `--download` flag the first time you run the experiments.


## Installation

In a virtual Python3 (version >=3.6) environment install [wilds](https://github.com/p-lambda/wilds), preferably using `pip`:

```bash
pip install wilds==1.1.0
```

Additionally, the code required the following packages:

```
torchvision>=0.8.1
transformers>=4.3.3
torch-scatter>=2.0.5
torch-geometric>=1.6.1
```

Make sure that you correctly install the `torch-scatter` and `torch-geometric` packages. See [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## Configuration

The configurations for the default parameters in the dataset are found in `configs/datasets.py`

Note that `configs/supported.py` and `configs/model.py` also have corresponding modifacation compared to original code.


## Run Experiments

You can use the following command to run a single configuration of the experiments (blog corpus, ERM, low DP):

```bash
sh scripts/blog/run_blog_erm_dp_low.sh
```

To run all experiments, execute the `run_all.sh` script.

The numbers we report in our experiments were run with `SEED` values of 0, 1 and 2, make sure to change these manually (for the time being) in each run script.
