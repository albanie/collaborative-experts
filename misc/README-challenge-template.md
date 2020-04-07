## CVPR 2020 Challenge: The video pentathlon

This page contains information relating to the [CVPR penathlon challenge](https://www.robots.ox.ac.uk/~vgg/challenges/video-pentathlon/). Details about the format of the challenge can be found [here](https://www.robots.ox.ac.uk/~vgg/challenges/video-pentathlon/challenge.html).

**Obtaining the features:**

The challenge has two phases: a `challenge-release-1` phase (for which the validation set is named `public_server_val`) and a `challenge-release-2` for which the test set is named `public_server_test`).  The features for the development phase can be used to train models and upload predictions the CodaLab server. Downloading the features is done in two stages:

1. This codebase assumes that you have created a symlink with the name `data` in the root of the project (i.e. so after cloning you have a symlink at `collaborative-experts/data` which points to whichever place on your filesystem you would like to keep checkpoints, datasets, logfiles etc.). To create this symlink, run:

```
ln -s /folder/where/you/are/happy/to/put/data /path/to/collaborative-experts/data
```

2. Then, to download and extract the features, run:
```
python misc/sync_experts.py --release challenge-release-1
```

Thirty days before the close of the challenge, the features for the test set will be released (see the workshop [page](https://www.robots.ox.ac.uk/~vgg/challenges/video-pentathlon/challenge.html) for dates).


**Training a baseline:**

There are two ways to get started with training models: (i) train a model on a single dataset to get your feet wet; (ii) train a model for each dataset and combine the results into a zip file that can be uploaded directly to CodaLab for evaluation.  

*Training on a single dataset:*

Alternatively, if you are looking to get your feet wet with a simple example, you can train a single model for a single dataset as follows:

```
# add the project root folder to the pythonpath
export PYTHONPATH=$(pwd):$PYTHONPATH

# pick a GPU device id
gpu_id=0

# pick a dataset
dataset=msvd

python train.py --config configs/cvpr2020-challenge/$dataset/baseline-public-trainval.json --device $gpu_id
```

*Training a full baseline*: To get started with the challenge, we have provided a [train_baselines.py](misc/cvpr2020_challenge/train_baselines.py) script which will train a basic model for each dataset and store the predictions in the format required for uploading to the server (described below). After fetching the features using the step above, you can launch this script:

```
# add the project root folder to the pythonpath
export PYTHONPATH=$(pwd):$PYTHONPATH

# pick a GPU device id
gpu_id=0

python misc/cvpr2020_challenge/train_baselines.py --device $gpu_id
```

NOTES:
* It is possible to train models without a GPU, but it's quite slow.
* If you have access to a SLURM scheduler, adding the `--yaspify` flag will train each of the baselines in parallel (this may be helpful to save some time, but will require `pip install yaspi`).

**Uploading predictions to the server:**

The challenge is hosted on CodaLab (Dear friendly tester, please use this temporary challenge link: https://competitions.codalab.org/competitions/24264?secret_key=11b7bbfd-8623-4c82-b2a4-f8586008681b#results).  

**Trying out new ideas!**

There are undoubtedly many ways to improve upon the baseline. Here are just a few suggestions/ideas for possible directions:
* Using more modalities: there are many more preprocessed features available than the ones used in the baseline so experimenting with different combinations may produce good results.
* Using temporal information: As well as average/max pooled features, there are also features which have been sampled into segments (suitable for use with a TRN-like architecture). 
* Combining different datasets: currenlty, a model is trained for each dataset independently of the others
* Trying out new text encodings: currently the system uses either word2vec or OpenAI embeddings and a simple NetVLAD text encoder: there are likely many better ways to encode queries/descriptions
* Handling missing modalities: currently, missing modalities are simply zero-padded, but handling them in a "smart" way may help performance
* There are surely many other things we haven't thought of :)

