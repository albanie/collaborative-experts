<p align="center">
<img width="300" alt="logo" src="../figs/logo-centre.png">
</p>

## CVPR 2020 Challenge: The video pentathlon

This page contains information relating to the [CVPR penathlon challenge](https://www.robots.ox.ac.uk/~vgg/challenges/video-pentathlon/). Details about the format of the challenge can be found [here](https://www.robots.ox.ac.uk/~vgg/challenges/video-pentathlon/challenge.html).

### Obtaining the features

The challenge has two phases: a `challenge-release-1` phase (for which the validation set is named `public_server_val`) and a `challenge-release-2` for which the test set is named `public_server_test`).  The features for the development phase can be used to train models and upload predictions the CodaLab server. Downloading the features is done in two stages:

1. This codebase assumes that you have created a symlink with the name `data` in the root of the project (i.e. so after cloning you have a symlink at `collaborative-experts/data` which points to whichever place on your filesystem you would like to keep checkpoints, datasets, logfiles etc.). To create this symlink, run:

```
ln -s /folder/where/you/are/happy/to/put/data /path/to/collaborative-experts/data
```

2. Then, to download and extract the features, run:
```
python misc/sync_experts.py --release challenge-release-1
```

3. To download data for the test-phase, run:
```
python misc/sync_experts.py --release challenge-release-2
```


### Training a baseline

There are two ways to get started with training models: (i) train a model on a single dataset to get your feet wet; (ii) train a model for each dataset and combine the results into a zip file that can be uploaded directly to CodaLab for evaluation.  

*Training on a single dataset:*

If you are looking to get started with a simple example, you can train a single model for a single dataset as follows:

```
# add the project root folder to the pythonpath
export PYTHONPATH=$(pwd):$PYTHONPATH

# pick a GPU device id
gpu_id=0

# pick a dataset
dataset=msvd

python train.py --config configs/cvpr2020-challenge/$dataset/baseline-public-trainval.json --device $gpu_id
```

*Training a full baseline*: To get started with the challenge, we have provided a [train_baselines.py](cvpr2020_challenge/train_baselines.py) script which will train a basic model for each dataset and store the predictions in the format required for uploading to the server (described below). After fetching the features using the step above, you can launch this script:

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

### Test phase

*Before submitting, please note that you are limited to a total of three test submissions over the course of the challenge*.

The test phase of the competition is now open (10th May 2020).  To test the baseline models described above, you can pass the json file of checkpoints produced by the `train_baselines.py` script to the [test_baselines.py](cvpr2020_challenge/test_baselines.py) :

```
# add the project root folder to the pythonpath
export PYTHONPATH=$(pwd):$PYTHONPATH

# provide path to checkpoints (you should replace this with the output produced by train_baselines.py)
CKPT_LIST=data/cvpr2020-challenge-submissions/ckpts-baselines-2020-04-05_09-53-14-public_server_val-MSRVTT-MSVD-DiDeMo-YouCook2-activity-net.json

python misc/cvpr2020_challenge/test_baselines.py  --ckpt_list_path ${CKPT_LIST}
```

This will produce a file in the test format required by the CodaLab server.


### Uploading predictions to the server

The challenge is hosted on CodaLab: https://competitions.codalab.org/competitions/24292

### Using multiple captions per video

Two of the datasets: `MSRVTT` and `MSVD` provide multiple captions per video (this can be seen by examining the raw captions for each dataset, or the collection of text features which are grouped by video).  This information is not exploited by the baseline code, but it is valid to use this information under the rules of the challenge (e.g. by ensembling predictions for each of the captions assigned to a signle video) and is expected to lead to a boost in performance on these two datasets.

### Trying out new ideas!

There are undoubtedly many ways to improve upon the baseline. Here are just a few suggestions/ideas for possible directions:
* Using more modalities: there are many more preprocessed features available than the ones used in the baseline so experimenting with different combinations may produce good results.
* Using temporal information: As well as average/max pooled features, there are also features which have been sampled into segments (suitable for use with a TRN-like architecture). 
* Combining different datasets: currenlty, a model is trained for each dataset independently of the others
* Trying out new text encodings: currently the system uses either word2vec or OpenAI embeddings and a simple NetVLAD text encoder: there are likely many better ways to encode queries/descriptions
* Handling missing modalities: currently, missing modalities are simply zero-padded, but handling them in a "smart" way may help performance
* There are surely many other things we haven't thought of :)

