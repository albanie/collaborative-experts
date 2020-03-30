### CVPR 2020 Challenge: The video pentathlon

This page contains information relating to the [CVPR penathlon challenge](https://www.robots.ox.ac.uk/~vgg/challenges/video-pentathlon/). Details about the format of the challenge can be found [here](https://www.robots.ox.ac.uk/~vgg/challenges/video-pentathlon/challenge.html).

**Obtaining the features:**

**Training a baseline:**

To get started with the challenge, we have provided a [train_baselines.py](misc/cvpr2020_challenge/train_baselines.py) script which will train a basic model for each dataset and store the predictions in the format required for uploading to the server (described below). After fetching the features using the step above, you can launch this script:

```
# pick a GPU device id
gpu_id=0
python misc/cvpr2020_challenge/train_baselines.py --device $gpu_id
```

Alternatively, if you are looking to get your feet wet with a simple example, you can train a single model for a single dataset as follows:

```
# pick a GPU device id
gpu_id=0
dataset=msvd
python train.py --config configs --device $gpu_id
```

NOTES:
* It is possible to train models without a GPU, but it's quite slow.
* If you have access to a SLURM scheduler, adding the `--yaspify` flag will train each of the baselines in parallel (this may be helpful to save some time).

**Uploading predictions to the server:**

The challenge is hosted on CodaLab [FIX LINK](Fix-link).  

**Trying out new ideas!**

There are undoubtedly many ways to improve upon the baseline. Here are just a few suggestions/ideas for possible directions:
* Using more modalities: there are many more preprocessed features available than the ones used in the baseline so experimenting with different combinations may produce good results.
* Using temporal information: As well as average/max pooled features, there are also features which have been sampled into segments (suitable for use with a TRN-like architecture). 
* Combining different datasets: currenlty, a model is trained for each dataset independently of the others
* Trying out new text encodings: currently the system uses either word2vec or OpenAI embeddings and a simple NetVLAD text encoder: there are likely many better ways to encode queries/descriptions
* Handling missing modalities: currently, missing modalities are simply zero-padded, but handling them in a "smart" way may help performance
* There are surely many other things we haven't thought of :)

