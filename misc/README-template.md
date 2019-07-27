## Collaborative Experts

This repo provides code for learning and evaluating joint video-text embeddings for the task of video retrieval.  Our approach is described in the [collaborative experts paper](link).  

![CE diagram](figs/CE-diagram.png)

In brief, the *Collaborative Experts* framework aims to achieve robustness through two mechanisms:
1. Extracting information from a wide range of modalities, including those that are typically always available in video (such as RGB) as well as more "specific" clues which may only occasionally be present (such as overlaid text).
2. A module that aims to combine these modalities into a fixed size representation that in a manner that is robust to noise.


**Requirements:** The code assumes PyTorch 1.1 and Python 3.7 (other versions may work, but have not been tested).  See the section on dependencies towards the end of this file for specific package requirements.

### Pretrained video embeddings

We provide pretrained models for each dataset to reproduce the results reported in the paper [1] (references follow at the end of this README).  Each model is accompanied by training and evaluation logs.  Performance is evalauted for retrieval in both directions (joint-embeddings can be used for either of these two tasks):
* `t2v` denotes that a text query is used to retrieve videos
* `v2t` denotes that a video query is used to retrieve text video descriptions

In the results reported below, the same model is used for both the t2v and v2t evaluations.  Each metric is reported as the mean and standard deviation (in parentheses) across three training runs.

**Reference results:** The results below are close to those in [1] for MSRVTT, LSMDC, MSRVTT (the mean performance should vary by at most +/- 0.6 across different metrics from those reported).  The performance for ActivityNet and DiDeMo has improved, in some cases quite significantly, after removing some bugs in the training code.


**MSRVTT Benchmark**


| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | ----- |
| CE    | Full  | t2v  | {{msrvtt-train-full-ce.t2v}} | [config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log]({{msrvtt-train-full-ce.log}}) |
| CE    | 1k-A  | t2v  | {{msrvtt-train-jsfusion-ce.t2v}} | [config]({{msrvtt-train-jsfusion-ce.config}}), [model]({{msrvtt-train-jsfusion-ce.model}}), [log]({{msrvtt-train-jsfusion-ce.log}}) |
| CE    | 1k-B  | t2v  | {{msrvtt-train-miech-ce.t2v}} | [config]({{msrvtt-train-miech-ce.config}}), [model]({{msrvtt-train-miech-ce.model}}), [log]({{msrvtt-train-miech-ce.log}}) |
| MoEE [2] | Full  | t2v  | {{msrvtt-train-full-moee.t2v}} | [config]({{msrvtt-train-full-moee.config}}), [model]({{msrvtt-train-full-moee.model}}), [log]({{msrvtt-train-full-moee.log}}) |
| MoEE [2]* | 1k-B  | t2v  | {{msrvtt-train-miech-miechfeats-moee.t2v}} | [config]({{msrvtt-train-miech-miechfeats-moee.config}}), [model]({{msrvtt-train-miech-miechfeats-moee.model}}), [log]({{msrvtt-train-miech-miechfeats-moee.log}}) |
| CE    | Full  | v2t  | {{msrvtt-train-full-ce.v2t}} | [config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log]({{msrvtt-train-full-ce.log}}) |
| CE    | 1k-A  | v2t  | {{msrvtt-train-jsfusion-ce.v2t}} | [config]({{msrvtt-train-jsfusion-ce.config}}), [model]({{msrvtt-train-jsfusion-ce.model}}), [log]({{msrvtt-train-jsfusion-ce.log}}) |
| CE    | 1k-B  | v2t  | {{msrvtt-train-miech-ce.v2t}} | [config]({{msrvtt-train-miech-ce.config}}), [model]({{msrvtt-train-miech-ce.model}}), [log]({{msrvtt-train-miech-ce.log}}) |
| MoEE [2] | Full  | v2t  | {{msrvtt-train-full-moee.v2t}} | [config]({{msrvtt-train-full-moee.config}}), [model]({{msrvtt-train-full-moee.model}}), [log]({{msrvtt-train-full-moee.log}}) |
| MoEE [2]* | 1k-B  | v2t  | {{msrvtt-train-miech-miechfeats-moee.v2t}} | [config]({{msrvtt-train-miech-miechfeats-moee.config}}), [model]({{msrvtt-train-miech-miechfeats-moee.model}}), [log]({{msrvtt-train-miech-miechfeats-moee.log}}) |

Models marked with * use the features made available with the MoEE model of [2] (without OCR, speech and scene features), unstarred models on the `1k-B` and `Full` splits make use of OCR, speech and scene features, as well slightly stronger text encodings (GPT, rather than word2vec - see the paper for details). The MoEE model is implemented as a baseline and as a sanity check that our codebase approximately reproduces [2] (the [MoEE paper](https://arxiv.org/abs/1804.02516)).


See the [MSRVTT README](misc/datasets/msrvtt/README.md) for links to the train/val/test lists of each split.

**LSMDC Benchmark**

| Model | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Links |
| ------| ------| ---:| ---:| ----:| ----:|----:|----:|------:|
| CE | t2v  | {{lsmdc-train-full-ce.t2v}} | [config]({{lsmdc-train-full-ce.config}}), [model]({{lsmdc-train-full-ce.model}}), [log]({{lsmdc-train-full-ce.log}}) |
| CE | v2t  | {{lsmdc-train-full-ce.v2t}} | [config]({{lsmdc-train-full-ce.config}}), [model]({{lsmdc-train-full-ce.model}}), [log]({{lsmdc-train-full-ce.log}}) |

See the [LSMDC README](misc/datasets/lsmdc/README.md) for descriptions of the train/test splits.

**MSVD Benchmark**

| Model | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Links |
| ------| ------| ---:| ---:| ----:| ----:|----:|----:|------:|
| CE | t2v  | {{msvd-train-full-ce.t2v}} | [config]({{msvd-train-full-ce.config}}), [model]({{msvd-train-full-ce.model}}), [log]({{msvd-train-full-ce.log}}) |
| CE | v2t  | {{msvd-train-full-ce.v2t}} | [config]({{msvd-train-full-ce.config}}), [model]({{msvd-train-full-ce.model}}), [log]({{msvd-train-full-ce.log}}) |

See the [MSVD README](misc/datasets/msvd/README.md) for descriptions of the train/test splits. Note that the videos in the MSVD dataset do not have soundtracks.

**DiDeMo Benchmark**

| Model | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Links |
| ------| ------| ---:| ---:| ----:| ----:|----:|----:|------:|
| CE | t2v  | {{didemo-train-full-ce.t2v}} | [config]({{didemo-train-full-ce.config}}), [model]({{didemo-train-full-ce.model}}), [log]({{didemo-train-full-ce.log}}) |
| CE | v2t  | {{didemo-train-full-ce.v2t}} | [config]({{didemo-train-full-ce.config}}), [model]({{didemo-train-full-ce.model}}), [log]({{didemo-train-full-ce.log}}) |

See the [DiDeMo README](misc/datasets/didemo/README.md) for descriptions of the train/val/test splits.

**ActivityNet Benchmark**

| Model | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Links |
| ------| ------| ---:| ---:| ----:| ----:|----:|----:|------:|
| CE | t2v  | {{activity-net-train-full-ce.t2v}} | [config]({{activity-net-train-full-ce.config}}), [model]({{activity-net-train-full-ce.model}}), [log]({{activity-net-train-full-ce.log}}) |
| CE | v2t  | {{activity-net-train-full-ce.v2t}} | [config]({{activity-net-train-full-ce.config}}), [model]({{activity-net-train-full-ce.model}}), [log]({{activity-net-train-full-ce.log}}) |

See the [ActivityNet README](misc/datasets/activity-net/README.md) for descriptions of the train/test splits.

### Expert Zoo

For each dataset, the Collaborative Experts model makes use of a collection of pretrained "expert" feature extractors (see the paper for more precise descriptions). Some experts have been obtained from other sources (described where applicable), rather than extracted by us.  To reproduce the experiments listed above, the experts for each dataset have been bundled into compressed tar files.  These can be downloaded and unpacked with a [utility script-TODO-LINK]() (recommended), which will store them in the locations expected by the training code. Each set of experts has a brief README, which also provides a link from which they can be downloaded directly.

  | Dataset           | Experts  |  Details and links | Archive size |
 |:-------------:|:-----:|:----:|:---:|
| MSRVTT | audio, face, flow, ocr, rgb, scene, speech | [README](misc/datasets/msrvtt/README.md)| 19.6 GiB
| LSMDC | audio, face, flow, ocr, rgb, scene | [README](misc/datasets/lsmdc/README.md)| 6.1 GiB
| MSVD | face, flow, ocr, rgb, scene | [README](misc/datasets/msvd/README.md)| 2.1 GiB
| DiDeMo | audio, face, flow, ocr, rgb, scene, speech | [README](misc/datasets/didemo/README.md)| 2.3 GiB
| ActivityNet | audio, face, flow, ocr, rgb, scene, speech | [README](misc/datasets/activity-net/README.md)| 3.8 GiB

### Evaluating a pretrained model

Evaluting a pretrained model for a given dataset requires:
1. The pretrained experts for the target dataset, which should be located in `<root>/data/<dataset-name>/symlinked-feats` (this will be done automatically by the [utility-script](), or can be done manually).
2. A `config.json` file.
3. A `trained_model.pth` file.

Evaluation is then performed with the following command:
```
python3 test.py --config <path-to-config.json> --resume <path-to-trained_model.pth> --device <gpu-id>
```
where `<gpu-id>` is the index of the GPU to evaluate on.  This option can be ommitted to run the evaluation on the CPU.

For example, to reproduce the MSVD results described above, run the following sequence of commands:

```
# fetch the pretrained experts for MSVD 
utility-expert FETCH MSVD

# find the name of a pretrained model using the links in the tables above 
MODEL=data/models/msvd-train-full-ce/07-25_15-18-17/trained_model.path

# create a local directory and download the model into it 
mkdir -p $(dirname "${MODEL}")
wget http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/models/${MODEL} ${MODEL}

# Evaluate the model
python3 test.py --config configs/msvd/eval-full-ce.json --resume ${MODEL} --device 0
```


### Training a new model

Training a new video-text embedding requires:
1. The pretrained experts for the dataset used for training, which should be located in `<root>/data/<dataset-name>/symlinked-feats` (this will be done automatically by the [utility-script](), or can be done manually).
2. A `config.json` file.  You can define your own, or use one of the provided configs in the [configs](configs) directory.

Training is then performed with the following command:
```
python3 train.py --config <path-to-config.json> --device <gpu-id>
```
where `<gpu-id>` is the index of the GPU to train on.  This option can be ommitted to run the training on the CPU.

For example, to train a new embedding for the LSMDC dataset, run the following sequence of commands:

```
# fetch the pretrained experts for LSMDC 
utility-expert FETCH LSMDC

# Train the model
python3 train.py --config configs/lsmdc/train-full-ce.json --device 0
```

### Visualising the retrieval ranking

Tensorboard lacks video support via HTML5 tags (at the time of writing) so after each evaluation of a retrieval model, a simple HTML file is generated to allow the predicted rankings of different videos to be visualised: an example screenshot is given below (this tool is inspired by the visualiser in the [pix2pix codebase](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)). To view the visualisation, navigate to the `web directory` (this is generated for each experiment, and will be printed in the log during training) and run `python3 -m http.server 9999`, then navigate to `localhost:9999` in your web browser.  You should see something like the following:

![visualisation](figs/vis-ranking.png)


### Dependencies

If you have enough disk space, the recommended approach to installing the dependencies for this project is to create a conda enviroment via the `requirements/conda-requirements.txt`:

```
conda create --name pt11 --file requirements/conda-requirements.txt
```

Otherwise, if you'd prefer to take a leaner approach, you can either:
1. `pip install` each missing package each time you hit an `ImportError`
2. manually inspect the slightly more readable `requirements/pip-requirements.txt`


### References

[1] If you find this code useful or use the extracted features, please consider citing:

```
@inproceedings{Liu2019a,
  author    = {Liu, Y. and Albanie, S. and Nagrani, A. and Zisserman, A.},
  booktitle = {British Machine Vision Conference},
  title     = {Use What You Have: Video retrieval using representations from collaborative experts},
  date      = {2019},
}
```

[2] If you make use of the MSRVTT or LSMDC features provided by Miech et al. (details are given in their respective READMEs [here](misc/datasets/msrvtt/README.md) and [here](misc/datasets/lsmdc/README.md)), please cite:

```
@article{miech2018learning,
  title={Learning a text-video embedding from incomplete and heterogeneous data},
  author={Miech, Antoine and Laptev, Ivan and Sivic, Josef},
  journal={arXiv preprint arXiv:1804.02516},
  year={2018}
}
```



### Acknowledgements

This work was inspired by a number of prior works for learning joint embeddings of text and video, but in particular the *Mixture-of-Embedding-Experts* method proposed by Antoine Miech, Ivan Laptev and Josef Sivic ([paper](https://arxiv.org/abs/1804.02516), [code](https://github.com/antoine77340/Mixture-of-Embedding-Experts)). We would also like to thank Zak Stone and Susie Lim for their considerable help with using Cloud TPUs.