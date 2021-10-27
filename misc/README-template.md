### Collaborative Experts

![CE diagram](figs/CE.png)



**High-level Overview**: The *Collaborative Experts* framework aims to achieve robustness through two mechanisms:
1. The use of information from a wide range of modalities, including those that are typically always available in video (such as RGB) as well as more "specific" clues which may only occasionally be present (such as overlaid text).
2. A module that aims to combine these modalities into a fixed size representation that in a manner that is robust to noise.

**Requirements:** The code assumes PyTorch 1.4 and Python 3.7 (other versions may work, but have not been tested).  See the section on dependencies towards the end of this file for specific package requirements.


**Important: A note on the updated results**: A previous version of the codebase (and paper) reported results on the retrieval benchmarks that included a signficant software bug leading to an overestimate of performance.  We are extremely grateful to Valentin Gabeur who discovered this bug (it has been corrected in the current codebase).


### CVPR 2020: Pentathlon challenge

<p align="center">
<img width="300" alt="logo" src="figs/logo-centre.png">
</p>

We are hosting a video retrieval challenge as part of the Video Pentathlon Workshop. Find out how to participate [here](misc/challenge.md)!


### Pretrained video embeddings

We provide pretrained models for each dataset to reproduce the results reported in the paper [1] (references follow at the end of this README).  Each model is accompanied by training and evaluation logs.  Performance is evalauted for retrieval in both directions (joint-embeddings can be used for either of these two tasks):
* `t2v` denotes that a text query is used to retrieve videos
* `v2t` denotes that a video query is used to retrieve text video descriptions

In the results reported below, the same model is used for both the t2v and v2t evaluations.  Each metric is reported as the mean and standard deviation (in parentheses) across three training runs.


**MSRVTT Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | ----- |
| CE    | Full  | t2v  | {{msrvtt-train-full-ce.t2v}} | [config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log]({{msrvtt-train-full-ce.log}}) |
| CE    | 1k-A  | t2v  | {{msrvtt-train-jsfusion-ce.t2v}} | [config]({{msrvtt-train-jsfusion-ce.config}}), [model]({{msrvtt-train-jsfusion-ce.model}}), [log]({{msrvtt-train-jsfusion-ce.log}}) |
| CE    | 1k-B  | t2v  | {{msrvtt-train-miech-ce.t2v}} | [config]({{msrvtt-train-miech-ce.config}}), [model]({{msrvtt-train-miech-ce.model}}), [log]({{msrvtt-train-miech-ce.log}}) |
| MoEE* | 1k-B  | t2v  | {{msrvtt-train-miech-miechfeats-moee.t2v}} | [config]({{msrvtt-train-miech-miechfeats-moee.config}}), [model]({{msrvtt-train-miech-miechfeats-moee.model}}), [log]({{msrvtt-train-miech-miechfeats-moee.log}}) |
| CE    | Full  | v2t  | {{msrvtt-train-full-ce.v2t}} | [config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log]({{msrvtt-train-full-ce.log}}) |
| CE    | 1k-A  | v2t  | {{msrvtt-train-jsfusion-ce.v2t}} | [config]({{msrvtt-train-jsfusion-ce.config}}), [model]({{msrvtt-train-jsfusion-ce.model}}), [log]({{msrvtt-train-jsfusion-ce.log}}) |
| CE    | 1k-B  | v2t  | {{msrvtt-train-miech-ce.v2t}} | [config]({{msrvtt-train-miech-ce.config}}), [model]({{msrvtt-train-miech-ce.model}}), [log]({{msrvtt-train-miech-ce.log}}) |
| MoEE* | 1k-B  | v2t  | {{msrvtt-train-miech-miechfeats-moee.v2t}} | [config]({{msrvtt-train-miech-miechfeats-moee.config}}), [model]({{msrvtt-train-miech-miechfeats-moee.model}}), [log]({{msrvtt-train-miech-miechfeats-moee.log}}) |

Models marked with * use the features made available with the MoEE model of [2] (without OCR, speech and scene features), unstarred models on the `1k-B` and `Full` splits make use of OCR, speech and scene features, as well slightly stronger text encodings (GPT, rather than word2vec - see [1] for details). The MoEE model is implemented as a sanity check that our codebase approximately reproduces [2] (the [MoEE paper](https://arxiv.org/abs/1804.02516)).


See the [MSRVTT README](misc/datasets/msrvtt/README.md) for links to the train/val/test lists of each split.

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

**LSMDC Benchmark**

| Model | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Links |
| ------| ------| ---:| ---:| ----:| ----:|----:|----:|------:|
| CE | t2v  | {{lsmdc-train-full-ce.t2v}} | [config]({{lsmdc-train-full-ce.config}}), [model]({{lsmdc-train-full-ce.model}}), [log]({{lsmdc-train-full-ce.log}}) |
| CE | v2t  | {{lsmdc-train-full-ce.v2t}} | [config]({{lsmdc-train-full-ce.config}}), [model]({{lsmdc-train-full-ce.model}}), [log]({{lsmdc-train-full-ce.log}}) |

See the [LSMDC README](misc/datasets/lsmdc/README.md) for descriptions of the train/test splits. Please note that to obtain the features and descriptions for this dataset, you must obtain permission from MPII to use the data (this is process is described [here](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/mpii-movie-description-dataset/request-access-to-mpii-movie-description-dataset/).  Once you have done so, please request that a member of the LSMDC team contacts us to confirm approval (via albanie at robots dot ox dot ac dot uk) - we can then provide you with a link to the features.



<<misc/README-ablations-template.md:msrvtt:msrvtt>>
<<misc/README-restricted-captions-msrvtt-template.md:msrvtt:msrvtt>>

Similar ablation studies for the remaining datasets can be found [here](misc/ablations.md).

### Expert Zoo

For each dataset, the Collaborative Experts model makes use of a collection of pretrained "expert" feature extractors (see [1] for more precise descriptions). Some experts have been obtained from other sources (described where applicable), rather than extracted by us.  To reproduce the experiments listed above, the experts for each dataset have been bundled into compressed tar files.  These can be downloaded and unpacked with a [utility script](misc/sync_experts.py) (recommended -- see example usage below), which will store them in the locations expected by the training code. Each set of experts has a brief README, which also provides a link from which they can be downloaded directly.

| Dataset           | Experts  |  Details and links | Archive size | sha1sum |
|:-------------:|:-----:|:----:|:---:|:---:|
| MSRVTT | audio, face, flow, ocr, rgb, scene, speech | [README](misc/datasets/msrvtt/README.md)| 19.6 GiB | <sup><sub><sup><sub>959bda588793ef05f348d16de26da84200c5a469</sub></sup></sub></sup> |
| LSMDC | audio, face, flow, ocr, rgb, scene | [README](misc/datasets/lsmdc/README.md)| 6.1 GiB | <sup><sub><sup><sub>7ce018e981752db9e793e449c2ba5bc88217373d</sub></sup></sub></sup> |
| MSVD | face, flow, ocr, rgb, scene | [README](misc/datasets/msvd/README.md)| 2.1 GiB | <sup><sub><sup><sub>6071827257c14de455b3a13fe1e885c2a7887c9e</sub></sup></sub></sup> | 
| DiDeMo | audio, face, flow, ocr, rgb, scene, speech | [README](misc/datasets/didemo/README.md)| 2.3 GiB | <sup><sub><sup><sub>6fd4bcc68c1611052de2499fd8ab3f488c7c195b</sub></sup></sub></sup> | 
| ActivityNet | audio, face, flow, ocr, rgb, scene, speech | [README](misc/datasets/activity-net/README.md)| 3.8 GiB | <sup><sub><sup><sub>b16685576c97cdec2783fb89ea30ca7d17abb021</sub></sup></sub></sup> | 

### QuerYD

<<misc/README-model-study.md:msrvtt:queryd>>

The influence of different pretrained experts for the performance of the CE model trained on QuerYD is studied. The value and cumulative effect of different experts for scene clas-sification (SCENE), ambient sound classification (AUDIO),image classification (OBJECT), and action recognition (ACTION) are presented. PREV. denotes the experts used in the previous row.

<<misc/README-queryd-ablations-template.md:queryd:queryd>>

### QuerYDSegments

<<misc/README-model-study.md:msrvtt:querydsegments>>

### Evaluating a pretrained model

Evaluting a pretrained model for a given dataset requires:
1. The pretrained experts for the target dataset, which should be located in `<root>/data/<dataset-name>/symlinked-feats` (this will be done automatically by the [utility script](misc/sync_experts.py), or can be done manually).
2. A `config.json` file.
3. A `trained_model.pth` file.

Evaluation is then performed with the following command:
```
python3 test.py --config <path-to-config.json> --resume <path-to-trained_model.pth> --device <gpu-id>
```
where `<gpu-id>` is the index of the GPU to evaluate on.  This option can be ommitted to run the evaluation on the CPU.

For example, to reproduce the MSVD results described above, run the following sequence of commands:

```
# fetch the pretrained experts for MSVD 
python3 misc/sync_experts.py --dataset MSVD

# find the name of a pretrained model using the links in the tables above 
export MODEL=data/models/msvd-train-full-ce/5bb8dda1/seed-0/2020-01-30_12-29-56/trained_model.pth

# create a local directory and download the model into it 
mkdir -p $(dirname "${MODEL}")
wget --output-document="${MODEL}" "http://www.robots.ox.ac.uk/~vgg/research/collaborative-experts/${MODEL}"

# Evaluate the model
python3 test.py --config configs/msvd/train-full-ce.json --resume ${MODEL} --device 0 --eval_from_training_config
```


### Training a new model

Training a new video-text embedding requires:
1. The pretrained experts for the dataset used for training, which should be located in `<root>/data/<dataset-name>/symlinked-feats` (this will be done automatically by the [utility script](misc/sync_experts.py), or can be done manually).
2. A `config.json` file.  You can define your own, or use one of the provided configs in the [configs](configs) directory.

Training is then performed with the following command:
```
python3 train.py --config <path-to-config.json> --device <gpu-id>
```
where `<gpu-id>` is the index of the GPU to train on.  This option can be ommitted to run the training on the CPU.

For example, to train a new embedding for the LSMDC dataset, run the following sequence of commands:

```
# fetch the pretrained experts for LSMDC 
python3 misc/sync_experts.py --dataset LSMDC

# Train the model
python3 train.py --config configs/lsmdc/train-full-ce.json --device 0
```

### Visualising the retrieval ranking

Tensorboard lacks video support via HTML5 tags (at the time of writing) so after each evaluation of a retrieval model, a simple HTML file is generated to allow the predicted rankings of different videos to be visualised: an example screenshot is given below (this tool is inspired by the visualiser in the [pix2pix codebase](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)). To view the visualisation, navigate to the `web directory` (this is generated for each experiment, and will be printed in the log during training) and run `python3 -m http.server 9999`, then navigate to `localhost:9999` in your web browser.  You should see something like the following:

![visualisation](figs/vis-ranking.png)

Note that the visualising the results in this manner requires that you also download the source videos for each of the datasets to some directory `<src-video-dir>`. Then set the `visualizer.args.src_video_dir` attribute of the training `config.json` file to point to `<src-video-dir>`.


### Dependencies

Dependencies can be installed via `pip install -r requirements/pip-requirements.txt`.


### References

[1] If you find this code useful or use the extracted features, please consider citing:

```
@inproceedings{croitoru2021teachtext,
  title={Teachtext: Crossmodal generalized distillation for text-video retrieval},
  author={Croitoru, I. and Bogolin, S. and Leordeanu, M. and Jin, H. and Zisserman, A. and Albanie, S. and Liu, Y.},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={11583--11593},
  year={2021}
}

@inproceedings{Liu2019a,
  author    = {Liu, Y. and Albanie, S. and Nagrani, A. and Zisserman, A.},
  booktitle = {arXiv preprint arxiv:1907.13487},
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

This work was inspired by a number of prior works for learning joint embeddings of text and video, but in particular the *Mixture-of-Embedding-Experts* method proposed by Antoine Miech, Ivan Laptev and Josef Sivic ([paper](https://arxiv.org/abs/1804.02516), [code](https://github.com/antoine77340/Mixture-of-Embedding-Experts)). We would also like to thank Zak Stone and Susie Lim for their help with using Cloud TPUs.  The code structure uses the [pytorch-template](https://github.com/victoresque/pytorch-template) by @victoresque.
