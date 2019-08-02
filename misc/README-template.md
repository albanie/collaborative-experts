## Collaborative Experts

This repo provides code for learning and evaluating joint video-text embeddings for the task of video retrieval.  Our approach is described in the BMVC 2019 paper "Use What You Have: Video retrieval using representations from collaborative experts" ([paper](https://arxiv.org/abs/1907.13487), [project page](https://www.robots.ox.ac.uk/~vgg/research/collaborative-experts/)).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/use-what-you-have-video-retrieval-using/video-retrieval-on-lsmdc)](https://paperswithcode.com/sota/video-retrieval-on-lsmdc?p=use-what-you-have-video-retrieval-using)


![CE diagram](figs/CE.png)



**High-level Overview**: The *Collaborative Experts* framework aims to achieve robustness through two mechanisms:
1. The use of information from a wide range of modalities, including those that are typically always available in video (such as RGB) as well as more "specific" clues which may only occasionally be present (such as overlaid text).
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
| MoEE* | 1k-B  | t2v  | {{msrvtt-train-miech-miechfeats-moee.t2v}} | [config]({{msrvtt-train-miech-miechfeats-moee.config}}), [model]({{msrvtt-train-miech-miechfeats-moee.model}}), [log]({{msrvtt-train-miech-miechfeats-moee.log}}) |
| CE    | Full  | v2t  | {{msrvtt-train-full-ce.v2t}} | [config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log]({{msrvtt-train-full-ce.log}}) |
| CE    | 1k-A  | v2t  | {{msrvtt-train-jsfusion-ce.v2t}} | [config]({{msrvtt-train-jsfusion-ce.config}}), [model]({{msrvtt-train-jsfusion-ce.model}}), [log]({{msrvtt-train-jsfusion-ce.log}}) |
| CE    | 1k-B  | v2t  | {{msrvtt-train-miech-ce.v2t}} | [config]({{msrvtt-train-miech-ce.config}}), [model]({{msrvtt-train-miech-ce.model}}), [log]({{msrvtt-train-miech-ce.log}}) |
| MoEE* | 1k-B  | v2t  | {{msrvtt-train-miech-miechfeats-moee.v2t}} | [config]({{msrvtt-train-miech-miechfeats-moee.config}}), [model]({{msrvtt-train-miech-miechfeats-moee.model}}), [log]({{msrvtt-train-miech-miechfeats-moee.log}}) |

Models marked with * use the features made available with the MoEE model of [2] (without OCR, speech and scene features), unstarred models on the `1k-B` and `Full` splits make use of OCR, speech and scene features, as well slightly stronger text encodings (GPT, rather than word2vec - see [1] for details). The MoEE model is implemented as a sanity check that our codebase approximately reproduces [2] (the [MoEE paper](https://arxiv.org/abs/1804.02516)).


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


### Ablation studies

We conduct several ablation studies to investigate the importance of different components in the Collaborative Experts design.  Each ablation is conducted on the `Full` MSRVTT split. 

**CE Design**: First, we investigate the importance of the parts used by the CE model.

| Model | Task | R@1 | R@5 | R@10 | MdR | Params | Links |
| ---   | :--: | :-: | :-: | :--: | :-: | :----: | :---: |
| Concat | t2v  | {{msrvtt-train-full-concat-ablation.short-t2v}} | {{msrvtt-train-full-concat-ablation.params}} | [config]({{msrvtt-train-full-concat-ablation.config}}), [model]({{msrvtt-train-full-concat-ablation.model}}), [log]({{msrvtt-train-full-concat-ablation.log}}) |
| Concat + G | t2v  | {{msrvtt-train-full-concat-mix-ablation.short-t2v}} | {{msrvtt-train-full-concat-mix-ablation.params}} | [config]({{msrvtt-train-full-concat-mix-ablation.config}}), [model]({{msrvtt-train-full-concat-mix-ablation.model}}), [log]({{msrvtt-train-full-concat-mix-ablation.log}}) |
| CE - MW,P,CG | t2v  | {{msrvtt-train-full-moee-minus-moe-weights.short-t2v}} | {{msrvtt-train-full-moee-minus-moe-weights.params}} | [config]({{msrvtt-train-full-moee-minus-moe-weights.config}}), [model]({{msrvtt-train-full-moee-minus-moe-weights.model}}), [log]({{msrvtt-train-full-moee-minus-moe-weights.log}}) |
| CE - P,CG | t2v  | {{msrvtt-train-full-moee.short-t2v}} | {{msrvtt-train-full-moee.params}} | [config]({{msrvtt-train-full-moee.config}}), [model]({{msrvtt-train-full-moee.model}}), [log]({{msrvtt-train-full-moee.log}}) |
| CE - CG  | t2v  | {{msrvtt-train-full-ce-ablation-dims.short-t2v}} | {{msrvtt-train-full-ce-ablation-dims.params}} | [config]({{msrvtt-train-full-ce-ablation-dims.config}}), [model]({{msrvtt-train-full-ce-ablation-dims.model}}), [log]({{msrvtt-train-full-ce-ablation-dims.log}}) |
| CE    | t2v  | {{msrvtt-train-full-ce.short-t2v}} | {{msrvtt-train-full-ce.params}} | [config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log]({{msrvtt-train-full-ce.log}}) |
| Concat | v2t  | {{msrvtt-train-full-concat-ablation.short-v2t}} | {{msrvtt-train-full-concat-ablation.params}} | [config]({{msrvtt-train-full-concat-ablation.config}}), [model]({{msrvtt-train-full-concat-ablation.model}}), [log]({{msrvtt-train-full-concat-ablation.log}}) |
| Concat + G | v2t  | {{msrvtt-train-full-concat-mix-ablation.short-v2t}} | {{msrvtt-train-full-concat-mix-ablation.params}} | [config]({{msrvtt-train-full-concat-mix-ablation.config}}), [model]({{msrvtt-train-full-concat-mix-ablation.model}}), [log]({{msrvtt-train-full-concat-mix-ablation.log}}) |
| CE - MW,P,CG | v2t  | {{msrvtt-train-full-moee-minus-moe-weights.short-v2t}} | {{msrvtt-train-full-moee-minus-moe-weights.params}} | [config]({{msrvtt-train-full-moee-minus-moe-weights.config}}), [model]({{msrvtt-train-full-moee-minus-moe-weights.model}}), [log]({{msrvtt-train-full-moee-minus-moe-weights.log}}) |
| CE - P,CG | v2t  | {{msrvtt-train-full-moee.short-v2t}} | {{msrvtt-train-full-moee.params}} | [config]({{msrvtt-train-full-moee.config}}), [model]({{msrvtt-train-full-moee.model}}), [log]({{msrvtt-train-full-moee.log}}) |
| CE - CG  | v2t  | {{msrvtt-train-full-ce-ablation-dims.short-v2t}} | {{msrvtt-train-full-ce-ablation-dims.params}} | [config]({{msrvtt-train-full-ce-ablation-dims.config}}), [model]({{msrvtt-train-full-ce-ablation-dims.model}}), [log]({{msrvtt-train-full-ce-ablation-dims.log}}) |
| CE    | v2t  | {{msrvtt-train-full-ce.short-v2t}} | {{msrvtt-train-full-ce.params}} | [config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log]({{msrvtt-train-full-ce.log}}) |

Each row adds an additional component to the model.  The names refer to the following model designs:
* **Concat**: A barebones concatenation model.  After aggregating each expert across time (which still requires some parameters for the variable-length VLAD layers), the experts are concatenated and compared directly against the aggregated text embeddings.  Note: this model uses a slightly greater number of VLAD clusters than the others to allow the concatentated embedding to match the dimensionality of the text.
* **Concat-G**: The experts are concatenated (similarly to the previous model), but are then passed through a single large context gating module before matching against the text embedding.
* **CE - MW,P,CG** - The CE model without MoE weights, projecting to a common dimension or Collaborative Gating.
* **CE - P,CG** - The CE model without projecting to a common dimension or Collaborative Gating (note that this is equivalent to the MoEE model proposed in [2]).
* **CE - CG** - The CE model without Collaborative Gating (CG).
* **CE** - The full CE model.

Note that in the table above some metrics have been removed to allow the number of parameters to be displayed---these additional metrics can be found in the linked logs.

**Importance of Different Experts**: The next ablation investigates the value of each of the different experts towards the final embedding.  Since not all experts are available in every video, we pair each expert with RGB, to give an approximation of their usefulness.  We use term "RGB" to denote features extracted from the RGB signal using a classification model (typically a [Squeeze-and-Excitation](https://arxiv.org/abs/1709.01507) network) pretrained on ImageNet (more details for the RGB and other experts can be found in [1]).

| Experts | Task | R@1 | R@5 | R@10 | MdR | Params | Links |
| -----   | :--: | :-: | :-: | :--: | :-: | :----: | :---: |
| RGB    | t2v  | {{msrvtt-train-full-ce-only-rgb.short-t2v}} | {{msrvtt-train-full-ce-only-rgb.params}} | [config]({{msrvtt-train-full-ce-only-rgb.config}}), [model]({{msrvtt-train-full-ce-only-rgb.model}}), [log]({{msrvtt-train-full-ce-only-rgb.log}}) |
| RGB + Scene | t2v  | {{msrvtt-train-full-ce-only-rgb-scene.short-t2v}} | {{msrvtt-train-full-ce-only-rgb-scene.params}} | [config]({{msrvtt-train-full-ce-only-rgb-scene.config}}), [model]({{msrvtt-train-full-ce-only-rgb-scene.model}}), [log]({{msrvtt-train-full-ce-only-rgb-scene.log}}) |
| RGB + Flow | t2v  | {{msrvtt-train-full-ce-only-rgb-flow.short-t2v}} | {{msrvtt-train-full-ce-only-rgb-flow.params}} | [config]({{msrvtt-train-full-ce-only-rgb-flow.config}}), [model]({{msrvtt-train-full-ce-only-rgb-flow.model}}), [log]({{msrvtt-train-full-ce-only-rgb-flow.log}}) |
| RGB + Audio | t2v  | {{msrvtt-train-full-ce-only-rgb-audio.short-t2v}} | {{msrvtt-train-full-ce-only-rgb-audio.params}} | [config]({{msrvtt-train-full-ce-only-rgb-audio.config}}), [model]({{msrvtt-train-full-ce-only-rgb-audio.model}}), [log]({{msrvtt-train-full-ce-only-rgb-audio.log}}) |
| RGB + OCR | t2v  | {{msrvtt-train-full-ce-only-rgb-ocr.short-t2v}} | {{msrvtt-train-full-ce-only-rgb-ocr.params}} | [config]({{msrvtt-train-full-ce-only-rgb-ocr.config}}), [model]({{msrvtt-train-full-ce-only-rgb-ocr.model}}), [log]({{msrvtt-train-full-ce-only-rgb-ocr.log}}) |
| RGB + Speech | t2v  | {{msrvtt-train-full-ce-only-rgb-speech.short-t2v}} | {{msrvtt-train-full-ce-only-rgb-speech.params}} | [config]({{msrvtt-train-full-ce-only-rgb-speech.config}}), [model]({{msrvtt-train-full-ce-only-rgb-speech.model}}), [log]({{msrvtt-train-full-ce-only-rgb-speech.log}}) |
| RGB + Face | t2v  | {{msrvtt-train-full-ce-only-rgb-face.short-t2v}} | {{msrvtt-train-full-ce-only-rgb-face.params}} | [config]({{msrvtt-train-full-ce-only-rgb-face.config}}), [model]({{msrvtt-train-full-ce-only-rgb-face.model}}), [log]({{msrvtt-train-full-ce-only-rgb-face.log}}) |
| RGB    | v2t  | {{msrvtt-train-full-ce-only-rgb.short-v2t}} | {{msrvtt-train-full-ce-only-rgb.params}} | [config]({{msrvtt-train-full-ce-only-rgb.config}}), [model]({{msrvtt-train-full-ce-only-rgb.model}}), [log]({{msrvtt-train-full-ce-only-rgb.log}}) |
| RGB + Scene | v2t  | {{msrvtt-train-full-ce-only-rgb-scene.short-v2t}} | {{msrvtt-train-full-ce-only-rgb-scene.params}} | [config]({{msrvtt-train-full-ce-only-rgb-scene.config}}), [model]({{msrvtt-train-full-ce-only-rgb-scene.model}}), [log]({{msrvtt-train-full-ce-only-rgb-scene.log}}) |
| RGB + Flow | v2t  | {{msrvtt-train-full-ce-only-rgb-flow.short-v2t}} | {{msrvtt-train-full-ce-only-rgb-flow.params}} | [config]({{msrvtt-train-full-ce-only-rgb-flow.config}}), [model]({{msrvtt-train-full-ce-only-rgb-flow.model}}), [log]({{msrvtt-train-full-ce-only-rgb-flow.log}}) |
| RGB + Audio | v2t  | {{msrvtt-train-full-ce-only-rgb-audio.short-v2t}} | {{msrvtt-train-full-ce-only-rgb-audio.params}} | [config]({{msrvtt-train-full-ce-only-rgb-audio.config}}), [model]({{msrvtt-train-full-ce-only-rgb-audio.model}}), [log]({{msrvtt-train-full-ce-only-rgb-audio.log}}) |
| RGB + OCR | v2t  | {{msrvtt-train-full-ce-only-rgb-ocr.short-v2t}} | {{msrvtt-train-full-ce-only-rgb-ocr.params}} | [config]({{msrvtt-train-full-ce-only-rgb-ocr.config}}), [model]({{msrvtt-train-full-ce-only-rgb-ocr.model}}), [log]({{msrvtt-train-full-ce-only-rgb-ocr.log}}) |
| RGB + Speech | v2t  | {{msrvtt-train-full-ce-only-rgb-speech.short-v2t}} | {{msrvtt-train-full-ce-only-rgb-speech.params}} | [config]({{msrvtt-train-full-ce-only-rgb-speech.config}}), [model]({{msrvtt-train-full-ce-only-rgb-speech.model}}), [log]({{msrvtt-train-full-ce-only-rgb-speech.log}}) |
| RGB + Face | v2t  | {{msrvtt-train-full-ce-only-rgb-face.short-v2t}} | {{msrvtt-train-full-ce-only-rgb-face.params}} | [config]({{msrvtt-train-full-ce-only-rgb-face.config}}), [model]({{msrvtt-train-full-ce-only-rgb-face.model}}), [log]({{msrvtt-train-full-ce-only-rgb-face.log}}) |

We can also study their cumulative effect (experts are added in the order of importance suggested by the table above).

| Experts | Task | R@1 | R@5 | R@10 | MdR | Params | Links |
| -----   | :--: | :-: | :-: | :--: | :-: | :----: | :---: |
| RGB    | t2v  | {{msrvtt-train-full-ce-only-rgb.short-t2v}} | {{msrvtt-train-full-ce-only-rgb.params}} | [config]({{msrvtt-train-full-ce-only-rgb.config}}), [model]({{msrvtt-train-full-ce-only-rgb.model}}), [log]({{msrvtt-train-full-ce-only-rgb.log}}) |
| Prev. + Scene    | t2v  | {{msrvtt-train-full-ce-only-rgb-scene.short-t2v}} | {{msrvtt-train-full-ce-only-rgb-scene.params}} | [config]({{msrvtt-train-full-ce-only-rgb-scene.config}}), [model]({{msrvtt-train-full-ce-only-rgb-scene.model}}), [log]({{msrvtt-train-full-ce-only-rgb-scene.log}}) |
| Prev. + Flow    | t2v  | {{msrvtt-train-full-ce-only-rgb-scene-flow.short-t2v}} | {{msrvtt-train-full-ce-only-rgb-scene-flow.params}} | [config]({{msrvtt-train-full-ce-only-rgb-scene-flow.config}}), [model]({{msrvtt-train-full-ce-only-rgb-scene-flow.model}}), [log]({{msrvtt-train-full-ce-only-rgb-scene-flow.log}}) |
| Prev. + Audio    | t2v  | {{msrvtt-train-full-ce-only-rgb-scene-flow-audio.short-t2v}} | {{msrvtt-train-full-ce-only-rgb-scene-flow-audio.params}} | [config]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio.config}}), [model]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio.model}}), [log]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio.log}}) |
| Prev. + OCR    | t2v  | {{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr.short-t2v}} | {{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr.params}} | [config]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr.config}}), [model]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr.model}}), [log]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr.log}}) |
| Prev. + Speech    | t2v  | {{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr-speech.short-t2v}} | {{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr-speech.params}} | [config]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr-speech.config}}), [model]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr-speech.model}}), [log]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr-speech.log}}) |
| Prev. + Face    | t2v  | {{msrvtt-train-full-ce.short-t2v}} | {{msrvtt-train-full-ce.params}} | [config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log]({{msrvtt-train-full-ce.log}}) |
| RGB    | v2t  | {{msrvtt-train-full-ce-only-rgb.short-v2t}} | {{msrvtt-train-full-ce-only-rgb.params}} | [config]({{msrvtt-train-full-ce-only-rgb.config}}), [model]({{msrvtt-train-full-ce-only-rgb.model}}), [log]({{msrvtt-train-full-ce-only-rgb.log}}) |
| Prev. + Scene    | v2t  | {{msrvtt-train-full-ce-only-rgb-scene.short-v2t}} | {{msrvtt-train-full-ce-only-rgb-scene.params}} | [config]({{msrvtt-train-full-ce-only-rgb-scene.config}}), [model]({{msrvtt-train-full-ce-only-rgb-scene.model}}), [log]({{msrvtt-train-full-ce-only-rgb-scene.log}}) |
| Prev. + Flow    | v2t  | {{msrvtt-train-full-ce-only-rgb-scene-flow.short-v2t}} | {{msrvtt-train-full-ce-only-rgb-scene-flow.params}} | [config]({{msrvtt-train-full-ce-only-rgb-scene-flow.config}}), [model]({{msrvtt-train-full-ce-only-rgb-scene-flow.model}}), [log]({{msrvtt-train-full-ce-only-rgb-scene-flow.log}}) |
| Prev. + Audio    | v2t  | {{msrvtt-train-full-ce-only-rgb-scene-flow-audio.short-v2t}} | {{msrvtt-train-full-ce-only-rgb-scene-flow-audio.params}} | [config]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio.config}}), [model]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio.model}}), [log]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio.log}}) |
| Prev. + OCR    | v2t  | {{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr.short-v2t}} | {{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr.params}} | [config]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr.config}}), [model]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr.model}}), [log]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr.log}}) |
| Prev. + Speech    | v2t  | {{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr-speech.short-v2t}} | {{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr-speech.params}} | [config]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr-speech.config}}), [model]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr-speech.model}}), [log]({{msrvtt-train-full-ce-only-rgb-scene-flow-audio-ocr-speech.log}}) |
| Prev. + Face    | v2t  | {{msrvtt-train-full-ce.short-v2t}} | {{msrvtt-train-full-ce.params}} | [config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log]({{msrvtt-train-full-ce.log}}) |

**Training with more captions:** Rather than varying the number of experts, we can also investigate how performance changes as we change the number of training captions available per-video.

| Experts | Caps. | Task | R@1 | R@5 | R@10 | MdR | Params | Links |
| -----   | :------: | :--: | :-: | :-: | :--: | :-: | :----: | :---: |
| RGB   | 1 | t2v | {{msrvtt-train-full-ce-ablation-restrict-captions-only-rgb.short-t2v}} | {{msrvtt-train-full-ce-ablation-restrict-captions-only-rgb.params}} | [config]({{msrvtt-train-full-ce-ablation-restrict-captions-only-rgb.config}}), [model]({{msrvtt-train-full-ce-ablation-restrict-captions-only-rgb.model}}), [log]({{msrvtt-train-full-ce-ablation-restrict-captions-only-rgb.log}}) |
| RGB   | 20 | t2v  | {{msrvtt-train-full-ce-only-rgb.short-t2v}} | {{msrvtt-train-full-ce-only-rgb.params}} | [config]({{msrvtt-train-full-ce-only-rgb.config}}), [model]({{msrvtt-train-full-ce-only-rgb.model}}), [log]({{msrvtt-train-full-ce-only-rgb.log}}) |
| All   | 1 | t2v | {{msrvtt-train-full-ce-ablation-restrict-captions.short-t2v}} | {{msrvtt-train-full-ce-ablation-restrict-captions.params}} | [config]({{msrvtt-train-full-ce-ablation-restrict-captions.config}}), [model]({{msrvtt-train-full-ce-ablation-restrict-captions.model}}), [log]({{msrvtt-train-full-ce-ablation-restrict-captions.log}}) |
| All   | 20 | t2v | {{msrvtt-train-full-ce.short-t2v}} | {{msrvtt-train-full-ce.params}} | [config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log]({{msrvtt-train-full-ce.log}}) |
| RGB   | 1 | v2t | {{msrvtt-train-full-ce-ablation-restrict-captions-only-rgb.short-v2t}} | {{msrvtt-train-full-ce-ablation-restrict-captions-only-rgb.params}} | [config]({{msrvtt-train-full-ce-ablation-restrict-captions-only-rgb.config}}), [model]({{msrvtt-train-full-ce-ablation-restrict-captions-only-rgb.model}}), [log]({{msrvtt-train-full-ce-ablation-restrict-captions-only-rgb.log}}) |
| RGB   | 20 | v2t  | {{msrvtt-train-full-ce-only-rgb.short-v2t}} | {{msrvtt-train-full-ce-only-rgb.params}} | [config]({{msrvtt-train-full-ce-only-rgb.config}}), [model]({{msrvtt-train-full-ce-only-rgb.model}}), [log]({{msrvtt-train-full-ce-only-rgb.log}}) |
| All   | 1 | v2t | {{msrvtt-train-full-ce-ablation-restrict-captions.short-v2t}} | {{msrvtt-train-full-ce-ablation-restrict-captions.params}} | [config]({{msrvtt-train-full-ce-ablation-restrict-captions.config}}), [model]({{msrvtt-train-full-ce-ablation-restrict-captions.model}}), [log]({{msrvtt-train-full-ce-ablation-restrict-captions.log}}) |
| All   | 20 | v2t | {{msrvtt-train-full-ce.short-v2t}} | {{msrvtt-train-full-ce.params}} | [config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log]({{msrvtt-train-full-ce.log}}) |

### Expert Zoo

For each dataset, the Collaborative Experts model makes use of a collection of pretrained "expert" feature extractors (see [1] for more precise descriptions). Some experts have been obtained from other sources (described where applicable), rather than extracted by us.  To reproduce the experiments listed above, the experts for each dataset have been bundled into compressed tar files.  These can be downloaded and unpacked with a [utility script](misc/sync_experts.py) (recommended -- see example usage below), which will store them in the locations expected by the training code. Each set of experts has a brief README, which also provides a link from which they can be downloaded directly.

  | Dataset           | Experts  |  Details and links | Archive size | sha1sum |
 |:-------------:|:-----:|:----:|:---:|:---:|
| MSRVTT | audio, face, flow, ocr, rgb, scene, speech | [README](misc/datasets/msrvtt/README.md)| 19.6 GiB | <sup><sub><sup><sub>959bda588793ef05f348d16de26da84200c5a469</sub></sup></sub></sup> |
| LSMDC | audio, face, flow, ocr, rgb, scene | [README](misc/datasets/lsmdc/README.md)| 6.1 GiB | <sup><sub><sup><sub>7ce018e981752db9e793e449c2ba5bc88217373d</sub></sup></sub></sup> |
| MSVD | face, flow, ocr, rgb, scene | [README](misc/datasets/msvd/README.md)| 2.1 GiB | <sup><sub><sup><sub>6071827257c14de455b3a13fe1e885c2a7887c9e</sub></sup></sub></sup> | 
| DiDeMo | audio, face, flow, ocr, rgb, scene, speech | [README](misc/datasets/didemo/README.md)| 2.3 GiB | <sup><sub><sup><sub>6fd4bcc68c1611052de2499fd8ab3f488c7c195b</sub></sup></sub></sup> | 
| ActivityNet | audio, face, flow, ocr, rgb, scene, speech | [README](misc/datasets/activity-net/README.md)| 3.8 GiB | <sup><sub><sup><sub>b16685576c97cdec2783fb89ea30ca7d17abb021</sub></sup></sub></sup> | 

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
# fetch the pretrained experts for MSVD 
python misc/sync_experts.py --dataset MSVD

# find the name of a pretrained model using the links in the tables above 
export MODEL=data/models/msvd-train-full-ce/07-25_15-18-17/trained_model.pth

# create a local directory and download the model into it 
mkdir -p $(dirname "${MODEL}")
wget --output-document="${MODEL}" "http://www.robots.ox.ac.uk/~vgg/research/collaborative-experts/${MODEL}"

# Evaluate the model
python3 test.py --config configs/msvd/eval-full-ce.json --resume ${MODEL} --device 0
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
# fetch the pretrained experts for LSMDC 
python misc/sync_experts.py --dataset LSMDC

# Train the model
python3 train.py --config configs/lsmdc/train-full-ce.json --device 0
```

### Visualising the retrieval ranking

Tensorboard lacks video support via HTML5 tags (at the time of writing) so after each evaluation of a retrieval model, a simple HTML file is generated to allow the predicted rankings of different videos to be visualised: an example screenshot is given below (this tool is inspired by the visualiser in the [pix2pix codebase](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)). To view the visualisation, navigate to the `web directory` (this is generated for each experiment, and will be printed in the log during training) and run `python3 -m http.server 9999`, then navigate to `localhost:9999` in your web browser.  You should see something like the following:

![visualisation](figs/vis-ranking.png)

Note that the visualising the results in this manner requires that you also download the source videos for each of the datasets to some directory <src-video-dir>. Then set the `visualizer.args.src_video_dir` attribute of the training `config.json` file to point to <src-video-dir>.


### Dependencies

If you have enough disk space, the recommended approach to installing the dependencies for this project is to create a conda enviroment via the `requirements/conda-requirements.txt`:

```
conda env create -f requirements/conda-freeze.yml
```

Otherwise, if you'd prefer to take a leaner approach, you can either:
1. `pip/conda install` each missing package each time you hit an `ImportError`
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

This work was inspired by a number of prior works for learning joint embeddings of text and video, but in particular the *Mixture-of-Embedding-Experts* method proposed by Antoine Miech, Ivan Laptev and Josef Sivic ([paper](https://arxiv.org/abs/1804.02516), [code](https://github.com/antoine77340/Mixture-of-Embedding-Experts)). We would also like to thank Zak Stone and Susie Lim for their help with using Cloud TPUs.  The code structure uses the [pytorch-template](https://github.com/victoresque/pytorch-template) by @victoresque.