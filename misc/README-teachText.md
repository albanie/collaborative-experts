This repo provides code:
- TeachText which leverages complementary cues from multiple text encoders to provide an enhanced supervisory signal to the retrieval model using a generalize distillation setup ([project page](https://www.robots.ox.ac.uk/~vgg/research/teachtext/))
- Learning and evaluating joint video-text embeddings for the task of video retrieval. The approach is described in the paper "Use What You Have: Video retrieval using representations from collaborative experts" ([paper](https://arxiv.org/abs/1907.13487), [project page](https://www.robots.ox.ac.uk/~vgg/research/collaborative-experts/))
- CVPR 2020 Pentathlon challenge

**Requirements:** The code assumes PyTorch 1.4 and Python 3.7 (other versions may work, but have not been tested).  See the section on dependencies towards the end of this file for specific package requirements.

### TeachText

![TeachText diagram](figs/TeachText_method.jpg)

**TeachText results on MSRVTT Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{msrvtt-train-full-ce.geomt2v}} | [config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log_TT]({{msrvtt-train-full-ce.log2}}) |
| CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-adam.geomt2v}} | [config]({{msrvtt-train-gpt2-xl-finetuned-adam.config}}), [model]({{msrvtt-train-gpt2-xl-finetuned-adam.model}}), [log_TT]({{msrvtt-train-gpt2-xl-finetuned-adam.log2}}) |
| TeachText - CE    | Full  | t2v  | {{msrvtt-train-ce-intra-mte.geomt2v}} | [config]({{msrvtt-train-ce-intra-mte.config}}), [model]({{msrvtt-train-ce-intra-mte.model}}), [log_TT]({{msrvtt-train-ce-intra-mte.log2}}) |
| TeachText - CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-mte-adam.geomt2v}} | [config]({{msrvtt-train-gpt2-xl-finetuned-mte-adam.config}}), [model]({{msrvtt-train-gpt2-xl-finetuned-mte-adam.model}}), [log_TT]({{msrvtt-train-gpt2-xl-finetuned-mte-adam.log2}}) |

Please note that the numbers are higher than in the original CE due to compression artefacts correction

**Denoising results on MSRVTT**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-denoising-adam.geomt2v}} | [config]({{msrvtt-train-gpt2-xl-finetuned-denoising-adam.config}}), [model]({{msrvtt-train-gpt2-xl-finetuned-denoising-adam.model}}), [log_TT]({{msrvtt-train-gpt2-xl-finetuned-denoising-adam.log2}}) |
| TeachText - CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.geomt2v}} | [config]({{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.config}}), [model]({{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.model}}), [log_TT]({{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.log2}}) |

**TeachText results on MSVD Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{msvd-train-full-ce.geomt2v}} | [config]({{msvd-train-full-ce.config}}), [model]({{msvd-train-full-ce.model}}), [log_TT]({{msvd-train-full-ce.log2}}) |
| CE+    | Full  | t2v  | {{msvd-train-gpt2-xl-finetuned-adam.geomt2v}} | [config]({{msvd-train-gpt2-xl-finetuned-adam.config}}), [model]({{msvd-train-gpt2-xl-finetuned-adam.model}}), [log_TT]({{msvd-train-gpt2-xl-finetuned-adam.log2}}) |
| TeachText - CE    | Full  | t2v  | {{msvd-train-ce-intra-mte.geomt2v}} | [config]({{msvd-train-ce-intra-mte.config}}), [model]({{msvd-train-ce-intra-mte.model}}), [log_TT]({{msvd-train-ce-intra-mte.log2}}) |
| TeachText - CE+    | Full  | t2v  | {{msvd-train-gpt2-xl-finetuned-mte-adam.geomt2v}} | [config]({{msvd-train-gpt2-xl-finetuned-mte-adam.config}}), [model]({{msvd-train-gpt2-xl-finetuned-mte-adam.model}}), [log_TT]({{msvd-train-gpt2-xl-finetuned-mte-adam.log2}}) |

**Denoising results on MSVD**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE+    | Full  | t2v  | {{msvd-train-gpt2-xl-finetuned-denoising-adam.geomt2v}} | [config]({{msvd-train-gpt2-xl-finetuned-denoising-adam.config}}), [model]({{msvd-train-gpt2-xl-finetuned-denoising-adam.model}}), [log_TT]({{msvd-train-gpt2-xl-finetuned-denoising-adam.log2}}) |
| TeachText - CE+    | Full  | t2v  | {{msvd-train-gpt2-xl-finetuned-mte-denoising-adam.geomt2v}} | [config]({{msvd-train-gpt2-xl-finetuned-mte-denoising-adam.config}}), [model]({{msvd-train-gpt2-xl-finetuned-mte-denoising-adam.model}}), [log_TT]({{msvd-train-gpt2-xl-finetuned-mte-denoising-adam.log2}}) |

**TeachText results on DiDeMo Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{didemo-train-full-ce.geomt2v}} | [config]({{didemo-train-full-ce.config}}), [model]({{didemo-train-full-ce.model}}), [log_TT]({{didemo-train-full-ce.log2}}) |
| CE+    | Full  | t2v  | {{didemo-train-gpt2-xl-finetuned-adam.geomt2v}} | [config]({{didemo-train-gpt2-xl-finetuned-adam.config}}), [model]({{didemo-train-gpt2-xl-finetuned-adam.model}}), [log_TT]({{didemo-train-gpt2-xl-finetuned-adam.log2}}) |
| TeachText - CE    | Full  | t2v  | {{didemo-train-ce-intra-mte.geomt2v}} | [config]({{didemo-train-ce-intra-mte.config}}), [model]({{didemo-train-ce-intra-mte.model}}), [log_TT]({{didemo-train-ce-intra-mte.log2}}) |
| TeachText - CE+    | Full  | t2v  | {{didemo-train-gpt2-xl-finetuned-mte-adam.geomt2v}} | [config]({{didemo-train-gpt2-xl-finetuned-mte-adam.config}}), [model]({{didemo-train-gpt2-xl-finetuned-mte-adam.model}}), [log_TT]({{didemo-train-gpt2-xl-finetuned-mte-adam.log2}}) |

**TeachText results on LSMDC Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{lsmdc-train-full-ce.geomt2v}} | [config]({{lsmdc-train-full-ce.config}}), [model]({{lsmdc-train-full-ce.model}}), [log_TT]({{lsmdc-train-full-ce.log2}}) |
| CE+    | Full  | t2v  | {{lsmdc-train-gpt2-xl-finetuned-adam.geomt2v}} | [config]({{lsmdc-train-gpt2-xl-finetuned-adam.config}}), [model]({{lsmdc-train-gpt2-xl-finetuned-adam.model}}), [log_TT]({{lsmdc-train-gpt2-xl-finetuned-adam.log2}}) |
| TeachText - CE    | Full  | t2v  | {{lsmdc-train-ce-intra-mte.geomt2v}} | [config]({{lsmdc-train-ce-intra-mte.config}}), [model]({{lsmdc-train-ce-intra-mte.model}}), [log_TT]({{lsmdc-train-ce-intra-mte.log2}}) |
| TeachText - CE+    | Full  | t2v  | {{lsmdc-train-gpt2-xl-finetuned-mte-adam.geomt2v}} | [config]({{lsmdc-train-gpt2-xl-finetuned-mte-adam.config}}), [model]({{lsmdc-train-gpt2-xl-finetuned-mte-adam.model}}), [log_TT]({{lsmdc-train-gpt2-xl-finetuned-mte-adam.log2}}) |

**TeachText results on Activity-Net Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{activity-net-train-full-ce.geomt2v}} | [config]({{activity-net-train-full-ce.config}}), [model]({{activity-net-train-full-ce.model}}), [log_TT]({{activity-net-train-full-ce.log2}}) |
| CE+    | Full  | t2v  | {{activity-net-train-gpt2-xl-finetuned-adam.geomt2v}} | [config]({{activity-net-train-gpt2-xl-finetuned-adam.config}}), [model]({{activity-net-train-gpt2-xl-finetuned-adam.model}}), [log_TT]({{activity-net-train-gpt2-xl-finetuned-adam.log2}}) |
| TeachText - CE    | Full  | t2v  | {{activity-net-train-ce-intra-mte.geomt2v}} | [config]({{activity-net-train-ce-intra-mte.config}}), [model]({{activity-net-train-ce-intra-mte.model}}), [log_TT]({{activity-net-train-ce-intra-mte.log2}}) |
| TeachText - CE+    | Full  | t2v  | {{activity-net-train-gpt2-xl-finetuned-mte-adam.geomt2v}} | [config]({{activity-net-train-gpt2-xl-finetuned-mte-adam.config}}), [model]({{activity-net-train-gpt2-xl-finetuned-mte-adam.model}}), [log_TT]({{activity-net-train-gpt2-xl-finetuned-mte-adam.log2}}) |


