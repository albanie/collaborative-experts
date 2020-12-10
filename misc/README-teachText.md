This repo provides code:
- TeachText which leverages complementary cues from multiple text encoders to provide an enhanced supervisory signal to the retrieval model using a generalize distillation setup.
- Learning and evaluating joint video-text embeddings for the task of video retrieval. The approach is described in the paper "Use What You Have: Video retrieval using representations from collaborative experts" ([paper](https://arxiv.org/abs/1907.13487), [project page](https://www.robots.ox.ac.uk/~vgg/research/collaborative-experts/))
- CVPR 2020 Pentathlon challenge

**Requirements:** The code assumes PyTorch 1.4 and Python 3.7 (other versions may work, but have not been tested).  See the section on dependencies towards the end of this file for specific package requirements.

### TeachText

![TeachText diagram](figs/TeachText_method.jpg)

**TeachText results on MSRVTT Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{msrvtt-train-full-ce.t2v}} | [config]({{msrvtt-train-full-ce.config}}), [model]({{msrvtt-train-full-ce.model}}), [log]({{msrvtt-train-full-ce.log}}) |
| CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-adam.t2v}} | [config]({{msrvtt-train-gpt2-xl-finetuned-adam.config}}), [model]({{msrvtt-train-gpt2-xl-finetuned-adam.model}}), [log]({{msrvtt-train-gpt2-xl-finetuned-adam.log}}) |
| TeachText - CE    | Full  | t2v  | {{msrvtt-train-ce-intra-mte.t2v}} | [config]({{msrvtt-train-ce-intra-mte.config}}), [model]({{msrvtt-train-ce-intra-mte.model}}), [log]({{msrvtt-train-ce-intra-mte.log}}) |
| TeachText - CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-mte-adam.t2v}} | [config]({{msrvtt-train-gpt2-xl-finetuned-mte-adam.config}}), [model]({{msrvtt-train-gpt2-xl-finetuned-mte-adam.model}}), [log]({{msrvtt-train-gpt2-xl-finetuned-mte-adam.log}}) |

**Denoising results on MSRVTT**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-denoising-adam.t2v}} | [config]({{msrvtt-train-gpt2-xl-finetuned-denoising-adam.config}}), [model]({{msrvtt-train-gpt2-xl-finetuned-denoising-adam.model}}), [log]({{msrvtt-train-gpt2-xl-finetuned-denoising-adam.log}}) |
| TeachText - CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.t2v}} | [config]({{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.config}}), [model]({{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.model}}), [log]({{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.log}}) |

**TeachText results on MSVD Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{msvd-train-full-ce.t2v}} | [config]({{msvd-train-full-ce.config}}), [model]({{msvd-train-full-ce.model}}), [log]({{msvd-train-full-ce.log}}) |
| CE+    | Full  | t2v  | {{msvd-train-gpt2-xl-finetuned-adam.t2v}} | [config]({{msvd-train-gpt2-xl-finetuned-adam.config}}), [model]({{msvd-train-gpt2-xl-finetuned-adam.model}}), [log]({{msvd-train-gpt2-xl-finetuned-adam.log}}) |
| TeachText - CE    | Full  | t2v  | {{msvd-train-ce-intra-mte.t2v}} | [config]({{msvd-train-ce-intra-mte.config}}), [model]({{msvd-train-ce-intra-mte.model}}), [log]({{msvd-train-ce-intra-mte.log}}) |
| TeachText - CE+    | Full  | t2v  | {{msvd-train-gpt2-xl-finetuned-mte-adam.t2v}} | [config]({{msvd-train-gpt2-xl-finetuned-mte-adam.config}}), [model]({{msvd-train-gpt2-xl-finetuned-mte-adam.model}}), [log]({{msvd-train-gpt2-xl-finetuned-mte-adam.log}}) |

**Denoising results on MSVD**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE+    | Full  | t2v  | {{msvd-train-gpt2-xl-finetuned-denoising-adam.t2v}} | [config]({{msvd-train-gpt2-xl-finetuned-denoising-adam.config}}), [model]({{msvd-train-gpt2-xl-finetuned-denoising-adam.model}}), [log]({{msvd-train-gpt2-xl-finetuned-denoising-adam.log}}) |
| TeachText - CE+    | Full  | t2v  | {{msvd-train-gpt2-xl-finetuned-mte-denoising-adam.t2v}} | [config]({{msvd-train-gpt2-xl-finetuned-mte-denoising-adam.config}}), [model]({{msvd-train-gpt2-xl-finetuned-mte-denoising-adam.model}}), [log]({{msvd-train-gpt2-xl-finetuned-mte-denoising-adam.log}}) |

**TeachText results on DiDeMo Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{didemo-train-full-ce.t2v}} | [config]({{didemo-train-full-ce.config}}), [model]({{didemo-train-full-ce.model}}), [log]({{didemo-train-full-ce.log}}) |
| CE+    | Full  | t2v  | {{didemo-train-gpt2-xl-finetuned-adam.t2v}} | [config]({{didemo-train-gpt2-xl-finetuned-adam.config}}), [model]({{didemo-train-gpt2-xl-finetuned-adam.model}}), [log]({{didemo-train-gpt2-xl-finetuned-adam.log}}) |
| TeachText - CE    | Full  | t2v  | {{didemo-train-ce-intra-mte.t2v}} | [config]({{didemo-train-ce-intra-mte.config}}), [model]({{didemo-train-ce-intra-mte.model}}), [log]({{didemo-train-ce-intra-mte.log}}) |
| TeachText - CE+    | Full  | t2v  | {{didemo-train-gpt2-xl-finetuned-mte-adam.t2v}} | [config]({{didemo-train-gpt2-xl-finetuned-mte-adam.config}}), [model]({{didemo-train-gpt2-xl-finetuned-mte-adam.model}}), [log]({{didemo-train-gpt2-xl-finetuned-mte-adam.log}}) |

**TeachText results on LSMDC Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{lsmdc-train-full-ce.t2v}} | [config]({{lsmdc-train-full-ce.config}}), [model]({{lsmdc-train-full-ce.model}}), [log]({{lsmdc-train-full-ce.log}}) |
| CE+    | Full  | t2v  | {{lsmdc-train-gpt2-xl-finetuned-adam.t2v}} | [config]({{lsmdc-train-gpt2-xl-finetuned-adam.config}}), [model]({{lsmdc-train-gpt2-xl-finetuned-adam.model}}), [log]({{lsmdc-train-gpt2-xl-finetuned-adam.log}}) |
| TeachText - CE    | Full  | t2v  | {{lsmdc-train-ce-intra-mte.t2v}} | [config]({{lsmdc-train-ce-intra-mte.config}}), [model]({{lsmdc-train-ce-intra-mte.model}}), [log]({{lsmdc-train-ce-intra-mte.log}}) |
| TeachText - CE+    | Full  | t2v  | {{lsmdc-train-gpt2-xl-finetuned-mte-adam.t2v}} | [config]({{lsmdc-train-gpt2-xl-finetuned-mte-adam.config}}), [model]({{lsmdc-train-gpt2-xl-finetuned-mte-adam.model}}), [log]({{lsmdc-train-gpt2-xl-finetuned-mte-adam.log}}) |

**TeachText results on Activity-Net Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{activity-net-train-full-ce.t2v}} | [config]({{activity-net-train-full-ce.config}}), [model]({{activity-net-train-full-ce.model}}), [log]({{activity-net-train-full-ce.log}}) |
| CE+    | Full  | t2v  | {{activity-net-train-gpt2-xl-finetuned-adam.t2v}} | [config]({{activity-net-train-gpt2-xl-finetuned-adam.config}}), [model]({{activity-net-train-gpt2-xl-finetuned-adam.model}}), [log]({{activity-net-train-gpt2-xl-finetuned-adam.log}}) |
| TeachText - CE    | Full  | t2v  | {{activity-net-train-ce-intra-mte.t2v}} | [config]({{activity-net-train-ce-intra-mte.config}}), [model]({{activity-net-train-ce-intra-mte.model}}), [log]({{activity-net-train-ce-intra-mte.log}}) |
| TeachText - CE+    | Full  | t2v  | {{activity-net-train-gpt2-xl-finetuned-mte-adam.t2v}} | [config]({{activity-net-train-gpt2-xl-finetuned-mte-adam.config}}), [model]({{activity-net-train-gpt2-xl-finetuned-mte-adam.model}}), [log]({{activity-net-train-gpt2-xl-finetuned-mte-adam.log}}) |


