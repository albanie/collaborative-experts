This repo provides code:
- TeachText which leverages complementary cues from multiple text encoders to provide an enhanced supervisory signal to the retrieval model using a generalize distillation setup ([paper](http://arxiv.org/abs/2104.08271), [project page](https://www.robots.ox.ac.uk/~vgg/research/teachtext/))
- Learning and evaluating joint video-text embeddings for the task of video retrieval. The approach is described in the paper "Use What You Have: Video retrieval using representations from collaborative experts" ([paper](https://arxiv.org/abs/1907.13487), [project page](https://www.robots.ox.ac.uk/~vgg/research/collaborative-experts/))
- CVPR 2020 Pentathlon challenge

**Requirements:** The code assumes PyTorch 1.4 and Python 3.7 (other versions may work, but have not been tested).  See the section on dependencies towards the end of this file for specific package requirements.

### TeachText

![TeachText diagram](figs/TeachText_method.jpg)

**TeachText results on MSRVTT Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{msrvtt-train-full-ce.geomt2v}} | [config_TT]({{msrvtt-train-full-ce.config_TT}}), [model_TT]({{msrvtt-train-full-ce.model_TT}}), [log_TT]({{msrvtt-train-full-ce.log_TT}}) |
| CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-adam.geomt2v}} | [config_TT]({{msrvtt-train-gpt2-xl-finetuned-adam.config_TT}}), [model_TT]({{msrvtt-train-gpt2-xl-finetuned-adam.model_TT}}), [log_TT]({{msrvtt-train-gpt2-xl-finetuned-adam.log_TT}}) |
| TeachText - CE    | Full  | t2v  | {{msrvtt-train-ce-intra-mte.geomt2v}} | [config_TT]({{msrvtt-train-ce-intra-mte.config_TT}}), [model_TT]({{msrvtt-train-ce-intra-mte.model_TT}}), [log_TT]({{msrvtt-train-ce-intra-mte.log_TT}}) |
| TeachText - CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-mte-adam.geomt2v}} | [config_TT]({{msrvtt-train-gpt2-xl-finetuned-mte-adam.config_TT}}), [model_TT]({{msrvtt-train-gpt2-xl-finetuned-mte-adam.model_TT}}), [log_TT]({{msrvtt-train-gpt2-xl-finetuned-mte-adam.log_TT}}) |

Please note that the numbers are higher than in the original CE due to compression artefacts correction

**Denoising results on MSRVTT**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-denoising-adam.geomt2v}} | [config_TT]({{msrvtt-train-gpt2-xl-finetuned-denoising-adam.config_TT}}), [model_TT]({{msrvtt-train-gpt2-xl-finetuned-denoising-adam.model_TT}}), [log_TT]({{msrvtt-train-gpt2-xl-finetuned-denoising-adam.log_TT}}) |
| TeachText - CE+    | Full  | t2v  | {{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.geomt2v}} | [config_TT]({{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.config_TT}}), [model_TT]({{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.model_TT}}), [log_TT]({{msrvtt-train-gpt2-xl-finetuned-mte-denoising-adam.log_TT}}) |

**TeachText results on MSVD Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{msvd-train-full-ce.geomt2v}} | [config_TT]({{msvd-train-full-ce.config_TT}}), [model_TT]({{msvd-train-full-ce.model_TT}}), [log_TT]({{msvd-train-full-ce.log_TT}}) |
| CE+    | Full  | t2v  | {{msvd-train-gpt2-xl-finetuned-adam.geomt2v}} | [config_TT]({{msvd-train-gpt2-xl-finetuned-adam.config_TT}}), [model_TT]({{msvd-train-gpt2-xl-finetuned-adam.model_TT}}), [log_TT]({{msvd-train-gpt2-xl-finetuned-adam.log_TT}}) |
| TeachText - CE    | Full  | t2v  | {{msvd-train-ce-intra-mte.geomt2v}} | [config_TT]({{msvd-train-ce-intra-mte.config_TT}}), [model_TT]({{msvd-train-ce-intra-mte.model_TT}}), [log_TT]({{msvd-train-ce-intra-mte.log_TT}}) |
| TeachText - CE+    | Full  | t2v  | {{msvd-train-gpt2-xl-finetuned-mte-adam.geomt2v}} | [config_TT]({{msvd-train-gpt2-xl-finetuned-mte-adam.config_TT}}), [model_TT]({{msvd-train-gpt2-xl-finetuned-mte-adam.model_TT}}), [log_TT]({{msvd-train-gpt2-xl-finetuned-mte-adam.log_TT}}) |

**Denoising results on MSVD**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE+    | Full  | t2v  | {{msvd-train-gpt2-xl-finetuned-denoising-adam.geomt2v}} | [config_TT]({{msvd-train-gpt2-xl-finetuned-denoising-adam.config_TT}}), [model_TT]({{msvd-train-gpt2-xl-finetuned-denoising-adam.model_TT}}), [log_TT]({{msvd-train-gpt2-xl-finetuned-denoising-adam.log_TT}}) |
| TeachText - CE+    | Full  | t2v  | {{msvd-train-gpt2-xl-finetuned-mte-denoising-adam.geomt2v}} | [config_TT]({{msvd-train-gpt2-xl-finetuned-mte-denoising-adam.config_TT}}), [model_TT]({{msvd-train-gpt2-xl-finetuned-mte-denoising-adam.model_TT}}), [log_TT]({{msvd-train-gpt2-xl-finetuned-mte-denoising-adam.log_TT}}) |

**TeachText results on DiDeMo Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{didemo-train-full-ce.geomt2v}} | [config_TT]({{didemo-train-full-ce.config_TT}}), [model_TT]({{didemo-train-full-ce.model_TT}}), [log_TT]({{didemo-train-full-ce.log_TT}}) |
| CE+    | Full  | t2v  | {{didemo-train-gpt2-xl-finetuned-adam.geomt2v}} | [config_TT]({{didemo-train-gpt2-xl-finetuned-adam.config_TT}}), [model_TT]({{didemo-train-gpt2-xl-finetuned-adam.model_TT}}), [log_TT]({{didemo-train-gpt2-xl-finetuned-adam.log_TT}}) |
| TeachText - CE    | Full  | t2v  | {{didemo-train-ce-intra-mte.geomt2v}} | [config_TT]({{didemo-train-ce-intra-mte.config_TT}}), [model_TT]({{didemo-train-ce-intra-mte.model_TT}}), [log_TT]({{didemo-train-ce-intra-mte.log_TT}}) |
| TeachText - CE+    | Full  | t2v  | {{didemo-train-gpt2-xl-finetuned-mte-adam.geomt2v}} | [config_TT]({{didemo-train-gpt2-xl-finetuned-mte-adam.config_TT}}), [model_TT]({{didemo-train-gpt2-xl-finetuned-mte-adam.model_TT}}), [log_TT]({{didemo-train-gpt2-xl-finetuned-mte-adam.log_TT}}) |

**TeachText results on LSMDC Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{lsmdc-train-full-ce.geomt2v}} | [config_TT]({{lsmdc-train-full-ce.config_TT}}), [model_TT]({{lsmdc-train-full-ce.model_TT}}), [log_TT]({{lsmdc-train-full-ce.log_TT}}) |
| CE+    | Full  | t2v  | {{lsmdc-train-gpt2-xl-finetuned-adam.geomt2v}} | [config_TT]({{lsmdc-train-gpt2-xl-finetuned-adam.config_TT}}), [model_TT]({{lsmdc-train-gpt2-xl-finetuned-adam.model_TT}}), [log_TT]({{lsmdc-train-gpt2-xl-finetuned-adam.log_TT}}) |
| TeachText - CE    | Full  | t2v  | {{lsmdc-train-ce-intra-mte.geomt2v}} | [config_TT]({{lsmdc-train-ce-intra-mte.config_TT}}), [model_TT]({{lsmdc-train-ce-intra-mte.model_TT}}), [log_TT]({{lsmdc-train-ce-intra-mte.log_TT}}) |
| TeachText - CE+    | Full  | t2v  | {{lsmdc-train-gpt2-xl-finetuned-mte-adam.geomt2v}} | [config_TT]({{lsmdc-train-gpt2-xl-finetuned-mte-adam.config_TT}}), [model_TT]({{lsmdc-train-gpt2-xl-finetuned-mte-adam.model_TT}}), [log_TT]({{lsmdc-train-gpt2-xl-finetuned-mte-adam.log_TT}}) |

**TeachText results on Activity-Net Benchmark**

| Model | Split | Task | R@1 | R@5 | R@10 | R@50 | MdR | MnR | Geom | Links |
| ----- | ------| ---- | --- | --- | ---- | ---- | --- | --- | --- | ----- |
| CE    | Full  | t2v  | {{activity-net-train-full-ce.geomt2v}} | [config_TT]({{activity-net-train-full-ce.config_TT}}), [model_TT]({{activity-net-train-full-ce.model_TT}}), [log_TT]({{activity-net-train-full-ce.log_TT}}) |
| CE+    | Full  | t2v  | {{activity-net-train-gpt2-xl-finetuned-adam.geomt2v}} | [config_TT]({{activity-net-train-gpt2-xl-finetuned-adam.config_TT}}), [model_TT]({{activity-net-train-gpt2-xl-finetuned-adam.model_TT}}), [log_TT]({{activity-net-train-gpt2-xl-finetuned-adam.log_TT}}) |
| TeachText - CE    | Full  | t2v  | {{activity-net-train-ce-intra-mte.geomt2v}} | [config_TT]({{activity-net-train-ce-intra-mte.config_TT}}), [model_TT]({{activity-net-train-ce-intra-mte.model_TT}}), [log_TT]({{activity-net-train-ce-intra-mte.log_TT}}) |
| TeachText - CE+    | Full  | t2v  | {{activity-net-train-gpt2-xl-finetuned-mte-adam.geomt2v}} | [config_TT]({{activity-net-train-gpt2-xl-finetuned-mte-adam.config_TT}}), [model_TT]({{activity-net-train-gpt2-xl-finetuned-mte-adam.model_TT}}), [log_TT]({{activity-net-train-gpt2-xl-finetuned-mte-adam.log_TT}}) |

You can download the high quality features used for TeachText from:

```
For MSRVTT:
http:/www.robots.ox.ac.uk/~vgg/research/teachtext/data-hq/high-quality/high-quality-MSRVTT-experts.tar.gz
sha1sum: 734650c3b98509996da75cdedc12101836624917

For MSVD:
http:/www.robots.ox.ac.uk/~vgg/research/teachtext/data-hq/high-quality/high-quality-MSVD-experts.tar.gz
sha1sum: c8eba8c5291dd6bb501757ed0cc327cd22217965

For DiDeMo:
http:/www.robots.ox.ac.uk/~vgg/research/teachtext/data-hq/high-quality/high-quality-DiDeMo-experts.tar.gz
sha1sum: 8e128309f12cf3260fe538f82578b5ad91a46bd0

For ActivityNet:
http:/www.robots.ox.ac.uk/~vgg/research/teachtext/data-hq/high-quality/high-quality-activity-net-experts.tar.gz
sha1sum: 2f3c7c2fe86bd6d0c6230464a940c429291a4012

```

