## Collaborative Experts

This repo provides code for learning and evaluating joint video-text embeddings for the task of video retrieval.

### Requirements

The code assumes PyTorch 1.1 and Python 3.7 (other versions may work, but have not been tested).

### Evaluating pretrained video embeddings

We provide pretrained models for each dataset to reproduce the results reported in the paper.  Each model is accompanied by training and evaluation logs.  

| Benchmark     | Model | Split | R@1 | R@5 | R@10 | MdR | MnR | Links |
|:-------------:| ------| ------| ---:| ---:|-----:|----:|----:|------:|
| MSRVTT        | CE | Full  | - | - | - | - | - | [config](configs/msrvtt/train-full-ce.json), [model](), [log]() |
| MSRVTT        | CE | 1k-A  | - | - | - | - | - | [config](configs/msrvtt/train-jsfusion-ce.json), [model](), [log]() |
| MSRVTT        | CE | 1k-B  | - | - | - | - | - | [config](configs/msrvtt/train-miech-ce.json), [model](), [log]() |
| MSRVTT        | CE* | 1k-B  | - | - | - | - | - | [config](configs/msrvtt/train-miech-miechfeats-ce.json), [model](), [log]() |
| MSRVTT        | MoEE* | 1k-B  | - | - | - | - | - | [config](configs/msrvtt/train-miech-miechfeats-moee.json), [model](), [log]() |

Models marked with * use the features made available with the work of [Miech et al.](https://arxiv.org/abs/1804.02516), unstarred models on the `1k-B` split make additional use of OCR, speech and scene features, as well slightly stronger text encodings (GPT, rather than word2vec - see the paper for details). The MoEE model is included as a sanity check that our codebase approximately reproduces the MoEE paper.

| Benchmark     | Model | Split | R@1 | R@5 | R@10 | MdR | MnR | Links |
|:-------------:| ------| ------| ---:| ---:|-----:|----:|----:|------:|
| LSMDC        | CE | Full  | - | - | - | - | - | [config](configs/msrvtt/train-full-ce.json), [model](), [log]() |

### Training a video embedding

To train a new video embedding, please see the scripts for each dataset.

### Expert Zoo

The links below point the features extracted from each dataset with a collection of pretrained "expert" models (see the paper for more precise descriptions). Some experts have been obtained from other sources (described where applicable), rather than extracted by us.

  | Expert           | Description  |  Feature dim | Links |
 |:-------------:| -----| -----:| ---:|
| RGB appearance | [SENet-154](https://arxiv.org/abs/1709.01507) evaluated on frames at 5ps  | 2048 | [MSR-VTT]() |

### Collaborative Gating module

The structure of the Collaborative Gating module is shown below (please see the [paper](link) for more details).

![CE diagram](figs/CE-diagram.png)

### Visualising the retrieval ranking

Tensorboard lacks video support via HTML5 tags (at the time of writing) so after each evaluation of a retrieval model, a simple HTML file is generated to allow the predicted rankings of different videos to be visualised: an example screenshot is given below (this tool is inspired by the visualiser in the [pix2pix codebase](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)). To view the visualisation, navigate to the `vis_dir` specified in the relevant config file (e.g. [here]()) and run `python3 -m http.server 9999`, then navigate to `localhost:9999` in your web browser.  You should see something like the following:

![visualisation](figs/vis-ranking.png)


### References

If you find this code useful or use the extracted features, please consider citing:

```
@inproceedings{Liu2019a,
  author    = {Liu, Y. and Albanie, S. and Nagrani, A. and Zisserman, A.},
  booktitle = {British Machine Vision Conference},
  title     = {Use What You Have: Video retrieval using representations from collaborative experts},
  date      = {2019},
}
```

### Acknowledgements

This work was inspired by the *Mixture-of-Embedding-Experts* method proposed by Antoine Miech, Ivan Laptev and Josef Sivic ([paper](https://arxiv.org/abs/1804.02516), [code](https://github.com/antoine77340/Mixture-of-Embedding-Experts)). We would also like to thank Zak Stone and Susie Lim for their considerable help with using Cloud TPUs.