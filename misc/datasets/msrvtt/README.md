## Pretrained Experts

This folder contains a collection of features, extracted from the MSRVTT [3] dataset as part of the paper:
*Use what you have: Video retrieval using representations from collaborative experts*.

For more details on the specific models used to compute the features, please see [1] for descriptions, or the [code repo](https://github.com/albanie/collaborative-experts).   With the kind permission of [Antoine Miech](https://www.di.ens.fr/~miech/) we also include some features made publicly available as part of the release of [2] (these features listed below). These features are required to reproduce some of the experiments in [1].

### Training splits

Prior work has used several different training/test splits on MSRVTT. These splits
are described in the paper [1] as `1k-A`, `1k-B` and `Full`.

The `1k-A` split was produced by the authors of JSFusion [4].  The train/val splits are listed in the files:

1. [train_list_jsfusion.txt](train_list_jsfusion.txt) (9000 videos) and [val_list_jsfusion.txt](val_list_jsfusion.txt) (1000 videos)

The `1k-B` split was produced by the authors of MoEE [2].  The train/test splits are listed in the files:

1. [train_list_miech.txt](train_list_miech.txt) (6656 videos) and [test_list_miech.txt](test_list_miech.txt) (1000 videos)

The `Full` split was produced by the authors of MSRVTT [3].  The train/val/test splits are listed in the files:

1. [train_list_full.txt](train_list_full.txt) (6513 videos), [val_list_full.txt](val_list_full.txt) (497 videos) and [test_list_full.txt](test_list_full.txt) (2990 videos).

**Label Noise**

It is worth being aware that there is a reasonable degree of label noise in the MSRVTT dataset (for instance, captions which are duplicated across videos).

**Tar contents**

The high quality features used for TeachText can be downloaded from:


```
http:/www.robots.ox.ac.uk/~vgg/research/teachtext/data-hq/high-quality/high-quality-MSRVTT-experts.tar.gz
sha1sum: 311529eeb17527a728a1b930f64415dc15a11298

```

For Collaborative Experts please download the following files:

```
http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/features-v2/MSRVTT-experts.tar.gz
sha1sum: 9018a9bf215e174c982b997fd4ee140eb0865040
```

A list of the contents of the tar file are given in [tar_include.txt](tar_include.txt).

[**Deprecated**] *The features made available with the previous code release are also available as a compressed tar file (19.6 GiB). These should be considered deprecated, since they are incompatible with the current codebase, but are still available and can be downloaded from:*

```
deprecated features: http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data-deprecated/features/MSRVTT-experts.tar.gz
```

**Features from MoEE [2]**

The specific features shared by Antoine Miech, Ivan Laptev and Josef Sivic are:

```
resnet_features.pickle
audio_features.pickle
face_features.pickle
flow_features.pickle
w2v_MSRVTT.pickle
```

The original versions of these features can be obtained at:
`https://www.rocq.inria.fr/cluster-willow/amiech/ECCV18/data.zip`

### References:

[1] If you use these features, please consider citing:
```
@inproceedings{Liu2019a,
  author    = {Liu, Y. and Albanie, S. and Nagrani, A. and Zisserman, A.},
  booktitle = {British Machine Vision Conference},
  title     = {Use What You Have: Video retrieval using representations from collaborative experts},
  date      = {2019},
}
```

[2] If you make use of the features shared by Antoine Miech and his coauthors, please cite:


```
@article{miech2018learning,
  title={Learning a text-video embedding from incomplete and heterogeneous data},
  author={Miech, Antoine and Laptev, Ivan and Sivic, Josef},
  journal={arXiv preprint arXiv:1804.02516},
  year={2018}
}
```

[3] Please also consider citing the original MSRVTT dataset, which was described in:

```
@inproceedings{xu2016msr,
  title={Msr-vtt: A large video description dataset for bridging video and language},
  author={Xu, Jun and Mei, Tao and Yao, Ting and Rui, Yong},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5288--5296},
  year={2016}
}
```

[4] The JSFusion method was described in:

```
@inproceedings{yu2018joint,
  title={A joint sequence fusion model for video question answering and retrieval},
  author={Yu, Youngjae and Kim, Jongseok and Kim, Gunhee},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={471--487},
  year={2018}
}
```
