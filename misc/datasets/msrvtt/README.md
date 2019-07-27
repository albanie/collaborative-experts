## Pretrained Experts

This folder contains a collection of features, extracted from the MSRVTT [3] dataset as part of the paper:
*Use what you have: Video retrieval using representations from collaborative experts*.

To help reproduce the results of the paper, it also includes some features made publicly available by [Antoine Miech](https://www.di.ens.fr/~miech/) (these features listed below).
For more details on the specific models used to compute the features, please see [1] for descriptions, or the [code repo](https://github.com/albanie/collaborative-experts).

### Training splits

Prior work has used several different training/test splits on MSRVTT. These splits
are described in the paper [1] as `1k-A`, `1k-B` and `Full`.

The `1k-A` split was produced by the authors of JSFusion [4].  The train/val splits are listed in the files:

1. [train_list_jsfusion.txt](train_list_jsfusion_.txt) (9000 videos) and [val_list_jsfusion.txt](val_list_jsfusion.txt) (1000 videos)

The `1k-B` split was produced by the authors of MoEE [2].  The train/val splits are listed in the files:

1. [train_list_miech.txt](train_list_miech.txt) (6656 videos) and [val_list_miech.txt](val_list_miech.txt) (1000 videos)

The `Full` split was produced by the authors of MSRVTT [3].  The train/val/test splits are listed in the files:

1. [train_list_full.txt](train_list_dev.txt) (6513 videos), [val_list_full.txt](val_list_full.txt) (497 videos) and [test_list_full.txt](test_list_full.txt) (2990 videos).

**Label Noise**

It is worth being aware that there is a reasonable degree of label noise in the MSRVTT dataset (for instance, captions which are duplicated across videos).

**Tar contents**

A list of the contents of the tar file are given in [tar_include.txt](tar_include.txt).
The gzipped form of the file is 19.6 GiB.

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

If you use these features, please consider citing:
```
[1] Yang Liu, Samuel Albanie, Arsha Nagrani and Andrew Zisserman,
"Use What You Have: Video retrieval using representations from collaborative experts"
British Machine Vision Conference, 2019
```

If you make use of the features shared by Antoine Miech and his coauthors, please cite:
```
[2] Miech, A., Laptev, I., & Sivic, J.
Learning a text-video embedding from incomplete and heterogeneous data.
arXiv preprint arXiv:1804.02516, 2018
```

The original MSRVTT dataset was described in:

```
[3] Xu, J., Mei, T., Yao, T., & Rui, Y.
"Msr-vtt: A large video description dataset for bridging video and language"
CVPR 2016
```

The JSFusion method was described in:

```
[4] Yu, Youngjae, Jongseok Kim, and Gunhee Kim.
"A joint sequence fusion model for video question answering and retrieval."
Proceedings of the European Conference on Computer Vision (ECCV). 2018.
```
