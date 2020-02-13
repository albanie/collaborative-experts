## Pretrained Experts

This folder contains a collection of features, extracted from the ActivityNet [2] and ActivityNet-captions [3] datasets as part of the paper:
*Use what you have: Video retrieval using representations from collaborative experts*.

### Training splits

The training splits were taken from [3] and are given in the files linked below:

* [train_list.txt](train_list.txt) (10009 videos)
* [val_1_list.txt](val_1_list.txt) (4917 videos)
* [val_2_list.txt](val_2_list.txt) (4885 videos)

In our work, we use the `train` split for training and the `val_1` split for evaluation (the `val_1` split forms a superset of the `val_2` split, with differing captions).


**Tar contents**

The compressed tar file (3.7 GiB) can be downloaded from:

```
http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/features-v2/activity-net-experts.tar.gz
sha1sum: 2901046fa6a3d6f6393ee0047818e960fcfabd69
```

A list of the contents of the tar file are given in [tar_include.txt](tar_include.txt).

[**Deprecated**] *The features made available with the previous code release are also available as a compressed tar file (3.8 GiB). These should be considered deprecated, since they are incompatible with the current codebase, but are still available and can be downloaded from:*

```
http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data-deprecated/features/activity-net-experts.tar.gz
sha1sum: b16685576c97cdec2783fb89ea30ca7d17abb021
```


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

[2] Please also consider citing the original ActivityNet dataset, which was described in:

```
@inproceedings{caba2015activitynet,
  title={Activitynet: A large-scale video benchmark for human activity understanding},
  author={Caba Heilbron, Fabian and Escorcia, Victor and Ghanem, Bernard and Carlos Niebles, Juan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={961--970},
  year={2015}
}
```

[3] In addition, please consider citing the ActivityNet-captions dataset, which provides the text descriptions, and which was described in:

```
@inproceedings{krishna2017dense,
  title={Dense-captioning events in videos},
  author={Krishna, Ranjay and Hata, Kenji and Ren, Frederic and Fei-Fei, Li and Carlos Niebles, Juan},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={706--715},
  year={2017}
}
```