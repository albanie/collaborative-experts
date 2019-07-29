## Pretrained Experts

This folder contains a collection of features, extracted from the LSMDC [3] dataset as part of the paper:
*Use what you have: Video retrieval using representations from collaborative experts* [1].

For more details on the specific models used to compute the features, please see [1] for descriptions, or the [code repo](https://github.com/albanie/collaborative-experts).   With the kind permission of [Antoine Miech](https://www.di.ens.fr/~miech/) we also include some features made publicly available as part of the release of [2] (these features listed below). These features are required to reproduce some of the experiments in [1].

### Training splits

The training splits used in this work were produced as part of the LSMDC challenge and are included in the tarred file:

The train/test splits are listed in the files:

* `LSMDC16_annos_training.csv` (101079 videos)
* `LSMDC16_challenge_1000_publictect.csv` (1000 videos)

**Tar contents**

A list of the contents of the tar file are given in [tar_include.txt](tar_include.txt).
The gzipped form of the file is 6.1 GiB.

**Features from MoEE [2]**

The specific features and files shared by Antoine Miech, Ivan Laptev and Josef Sivic are:

```
X_resnet.npy
X_flow.npy
X_face.npy
resnet-qcm.npy
w2v_LSMDC_qcm.npy
X_audio_test.npy
flow-qcm.npy
face-qcm.npy
w2v_LSMDC.npy
X_audio_train.npy
resnet152-retrieval.npy.tensor.npy
flow-retrieval.npy.tensor.npy
face-retrieval.npy.tensor.npy
w2v_LSMDC_retrieval.npy
X_audio_retrieval.npy.tensor.npy
multiple_choice_gt.txt
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

[3] Please also consider citing the original LSMDC dataset, which was described in:

```
@inproceedings{rohrbach2015dataset,
  title={A dataset for movie description},
  author={Rohrbach, Anna and Rohrbach, Marcus and Tandon, Niket and Schiele, Bernt},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={3202--3212},
  year={2015}
}
```