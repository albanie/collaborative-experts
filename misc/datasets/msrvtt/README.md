### Pretrained Experts

This folder contains a collection of features, extracted from the MSRVTT dataset as part of the paper:
*Use what you have: Video retrieval using representations from collaborative experts*.

To help reproduce the results of the paper, it also includes some features made publicly available by [Antoine Miech](https://www.di.ens.fr/~miech/) (these features listed below).
For more details on the specific models used to compute the features, please see [1] for descriptions, or the [code repo](https://github.com/albanie/collaborative-experts).

The specific features shared by Antoine Miech, Ivan Laptev and Josef Sivic are:

```
X_train2014_resnet152.npy
Audio_MSRVTT_new.pickle
Face_MSRVTT_new.pickle
I3D_MSRVTT_new.pickle
resnet_MSRVTT_new.pickle
resnet_MSRVTT_test.pickle
resnet_MSRVTT_train.pickle
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
[2] Miech, A., Laptev, I., & Sivic, J. (2018).
Learning a text-video embedding from incomplete and heterogeneous data.
arXiv preprint arXiv:1804.02516.
```
