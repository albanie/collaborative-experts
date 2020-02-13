## Pretrained Experts

This folder contains a collection of features, extracted from the DiDeMo [2] dataset as part of the paper:
*Use what you have: Video retrieval using representations from collaborative experts*.

### Training splits

The training splits were taken from [2] and are given in the files linked below:

* [train_list.txt](train_list.txt) (8392 videos)
* [val_list.txt](val_list.txt) (1065 videos)
* [test_list.txt](test_list.txt) (1004 videos)

NOTE: That in the original paper [2], the training split contained 8395 videos (three are missing in our split).


**Tar contents**

The compressed tar file (1.2 GiB) can be downloaded from:

```
http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data/features-v2/didemo-experts.tar.gz
sha1sum: 19dfa1272e5db57191446a0c4fca59d298cb8663
```

A list of the contents of the tar file are given in [tar_include.txt](tar_include.txt).

[**Deprecated**] *The features made available with the previous code release are also available as a compressed tar file (2.3 GiB). These should be considered deprecated, since they are incompatible with the current codebase, but are still available and can be downloaded from:*

```
deprecated features: http:/www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data-deprecated/features/didemo-experts.tar.gz
sha1sum: 6fd4bcc68c1611052de2499fd8ab3f488c7c195b
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

[2] Please also consider citing the original DiDeMo dataset, which was described in:

```
@inproceedings{anne2017localizing,
  title={Localizing moments in video with natural language},
  author={Anne Hendricks, Lisa and Wang, Oliver and Shechtman, Eli and Sivic, Josef and Darrell, Trevor and Russell, Bryan},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={5803--5812},
  year={2017}
}
```