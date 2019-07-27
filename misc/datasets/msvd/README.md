## Pretrained Experts

This folder contains a collection of features, extracted from the MSVD [2] dataset as part of the paper:
*Use what you have: Video retrieval using representations from collaborative experts*.

### Training splits

The training splits were taken from prior work [3] and are given in the files linked below:

* [train_list.txt](train_list.txt) (1200 videos)
* [val_list.txt](val_list.txt) (100 videos)
* [test_list.txt](test_list.txt) (670 videos)


**Tar contents**

A list of the contents of the tar file are given in [tar_include.txt](tar_include.txt).
The gzipped form of the file is 2.1 GiB.

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

[2] Please also consider citing the original MSVD dataset, which was described in:

```
@inproceedings{chen2011collecting,
  title={Collecting highly parallel data for paraphrase evaluation},
  author={Chen, David L and Dolan, William B},
  booktitle={Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies-Volume 1},
  pages={190--200},
  year={2011},
  organization={Association for Computational Linguistics}
}
```

The train/val/test splits were also used in (and possibly produced by)

```
@inproceedings{venugopalan2015sequence,
  title={Sequence to sequence-video to text},
  author={Venugopalan, Subhashini and Rohrbach, Marcus and Donahue, Jeffrey and Mooney, Raymond and Darrell, Trevor and Saenko, Kate},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={4534--4542},
  year={2015}
}
```