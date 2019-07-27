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
The gzipped form of the file is <TODO> GiB.

### References:

If you use these features, please consider citing:
```
[1] Yang Liu, Samuel Albanie, Arsha Nagrani and Andrew Zisserman,
"Use What You Have: Video retrieval using representations from collaborative experts"
British Machine Vision Conference, 2019
```

The original MSVD dataset was described in:

```
[2] Chen, David L., and William B. Dolan.
"Collecting highly parallel data for paraphrase evaluation."
In Proceedings of the 49th Annual Meeting of the ACL: Human Language Technologies
Volume 1, pp. 190-200. ACL, 2011.
```

The train/val/test splits we also used in (and possibly produced by)

```
Venugopalan, S., Rohrbach, M., Donahue, J., Mooney, R., Darrell, T. and Saenko, K., 2015.
"Sequence to sequence-video to text".
In Proceedings of the IEEE international conference on computer vision (pp. 4534-4542).
```