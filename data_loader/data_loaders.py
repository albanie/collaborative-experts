import functools

import torch
from torch.utils.data import DataLoader
from zsvision.zs_utils import memcache

from utils.util import HashableDict, HashableOrderedDict
from data_loader.MSVD_dataset import MSVD
from data_loader.LSMDC_dataset import LSMDC
from data_loader.DiDeMo_dataset import DiDeMo
from data_loader.MSRVTT_dataset import MSRVTT
from data_loader.ActivityNet_dataset import ActivityNet


@functools.lru_cache(maxsize=64, typed=False)
def dataset_loader(dataset_name, data_dir, raw_input_dims, num_test_captions, text_dim,
                   feat_aggregation, split_name, text_feat, task, text_agg, logger,
                   fuse_captions, max_tokens, spatial_feats, cls_partition,
                   restrict_train_captions, use_zeros_for_missing, text_dropout):
    print(f"refreshing cache for {dataset_name} data loader [{split_name}]")
    kwargs = dict(
        task=task,
        data_dir=data_dir,
        text_dim=text_dim,
        logger=logger,
        text_agg=text_agg,
        text_feat=text_feat,
        max_tokens=max_tokens,
        split_name=split_name,
        cls_partition=cls_partition,
        spatial_feats=spatial_feats,
        text_dropout=text_dropout,
        fuse_captions=fuse_captions,
        raw_input_dims=raw_input_dims,
        feat_aggregation=feat_aggregation,
        num_test_captions=num_test_captions,
        use_zeros_for_missing=use_zeros_for_missing,
        restrict_train_captions=restrict_train_captions,
    )
    if dataset_name == "MSRVTT":
        dataset = MSRVTT(**kwargs)
    elif dataset_name == "LSMDC":
        dataset = LSMDC(**kwargs)
    elif dataset_name == "MSVD":
        dataset = MSVD(**kwargs)
    elif dataset_name == "DiDeMo":
        dataset = DiDeMo(**kwargs)
    elif dataset_name == "ActivityNet":
        dataset = ActivityNet(**kwargs)
    return dataset


class ExpertDataLoader:

    def __init__(self, dataset_name, data_dir, num_workers, batch_size, raw_input_dims,
                 split_name, feat_aggregation, num_test_captions, text_feat, text_dim,
                 fuse_captions, max_tokens, use_zeros_for_missing, task, cls_partitions,
                 trn_cat, text_agg, text_dropout, logger, spatial_feats=False,
                 restrict_train_captions=0, drop_last=False, refresh_lru_cache=False):

        # Ensure that the dictionaries are hashable to allow use of caching
        raw_input_dims = HashableOrderedDict(raw_input_dims)
        feat_aggregation = HashableDict(feat_aggregation)
        max_tokens = HashableDict(max_tokens)

        if refresh_lru_cache:
            logger.info("Explicitly refreshing dataloader and cuda cache")
            dataset_loader.cache_clear()
            torch.cuda.empty_cache()
            memcache.cache_clear()

        if trn_cat:
            raise NotImplementedError(f"Support for trn cat will need to be re-added")

        if "retrieval" in task:
            dataset = dataset_loader(
                dataset_name=dataset_name,
                task=task,
                logger=logger,
                data_dir=data_dir,
                text_dim=text_dim,
                text_agg=text_agg,
                text_feat=text_feat,
                max_tokens=max_tokens,
                text_dropout=text_dropout,
                use_zeros_for_missing=use_zeros_for_missing,
                fuse_captions=fuse_captions,
                spatial_feats=spatial_feats,
                split_name=split_name,
                raw_input_dims=raw_input_dims,
                num_test_captions=num_test_captions,
                feat_aggregation=feat_aggregation,
                restrict_train_captions=restrict_train_captions,
                cls_partition="train",
            )
            x = dataset_loader.cache_info()  # pylint: disable=no-value-for-parameter
            logger.info(f"cache info {x}")
            train_loader = DataLoader(
                dataset=dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=dataset.collate_data,
                drop_last=drop_last,
                shuffle=True,
            )
            self.dataloaders = {"train": train_loader, "dataset": dataset}
            self.dataloaders["retrieval"] = dataset.get_retrieval_data()
        else:
            self.dataloaders = {}
            for cls_partition in cls_partitions:
                cls_dataset = dataset_loader(
                    dataset_name=dataset_name,
                    task=task,
                    logger=logger,
                    data_dir=data_dir,
                    text_dim=text_dim,
                    text_agg=text_agg,
                    text_feat=text_feat,
                    max_tokens=max_tokens,
                    text_dropout=text_dropout,
                    fuse_captions=fuse_captions,
                    spatial_feats=spatial_feats,
                    split_name=split_name,
                    raw_input_dims=raw_input_dims,
                    feat_aggregation=feat_aggregation,
                    num_test_captions=num_test_captions,
                    use_zeros_for_missing=use_zeros_for_missing,
                    restrict_train_captions=restrict_train_captions,
                    cls_partition=cls_partition,
                )
                x = dataset_loader.cache_info()  # pylint: disable=no-value-for-parameter
                logger.info(f"cache info [{cls_partition}] {x}")
                loader = DataLoader(
                    dataset=cls_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=cls_dataset.collate_data,
                    drop_last=False,
                    shuffle=False,
                )
                self.dataloaders[cls_partition] = loader

        logger.info(f"Loading data loaders with {num_workers} workers")
        self.num_test_captions = num_test_captions
        self.dataset_name = dataset_name

    def __getitem__(self, key):
        return self.dataloaders[key]
