import functools
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from data_loader.MSRVTT_dataset import MSRVTT
from data_loader.MSVD_dataset import MSVD
from utils.util import HashableDict, HashableOrderedDict
from base import BaseDataLoader


@functools.lru_cache(maxsize=64, typed=False)
def dataset_loader(dataset_name, data_dir, raw_input_dims, num_test_captions, text_dim,
                   feat_aggregation, split_name, text_feat, rgb_model_name):
    print(f"refreshing cache for {dataset_name} data loader")
    if dataset_name == "MSRVTT":
        dataset = MSRVTT(
            data_dir=data_dir,
            text_dim=text_dim,
            text_feat=text_feat,
            split_name=split_name,
            rgb_model_name=rgb_model_name,
            feat_aggregation=feat_aggregation,
            num_test_captions=num_test_captions,
            raw_input_dims=raw_input_dims,
        )
    elif dataset_name == "MSVD":
        dataset = MSVD(
            data_dir=data_dir,
            text_dim=text_dim,
            text_feat=text_feat,
            split_name=split_name,
            rgb_model_name=rgb_model_name,
            feat_aggregation=feat_aggregation,
            num_test_captions=num_test_captions,
            raw_input_dims=raw_input_dims,
        )
    return dataset


class ExpertDataLoader:

    def __init__(self, dataset_name, data_dir, num_workers, batch_size, raw_input_dims,
                 rgb_model_name, split_name, feat_aggregation, num_test_captions,
                 text_feat, text_dim):

        # Ensure that the dictionaries are hashable to allow use of caching
        raw_input_dims = HashableOrderedDict(raw_input_dims)
        feat_aggregation = HashableDict(feat_aggregation)

        dataset = dataset_loader(
            dataset_name=dataset_name,
            data_dir=data_dir,
            text_dim=text_dim,
            rgb_model_name=rgb_model_name,
            text_feat=text_feat,
            split_name=split_name,
            raw_input_dims=raw_input_dims,
            num_test_captions=num_test_captions,
            feat_aggregation=feat_aggregation,
        )
        print("cache info", dataset_loader.cache_info())
        train_loader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=dataset.collate_data,
            drop_last=True,
            shuffle=True,
        )
        self.dataloaders = {
            "train": train_loader,
            "dataaset": dataset,
            "retrieval": dataset.get_retrieval_data(),
        }
        print(f"Loading data loaders with {num_workers} workers")
        self.dataset_name = dataset_name
        self.num_test_captions = num_test_captions

    def __getitem__(self, key):
        return self.dataloaders[key]
