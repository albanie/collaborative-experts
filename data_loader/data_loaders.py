import functools
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import data_loader.MSR_dataset as MSR
from utils.util import hashabledict
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0,
                 num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True,
                                      transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split,
                         num_workers)


@functools.lru_cache(maxsize=64, typed=False)
def dataset_loader(dataset, data_dir, raw_input_dims, num_test_captions, text_dim,
                   feat_aggregation, split_name, text_feat, rgb_model_name):
    print("REFRESHING CACHE")
    if dataset == "MSRVTT":
        res = MSR.MSRVTT_new(
            data_dir=data_dir,
            text_dim=text_dim,
            text_feat=text_feat,
            split_name=split_name,
            rgb_model_name=rgb_model_name,
            feat_aggregation=feat_aggregation,
            num_test_captions=num_test_captions,
            raw_input_dims=raw_input_dims,
        )
    return res


class MSRVTTDataLoader:

    def __init__(self, data_dir, num_workers, batch_size, raw_input_dims,
                 rgb_model_name, split_name, feat_aggregation, num_test_captions,
                 text_feat, text_dim):

        # Ensure that the dictionaries are hashing to allow use of caching
        raw_input_dims = hashabledict(raw_input_dims)
        feat_aggregation = hashabledict(feat_aggregation)

        dataset = dataset_loader(
            dataset="MSRVTT",
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
            "retrieval": dataset.getRetrievalSamples(),
        }
        print("Loading data loaders with {} workers".format(num_workers))
        self.dataset_name = "MSRVTT"
        self.num_test_captions = num_test_captions

    def __getitem__(self, key):
        return self.dataloaders[key]