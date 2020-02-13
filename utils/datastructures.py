"""This module defines a datastructure for storing pre-computed features for datasets.

It provides key-value access, but is backed by a monolithic array to prevent memory
fragmentation.  This can be useful for loading large feature sets into memory (e.g. 
those that are > 100 GiB) in a manner that minimises OOM issues.
"""

import pickle
import argparse
import numpy as np
import humanize


class ExpertStore:

    def __init__(self, keylist, dim, dtype=np.float16):
        self.keys = keylist
        self.dim = dim
        self.store_dtype = dtype
        self.store = np.zeros((len(keylist), dim), dtype=dtype)
        self.keymap = {}
        self.missing = set()
        self.rebuild_keymap()

    def __setitem__(self, key, value):
        idx = self.keymap[key]
        if isinstance(value, np.ndarray):
            # non-nan values must be vectors of the appropriate size
            assert value.size == self.dim, f"cannot set value with size {value.size}"
        else:
            assert np.isnan(value)
        self.store[idx] = value

    def rebuild_keymap(self):
        for idx, key in enumerate(self.keys):
            self.keymap[key] = idx

    def filter_keys(self, keys, tag, allow_mismatch="", exceptions=None):
        keyset = set(keys)
        missing = keyset - set(self.keys)
        if exceptions is not None and missing:
            excluded = missing.intersection(set(exceptions))
            print(f"filter_keys >>> applying exceptions for {len(excluded)} videos")
            missing = missing - excluded
        print(f"filter_keys >>> {tag}")
        if allow_mismatch and missing:
            print(f"Key mismatch (missing {len(missing)}) {allow_mismatch}")
        else:
            samples = list(missing)[:3]
            msg = f"cannot apply filter since missing {len(missing)} keys e.g. {samples}"
            assert not missing, msg
        keep = np.array([x in keyset for x in self.keys])
        filtered_keys = np.array(self.keys)[keep]
        print(f"Filtering from {len(self.keys)} keys to {len(filtered_keys)} keys")
        self.keys = filtered_keys
        self.store = self.store[keep]
        self.rebuild_keymap()

    def __getitem__(self, key):
        return self.store[self.keymap[key]]

    def __len__(self):
        return len(self.keys)

    def __repr__(self):
        keep_samples = 3
        samples = list(self.keymap.items())[:keep_samples]
        sample_str = "\n".join([f"{key}: {val}" for key, val in samples])
        summary = (
            f"ExpertStore object with {len(self.keys)} features (dim: {self.dim})"
            f" (storage is using {humanize.naturalsize(self.store.nbytes)})"
            f"\nFirst {keep_samples} elements of keymap: \n{sample_str}"
        )
        return summary


def gen_dict_store(keylist, dim):
    store = dict()
    for key in keylist:
        store[key] = np.random.rand(1, dim).astype(np.float16)
    return store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="moments-in-time")
    parser.add_argument("--dim", type=int, default=2048)
    args = parser.parse_args()

    from config import get_data_paths
    data_paths = get_data_paths(args.dataset)
    relevant_path = data_paths["relevant-id-list"]
    with open(relevant_path, "r") as f:
        relevant_ids = sorted(f.read().splitlines())

    for store_name in "dict", "np", "expert_store":
        if store_name == "dict":
            store = gen_dict_store(keylist=relevant_ids, dim=args.dim)
        elif store_name == "np":
            store = np.random.rand(len(relevant_ids), args.dim).astype(np.float16)
        elif store_name == "expert_store":
            store = ExpertStore(keylist=relevant_ids, dim=args.dim)
            print(store)
        serialised = pickle.dumps(store)
        print(f"Memory needs for {store_name}: {humanize.naturalsize(len(serialised))}")



if __name__ == "__main__":
    main()
