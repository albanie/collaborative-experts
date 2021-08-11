"""Prepare a collection of text embeddings of video descriptions for datasets.

ipy misc/prepare_text_embeddings.py \
    -- --datasets ActivityNetSegments --embedding_name all-but-slow \
    --yaspify --refresh

ipy misc/prepare_text_embeddings.py \
    -- --datasets DiDeMoSegments --embedding_name w2v --refresh

ipy misc/prepare_text_embeddings.py \
    -- --datasets YouDescribeSegments --embedding_name howto100m_mil_nce

"""
import os
import sys
import json
import pickle
import socket
import argparse
import itertools
from typing import Dict, List, Tuple, Union
from pathlib import Path
from collections import defaultdict

import ray
import tqdm
import torch
import numpy as np
from typeguard import typechecked
from zsvision.zs_utils import BlockTimer, memcache

from model.text import (
    GrOVLE,
    W2VEmbedding,
    TextEmbedding,
    HowTo100M_MIL_NCE,
    HuggingFaceWrapper
)
from utils.util import filter_cmd_args
from yaspi.yaspi import Yaspi
from misc.gen_readme import dataset_paths


@typechecked
def prepare_embedding_model(
        embedding_name: str,
        text_embedding_config: Dict[str, Dict[str, Union[str, int]]],
) -> TextEmbedding:
    conf = text_embedding_config[embedding_name]
    for key in conf:
        if key.endswith("_path"):
            conf[key] = Path(conf[key])
    cls_map = defaultdict(lambda: HuggingFaceWrapper)
    cls_map.update({
        "w2v": W2VEmbedding,
        "grovle": GrOVLE,
        "mt_grovle": GrOVLE,
        "hglmm_300d": GrOVLE,
        "hglmm_6kd": GrOVLE,
        "howto100m_mil_nce": HowTo100M_MIL_NCE,
    })
    return cls_map[embedding_name](embedding_name=embedding_name, **conf)


@typechecked
def validate_embeddings_against_reference(
    computed_embeddings: Dict[str, List[np.ndarray]],
    embedding_name: str,
    dataset: str,
):
    root_feat, paths = dataset_paths(dataset)
    reference_dict = {}
    for path in paths["text_feat_paths"][embedding_name].values():
        reference_dict.update(memcache(root_feat / path))

    # We handle MSVD as a special case, because video keys != feature keys
    if dataset == "MSVD":
        key_map = memcache(root_feat / paths["dict_youtube_mapping_path"])
        inverse_map = {val: key for key, val in key_map.items()}
        reference_dict = {inverse_map[key]: val for key, val in reference_dict.items()}

    print(f"Validating embeddings against reference....")
    for key, val in tqdm.tqdm(computed_embeddings.items()):
        ref_val = reference_dict[key]
        msg = (f"[{embedding_name}] {key} Different number of "
               f"embeddings {len(ref_val)} vs {len(val)}")
        assert len(ref_val) == len(val), msg
        msg = f"[{embedding_name}] Embedding mismatch for {key}"
        for vec, ref_vec in zip(val, ref_val):
            assert np.abs(vec - ref_vec).max() < 1E-5, msg


def extract_embeddings_for_video(
        descriptions: List[str],
        key: str,
        model,
) -> Tuple[List[np.ndarray], List[str]]:
    embeddings_for_video, failed_tokens = [], []
    for description in descriptions:
        msg = (f"Expected descripton to be a list of string tokens, "
               f" but was {type(description)} instead for {key}")
        assert isinstance(description, List), msg
        assert isinstance(description[0], str), msg
        description_str = " ".join(description)
        embedded, failed = model.text2vec(description_str)
        embeddings_for_video.append(embedded)
        failed_tokens.extend(failed)
    return embeddings_for_video, failed_tokens


@typechecked
def extract_embeddings(
    text_embedding_config_path: Path,
    rel_dest_dir: Path,
    data_dir: Path,
    refresh: bool,
    validate_embeddings: bool,
    limit: int,
    processes: int,
    embedding_name: str,
    datasets: List[str],
):
    for dataset in datasets:
        dest_dir = data_dir / dataset / rel_dest_dir
        dest_name = embedding_name
        if limit:
            dest_name = f"{embedding_name}-limit{limit}"
        dest_path = dest_dir / f"{dest_name}.pkl"

        if dest_path.exists() and not refresh:
            print(f"Found existing text embeddings at {dest_path}, skipping....")
            return

        dest_dir.mkdir(exist_ok=True, parents=True)
        # handle the activity-net exception
        if dataset == "activity-net":
            fname = "raw-captions-train-val_1.pkl"
        else:
            fname = "raw-captions.pkl"
        captions_path = data_dir / dataset / "structured-symlinks" / fname
        video_descriptions = memcache(captions_path)
        with open(text_embedding_config_path, "r") as f:
            text_embedding_config = json.load(f)

        force_cpu = text_embedding_config[embedding_name].pop("force_cpu", False)
        dev_name = "cuda:0" if torch.cuda.device_count() > 0 and not force_cpu else "cpu"
        device = torch.device(dev_name)

        model = prepare_embedding_model(embedding_name, text_embedding_config)
        model.set_device(device)
        if limit:
            keep = set(list(video_descriptions.keys())[:limit])
            video_descriptions = {key: val for key, val in video_descriptions.items()
                                  if key in keep}

        computed_embeddings = {}
        kwarg_list = []
        for key, descriptions in tqdm.tqdm(video_descriptions.items()):
            kwarg_list.append({"key": key, "descriptions": descriptions})

        all_failed_tokens = []
        func = extract_embeddings_for_video
        if processes > 1:
            # Note: An experimental approach with Ray.  Unfortunately, it seems that
            # the overhead is too great to justify this approach (it's slower than
            # using a single process). TODO(Samuel): revisit.
            func = ray.remote(extract_embeddings_for_video)
            ray.init(num_cpus=processes)

            # Store model in shared memory object store to avoid multiple copies
            model_id = ray.put(model)

            def to_iterator(obj_ids):
                while obj_ids:
                    done, obj_ids = ray.wait(obj_ids)
                    yield ray.get(done[0])

            result_ids = [func.remote(model=model_id, **kwargs) for kwargs in kwarg_list]
            zipped = zip(to_iterator(result_ids), kwarg_list)
            for (embeddings, failed), kwargs in tqdm.tqdm(zipped, total=len(result_ids)):
                computed_embeddings[kwargs["key"]] = embeddings
                all_failed_tokens.extend(failed)
        else:
            for kwargs in tqdm.tqdm(kwarg_list):
                embeddings_for_video, failed_tokens = func(**kwargs, model=model)
                computed_embeddings[kwargs["key"]] = embeddings_for_video
                all_failed_tokens.extend(failed_tokens)

        stats = [len(x) for sublist in computed_embeddings.values() for x in sublist]
        print(f"Average num embedding tokens: {np.mean(stats):.1f} tokens")
        fail_rate = len(all_failed_tokens) / np.sum(stats)
        stat_str = f"{len(all_failed_tokens)}/{np.sum(stats)} [{100 * fail_rate:.1f}%]"
        print(f"Failed tokens: {stat_str} tokens")

        if validate_embeddings:
            validate_embeddings_against_reference(
                computed_embeddings=computed_embeddings,
                embedding_name=embedding_name,
                dataset=dataset,
            )
        with BlockTimer(f"Writing embeddings to {dest_path}"):
            with open(dest_path, "wb") as f:
                pickle.dump(computed_embeddings, f)


@typechecked
def prepare_text_with_yaspi(
        yaspi_defaults: Dict[str, Union[str, int]],
        common_kwargs: Dict,
        datasets: List[str],
        embedding_names: List[str],
):
    cmd_args = sys.argv
    remove = ["--yaspify", "--datasets", "--embedding_name"]
    cmd_args = filter_cmd_args(cmd_args, remove=remove)
    base_cmd = f"python {' '.join(cmd_args)} --slurm"
    # avoid filename limit
    embedding_acronyms = []
    for embedding_name in embedding_names:
        acronym = "".join([x[0].upper() for x in embedding_name.split("-")])
        embedding_acronyms.append(acronym)

    job_name = f"prepare-text-{'-'.join(datasets)}-{'-'.join(embedding_acronyms)}"
    pairs = list(itertools.product(embedding_names, datasets))
    job_queue = []
    for embedding_name, dataset in pairs:
        job_queue.append(f'"--embedding_name {embedding_name} --datasets {dataset}"')
    job_queue = " ".join(job_queue)
    job = Yaspi(
        cmd=base_cmd,
        job_queue=job_queue,
        job_name=job_name,
        job_array_size=len(pairs),
        **yaspi_defaults,
    )
    job.submit(watch=True, conserve_resources=5)
    extract_embeddings(**common_kwargs)


def main():
    parser = argparse.ArgumentParser("Prepare text embeddings for a given dataset")
    parser.add_argument('--refresh', action="store_true")
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--processes', type=int, default=1)
    parser.add_argument("--data_dir", type=Path, default="data",
                        help="the location of the data directory")
    parser.add_argument("--dataset", type=str, default="MSVD",
                        help="the name of the dataset to process")
    parser.add_argument("--datasets", nargs="+",
                        default=["MSRVTT", "MSVD", "DiDeMo", "YouCook2", "activity-net",
                                 "LSMDC", "YouDescribe", "YouDescribeSegments",
                                 "ActivityNetSegments", "DiDeMoSegments"],
                        help="The datasets to prepare text embeddings for")
    parser.add_argument("--rel_dest_dir", type=Path,
                        default="processing/text_embeddings",
                        help="the relative path of destination folder")
    parser.add_argument("--embedding_name", type=str, default="all-but-slow",
                        choices=[
                            "w2v", "bert", "grovle", "mt_grovle",
                            "electra", "howto100m_mil_nce", "hglmm_300d",
                            "hglmm_6kd", "openai-gpt", "gpt2", "gpt2-medium",
                            "gpt2-large", "gpt2-xl", "bert-base-uncased", "t5-small",
                            "t5-base", "t5-large", "t5-3b", "t5-11b", "ctrl",
                            "albert-base-v2", "albert-large-v2", "albert-xlarge-v2",
                            "roberta-base", "roberta-large", "xlnet-base-cased",
                            "xlnet-large-cased", "transfo-xl-wt103", "all",
                            "all-but-slow"
                        ],
                        help="the name of the embedding model to prepare")
    parser.add_argument('--device', default="0", help="indices of GPUs to enable")
    parser.add_argument("--yaspify", action="store_true", help="launch via slurm")
    parser.add_argument("--slurm", action="store_true")
    parser.add_argument("--use_cnodes", action="store_true")
    parser.add_argument("--text_embedding_config_path", type=Path,
                        default="model/text_embedding_models.json",
                        help="the location of the config file containing the model paths")
    parser.add_argument('--validate_embeddings', action="store_true",
                        help="If given, compare the embeddings to a reference set")
    parser.add_argument("--yaspi_defaults_path", type=Path,
                        default="misc/yaspi_gpu_defaults.json")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    common_kwargs = dict(
        refresh=args.refresh,
        limit=args.limit,
        processes=args.processes,
        data_dir=args.data_dir,
        rel_dest_dir=args.rel_dest_dir,
        datasets=args.datasets,
        embedding_name=args.embedding_name,
        text_embedding_config_path=args.text_embedding_config_path,
        validate_embeddings=args.validate_embeddings,
    )

    if args.yaspify:
        slow_models = {
            "ctrl",
            "t5-3b",
            "t5-11b",
            "xlnet-base-cased",
            "xlnet-large-cased",
            "transfo-xl-wt103",
            "roberta-large",
        }
        with open(args.yaspi_defaults_path, "r") as f:
            yaspi_defaults = json.load(f)
        if args.use_cnodes:
            yaspi_defaults.update({"partition": "compute", "gpus_per_task": 0})
        if args.embedding_name in {"all", "all-but-slow"}:
            embedding_names = []
            with open(args.text_embedding_config_path, "r") as f:
                for name, vals in json.load(f).items():
                    if args.embedding_name == "all-but-slow":
                        if name in slow_models:
                            continue
                    if not vals.get("custom_pipeline", False):
                        embedding_names.append(name)
        else:
            embedding_names = [args.embedding_name]

        prepare_text_with_yaspi(
            datasets=args.datasets,
            common_kwargs=common_kwargs,
            yaspi_defaults=yaspi_defaults,
            embedding_names=embedding_names,
        )
    else:
        if args.slurm:
            os.system(str(Path.home() / "configure_tmp_data.sh"))
            print(f"Preparing embeddings via slurm on {socket.gethostname()}")
        extract_embeddings(**common_kwargs)


if __name__ == "__main__":
    main()
