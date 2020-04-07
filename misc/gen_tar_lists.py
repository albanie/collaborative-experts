import argparse
import json
import tqdm
from typeguard import typechecked
from typing import Tuple, Dict, List
from pathlib import Path
from misc.gen_readme import dataset_paths, model_specs2path
from parse_config import ConfigParser


@typechecked
def generate_tar_lists(save_dir: Path, experiments: Dict[str, Tuple[str, str]]):
    all_feat_paths = {}
    for exp_name, (group_id, timestamp) in tqdm.tqdm(experiments.items()):
        rel_path = Path(group_id) / "seed-0" / timestamp / "config.json"
        config_path = Path(save_dir) / "models" / exp_name / rel_path
        with open(config_path, "r") as f:
            config = json.load(f)
        feat_aggregation = config["data_loader"]["args"]["feat_aggregation"]
        dataset_name = exp_name.split("-train")[0]
        if dataset_name not in all_feat_paths:
            all_feat_paths[dataset_name] = set()
        split_names = [config["data_loader"]["args"]["split_name"]]
        if "eval_settings" in config and config["eval_settings"]:
            test_split = config["eval_settings"]["data_loader"]["args"]["split_name"]
            split_names.append(test_split)
        keep = set(config["experts"]["modalities"])
        text_feat = config["experts"]["text_feat"]
        root_feat, paths = dataset_paths(dataset_name)
        modern_feat_agg = {key: val for key, val in feat_aggregation.items()
                           if key in paths["feature_names"]}
        feat_paths = model_specs2path(modern_feat_agg, keep)
        all_feat_paths[dataset_name].update({root_feat / x for x in feat_paths})
        for key, feat_list in paths["custom_paths"].items():
            for feat_path in feat_list:
                all_feat_paths[dataset_name].add(root_feat / feat_path)
        text_paths = [root_feat / rel_path for rel_path in
                      paths["text_feat_paths"][text_feat].values()]
        all_feat_paths[dataset_name].update(set(text_paths))
        all_feat_paths[dataset_name].add(root_feat / paths["raw_captions_path"])
        if "dict_youtube_mapping_path" in paths:
            all_feat_paths[dataset_name].add(
                root_feat / paths["dict_youtube_mapping_path"])
        for split_name in split_names:
            split_paths = set(root_feat / x for x in
                              paths["subset_list_paths"][split_name].values())
            all_feat_paths[dataset_name].update(split_paths)

    for dataset_name, paths in all_feat_paths.items():
        tar_include_list = Path("misc") / "datasets" / dataset_name / "tar_include.txt"
        tar_include_list.parent.mkdir(exist_ok=True, parents=True)
        with open(tar_include_list, "w") as f:
            for path in sorted(paths):
                print(f"Writing {path} to {tar_include_list}")
                f.write(f"{path}\n")


@typechecked
def generate_tar_lists_for_challenge(
        refresh: bool,
        datasets: List[str],
        challenge_phase: str,
        data_dir: Path,
):
    phase_dirs = {
        "public_server_val": "challenge-release-1",
        "public_server_test": "challenge-release-2",
    }
    base = Path("misc/cvpr2020_challenge/datasets")
    challenge_dir = phase_dirs[challenge_phase]
    for dataset in datasets:
        tar_include_list = base / dataset / challenge_dir / "tar_include.txt"
        if tar_include_list.exists() and not refresh:
            print(f"Found existing tar_include_list at {tar_include_list}, skipping")
            continue
        tar_include_list.parent.mkdir(exist_ok=True, parents=True)
        src_folder = data_dir / dataset / challenge_dir
        rel_paths = list(src_folder.glob("**/*"))

        print(f"Found {len(rel_paths)} files in {src_folder}")
        fname = f"data_loader_{dataset.lower()}.json"
        config_path = Path("configs") / "cvpr2020-challenge" / fname
        config = ConfigParser.load_config(config_path)

        keep = set(config["experts"]["modalities"])
        feat_aggregation = config["data_loader"]["args"]["feat_aggregation"]
        _, all_dataset_paths = dataset_paths(dataset=dataset)
        modern_feat_agg = {key: val for key, val in feat_aggregation.items()
                           if key in all_dataset_paths["feature_names"]}
        expected_feat_paths = set(model_specs2path(modern_feat_agg, keep))
        keep = {
            "custom_paths",
            "challenge_text_feat_paths",
            "raw_captions_path",
            "subset_list_paths",
        }
        for key in keep:
            paths = all_dataset_paths[key]
            if isinstance(paths, str):
                expected_feat_paths.add(paths)
            elif isinstance(paths, dict):
                for val in paths.values():
                    if isinstance(val, dict):
                        expected_feat_paths.update(val.values())
                    elif isinstance(val, (str, Path)):
                        expected_feat_paths.add(val)
                    elif isinstance(val, list):
                        expected_feat_paths.update(val)
                    else:
                        raise TypeError(f"Unexpected type: {type(val)}")
            else:
                raise TypeError(f"Unexpected type: {type(paths)}")
        # filter to relevant features
        filtered_rel_paths = []
        for rel_path in rel_paths:
            if any([str(rel_path).endswith(str(x)) for x in expected_feat_paths]):
                filtered_rel_paths.append(rel_path)
        print(f"[{dataset}] Filtered from {len(rel_paths)} to {len(filtered_rel_paths)}")

        with open(tar_include_list, "w") as f:
            print(f"Writing paths to {tar_include_list}")
            for path in sorted(filtered_rel_paths):
                f.write(f"{path}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="data/saved", type=Path)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--experiments_path", default="misc/experiments.json")
    parser.add_argument("--target", default="main",
                        choices=["main", "cvpr2020_challenge"])
    parser.add_argument("--data_dir", type=Path, default="data")
    parser.add_argument("--challenge_phase", default="public_server_val",
                        choices=["public_server_val", "public_server_test"])
    parser.add_argument("--datasets", nargs="+",
                        default=["MSRVTT", "MSVD", "DiDeMo", "LSMDC", "activity-net", 
                                 "YouCook2"])
    args = parser.parse_args()

    with open(args.experiments_path, "r") as f:
        experiments = json.load(f)

    if args.target == "main":
        generate_tar_lists(
            save_dir=args.save_dir,
            experiments=experiments,
        )
    elif args.target == "cvpr2020_challenge":
        generate_tar_lists_for_challenge(
            refresh=args.refresh,
            datasets=args.datasets,
            data_dir=args.data_dir,
            challenge_phase=args.challenge_phase,
        )


if __name__ == "__main__":
    main()
