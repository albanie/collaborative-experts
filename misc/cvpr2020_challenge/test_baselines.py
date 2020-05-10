"""Evaluating simple baselines for the CVPR2020 video pentathlon challenge

Example usage (evaluates checkpoints produced by the train_baselines.py script):

# add the project root folder to the pythonpath
export PYTHONPATH=$(pwd):$PYTHONPATH

# provide path to checkpoints (produced by train_baselines.py)
CKPT_LIST=data/cvpr2020-challenge-submissions/ckpts-baselines-2020-04-05_09-53-14-public_server_val-MSRVTT-MSVD-DiDeMo-YouCook2-activity-net.json

python misc/cvpr2020_challenge/test_baselines.py  --ckpt_list_path ${CKPT_LIST}
"""
import json
import argparse
from pathlib import Path
from datetime import datetime

from typeguard import typechecked

from misc.cvpr2020_challenge.prepare_submission import generate_predictions
from misc.cvpr2020_challenge.train_baselines import (
    parse_paths_from_logs,
    launch_and_monitor_cmd
)


@typechecked
def json_key2dataset_name(json_key: str) -> str:
    """Convert json key used for a given datset into the name used in the codebase.

    Args:
        json_key: the key to be converted

    Returns:
        the name of the dataset

    NOTE: This is used to patch around inconsistency in the naming scheme used for
    ActivityNet and every other datsaet.
    """
    dataset_name = {"ActivityNet": "activity-net"}.get(json_key, json_key)
    return dataset_name


@typechecked
def evaluate_from_ckpts(
        ckpt_list_path: Path,
        challenge_config_dir: Path,
        dest_dir: Path,
        device: int,
):
    with open(ckpt_list_path, "r") as f:
        ckpts = json.load(f)

    challenge_phase = "public_server_test"
    timestamp = datetime.now().strftime(r"%Y-%m-%d_%H-%M-%S")

    test_set_prediction_paths = {}
    datasets = []
    for json_key, ckpt_path in ckpts.items():
        dataset = json_key2dataset_name(json_key)
        datasets.append(dataset)
        folder = dataset.lower()
        config_path = challenge_config_dir / folder / "baseline-public-test.json"
        flags = (f" --config {config_path} --device {device} --eval_from_training_config"
                 f" --resume {ckpt_path}")
        cmd = f"python -u test.py {flags}"
        print(f"Launching baseline evaluation for {dataset} with command: {cmd}")
        logs = launch_and_monitor_cmd(cmd=cmd)
        exp_paths = parse_paths_from_logs(logs, queries=["predictions"])
        test_set_prediction_paths[json_key] = {challenge_phase: exp_paths["predictions"]}

    fname = f"baselines-{timestamp}-{challenge_phase}-{'-'.join(datasets)}.json"
    dest_path = dest_dir / f"predictions-{fname}"

    print(f"Writing baseline predicitions list to {dest_path}")
    with open(dest_path, "w") as f:
        json.dump(test_set_prediction_paths, f, indent=4, sort_keys=True)

    generate_predictions(
        refresh=True,
        validate=True,
        dest_dir=dest_dir,
        challenge_phase=challenge_phase,
        predictions_path=dest_path,
    )


def main():
    parser = argparse.ArgumentParser(description="evaluate baselines from ckpts")
    parser.add_argument("--device", type=int, default=0,
                        help="gpu device to use for training")
    parser.add_argument("--ckpt_list_path", type=Path, required=True,
                        help="Path to json file containing model checkpoint locations")
    parser.add_argument('--challenge_config_dir', type=Path,
                        default="configs/cvpr2020-challenge")
    parser.add_argument('--dest_dir', type=Path,
                        default="data/cvpr2020-challenge-submissions")
    args = parser.parse_args()

    evaluate_from_ckpts(
        device=args.device,
        dest_dir=args.dest_dir,
        challenge_config_dir=args.challenge_config_dir,
        ckpt_list_path=args.ckpt_list_path,
    )


if __name__ == "__main__":
    main()
