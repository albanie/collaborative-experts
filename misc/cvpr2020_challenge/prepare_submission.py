"""This script prepares a submission for the CVPR2020 video pentathlon challenge.

Example usage:
python misc/cvpr2020_challenge/prepare_submission.py --challenge_phase public_server_val
"""

import json
import hashlib
import zipfile
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import humanize
from typing import Dict
from typeguard import typechecked


@typechecked
def get_dataset_num_queries(dataset: str, challenge_phase: str) -> int:
    stats = {
        "max_descriptions_per_video": {
            "MSRVTT": 20,
            "LSMDC": 1,
            "MSVD": 81,
            "DiDeMo": 1,
            "YouCook2": 1,
            "ActivityNet": 1,
        },
        "expected_videos": {
            "MSRVTT": {"public_server_val": 497, "public_server_test": 2990},
            "LSMDC": {"public_server_val": 7408, "public_server_test": 1000},
            "YouCook2": {"public_server_val": 969, "public_server_test": 3310},
            "MSVD": {"public_server_val": 100, "public_server_test": 670},
            "DiDeMo": {"public_server_val": 1065, "public_server_test": 1004},
            "ActivityNet": {"public_server_val": 1001, "public_server_test": 4917}
        },
        "expected_invalid_queries": {
            "MSRVTT": {"public_server_val": 0, "public_server_test": 6},
            "LSMDC": {"public_server_val": 0, "public_server_test": 0},
            "YouCook2": {"public_server_val": 0, "public_server_test": 0},
            "MSVD": {"public_server_val": 3810, "public_server_test": 26507},
            "DiDeMo": {"public_server_val": 0, "public_server_test": 0},
            "ActivityNet": {"public_server_val": 0, "public_server_test": 0},
        },
    }
    num_videos = stats["expected_videos"][dataset][challenge_phase]
    num_queries = num_videos * stats["max_descriptions_per_video"][dataset]
    invalid_queries = stats["expected_invalid_queries"][dataset][challenge_phase]
    return num_queries - invalid_queries


@typechecked
def validate_predictions(
        preds: np.ndarray,
        dataset: str,
        challenge_phase: str,
        topk: int = 10,
):
    shape = preds.shape
    num_queries = get_dataset_num_queries(dataset, challenge_phase=challenge_phase)
    expected = (num_queries, topk)
    msg = f"Expected ranks with shape {expected}, but found {shape} for {dataset}"
    assert shape == expected, msg
    print(f"Found valid rank matrix for {dataset} [shape: {shape}]")


@typechecked
def generate_predictions(
        refresh: bool,
        validate: bool,
        dest_dir: Path,
        challenge_phase: str,
        predictions_path: Path,
):
    """Generate a zip file for a submission in the challenge format."""
    with open(predictions_path, "r") as f:
        predictions = json.load(f)

    # generate a filename from a hash of the prediction locations and the current date
    contents = "".join(sorted([val[challenge_phase] for val in predictions.values()]))
    content_hash = hashlib.sha256(contents.encode("utf-8")).hexdigest()[:10]
    dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dest_name = f"{challenge_phase}-{content_hash}-{dt}.zip"

    dest_path = dest_dir / dest_name
    dest_dir.mkdir(exist_ok=True, parents=True)
    if list(dest_dir.glob(f"{challenge_phase}-{content_hash}*.zip")) and not refresh:
        print(f"Found existing submission for {predictions_path}, skipping")
        return
    with zipfile.ZipFile(dest_path, 'w') as f:
        for ii, (dataset, paths) in enumerate(predictions.items()):
            pred_path = Path(paths[challenge_phase])
            if validate:
                preds = np.loadtxt(pred_path, dtype=int, delimiter=",")
                pred_size = humanize.naturalsize(pred_path.stat().st_size)
                print(f"Validating predictions from [{pred_path}] [{pred_size}]")
                validate_predictions(preds, dataset, challenge_phase=challenge_phase)
            print(f"{ii}/{len(predictions)} Writing predictions for {dataset} "
                  f"from {pred_path} to {dest_path} as {pred_path.name}")
            f.write(pred_path, arcname=pred_path.name)
    zip_size = humanize.naturalsize(dest_path.stat().st_size)
    print(f"Zipfile {dest_path} [{zip_size}] can now be uploaded to CodaLab")


def main():
    parser = argparse.ArgumentParser(description='Prepare challenge submission')
    parser.add_argument("--refresh", action="store_true",
                        help="whether to overwrite existing submission files")
    parser.add_argument("--validate", type=int, default=1,
                        help="whether to validate the submission shapes")
    parser.add_argument("--predictions_path", type=Path,
                        default="misc/cvpr2020-challenge/predictions.json",
                        help="location of file containing paths to predictions")
    parser.add_argument("--dest_dir", type=Path,
                        default="data/cvpr2020-challenge-submissions",
                        help="directory location where submissions will be stored")
    parser.add_argument("--challenge_phase", default="public_server_val",
                        choices=["public_server_val", "public_server_test"],
                        help=("whether the submission is for the validation phase of the"
                              " competition, or the final test phase"))
    args = parser.parse_args()

    generate_predictions(
        refresh=args.refresh,
        validate=bool(args.validate),
        dest_dir=args.dest_dir,
        challenge_phase=args.challenge_phase,
        predictions_path=args.predictions_path,
    )


if __name__ == "__main__":
    main()
