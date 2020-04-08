"""A small utility for transferring features to/from the webserver.

Example usage:
To fetch features for MSRVTT, run the following in the project root folder:
python misc/sync_experts.py --dataset MSRVTT

Note that to fetch the features for LSMDC, you must additionally supply an access
code after filling out the MPII dataset agreement form:
python misc/sync_experts.py --dataset LSMDC --access_code <put-code-here>
"""
import os
import time
import hashlib
import argparse
import subprocess
from typing import Dict
from pathlib import Path

from typeguard import typechecked


@typechecked
def get_archive_name(dataset: str, release: str, archive_type: str) -> str:
    if release.startswith("challenge-release"):
        archive_name = f"{release}-{dataset}-experts.tar.gz"
    else:
        archive_name = f"{dataset}-experts.tar.gz"
    if archive_type == "features":
        pass
    elif archive_type == "videos":
        archive_name = f"{archive_type}-{archive_name}"
    else:
        raise NotImplementedError(f"Unsupported archive type: {archive_type}")
    return archive_name


@typechecked
def upload_to_server(
    web_dir: Path,
    dataset: str,
    release: str,
    webserver: str,
    refresh: Dict[str, bool],
):
    server_dir = web_dir / "data" / release
    subprocess.call(["ssh", webserver, "mkdir -p", str(server_dir)])
    if release.startswith("challenge-release"):
        dataset_dir = Path("misc/cvpr2020_challenge/datasets") / dataset
        # tar_include = dataset_dir / release / "tar_include.txt"
        tar_lists = {
            "features": "tar_include.txt",
            "videos": "video_tar_include.txt",
        }
        tar_includes, compressed_paths = [], []
        for key, tar_list in tar_lists.items():
            tar_includes.append(dataset_dir / release / tar_list)
            compressed_file = get_archive_name(
                dataset=dataset,
                release=release,
                archive_type=key,
            )
            compressed_path = Path(f"data/{dataset}/webserver-files") / compressed_file
            compressed_paths.append(compressed_path)
    else:
        tar_includes = [Path("misc/datasets") / dataset.lower() / "tar_include.txt"]
        compressed_file = get_archive_name(
            dataset=dataset,
            release=release,
            archive_type="features",
        )
        compressed_paths = [Path("data") / dataset / "webserver-files" / compressed_file]

    for tar_include, compressed_path in zip(tar_includes, compressed_paths):
        if not compressed_path.parent.exists():
            compressed_path.parent.mkdir(exist_ok=True, parents=True)
        if not Path(compressed_path).exists() or refresh["compression"]:
            compression_args = (f"tar --dereference --create --verbose"
                                f" --file={str(compressed_path)}"
                                f" --use-compress-program=pigz"
                                f" --files-from={tar_include}")
            print(f"running command {compression_args}")
            tic = time.time()
            os.system(compression_args)
            duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - tic))
            print(f"Finished tar contents features in {duration}")
        else:
            print(f"Found existing compressed file at {compressed_path}, skipping....")

        dest = f"{webserver}:{str(server_dir / compressed_path.name)}"
        rsync_args = ["rsync", "-av", "--progress", str(compressed_path), dest]
        if not refresh["server"]:
            rsync_args.insert(1, "--ignore-existing")
        tic = time.time()
        subprocess.call(rsync_args)
        duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - tic))
        print(f"Finished transferring tar file in {duration}")


@typechecked
def fetch_from_server(
        dataset: str,
        root_url: str,
        purge_tar_file: bool,
        release: str,
        refresh: Dict[str, bool],
        access_code: str = None,
):
    local_data_dir = Path("data") / dataset
    symlinked_feats_dir = local_data_dir / "symlinked-feats"
    if symlinked_feats_dir.exists() and not refresh["symlinked-feats"]:
        print(f"Found symlinked feats at {symlinked_feats_dir}, skipping")
        return

    local_data_dir.mkdir(exist_ok=True, parents=True)
    archive_name = get_archive_name(dataset, release=release, archive_type="features")
    local_archive = local_data_dir / archive_name
    if not local_archive.exists():
        if access_code:
            access_hash = hashlib.sha256(access_code.encode("utf-8")).hexdigest()[:10]
            archive_name = f"{access_hash}-{archive_name}"
        src_url = f"{root_url}/{release}/{archive_name}"
        wget_args = ["wget", f"--output-document={str(local_archive)}", src_url]
        print(f"running command: {' '.join(wget_args)}")
        subprocess.call(wget_args)
    else:
        print(f"found archive at {local_archive}, skipping...")

    # unpack the archive and optionally clean up
    untar_args = ["tar", "-xvf", str(local_archive)]
    subprocess.call(untar_args)
    if purge_tar_file:
        local_archive.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", nargs="+",
                        default=["MSRVTT", "MSVD", "DiDeMo", "activity-net", "YouCook2"],
                        choices=["LSMDC", "MSRVTT", "MSVD", "DiDeMo", "activity-net",
                                 "YouCook2"])
    parser.add_argument("--action", default="fetch", choices=["upload", "fetch"])
    parser.add_argument("--webserver", default="login.robots.ox.ac.uk")
    parser.add_argument("--refresh_compression", action="store_true")
    parser.add_argument("--refresh_server", action="store_true")
    parser.add_argument("--refresh_symlinked_feats", action="store_true")
    parser.add_argument("--purge_tar_file", action="store_true")
    parser.add_argument("--release", default="features-v2",
                        choices=["features-v2", "challenge-release-1",
                                 "challenge-release-2"],
                        help=("The features to fetch (features-v2 refers to the features"
                              " that can be used to reproduce the collaborative experts"
                              "paper"))
    parser.add_argument("--access_code", help="Code to access LSMDC")
    parser.add_argument("--web_dir", type=Path,
                        default="/projects/vgg/vgg/WWW/research/collaborative-experts")
    parser.add_argument(
        "--root_url",
        default="http://www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data",
    )
    args = parser.parse_args()

    refresh_targets = {
        "server": args.refresh_server,
        "compression": args.refresh_compression,
        "symlinked-feats": args.refresh_symlinked_feats,
    }

    for dataset in args.dataset:
        if dataset == "LSMDC":
            msg = ("To download LSMDC, you must obtain an access code (please see "
                   "README.md for details")
            assert args.access_code, msg
        if args.action == "upload":
            upload_to_server(
                web_dir=args.web_dir,
                dataset=dataset,
                refresh=refresh_targets,
                webserver=args.webserver,
                release=args.release,
            )
        elif args.action == "fetch":
            fetch_from_server(
                dataset=dataset,
                release=args.release,
                root_url=args.root_url,
                refresh=refresh_targets,
                purge_tar_file=args.purge_tar_file,
                access_code=args.access_code,
            )
        else:
            raise ValueError(f"unknown action: {args.action}")

if __name__ == "__main__":
    main()
