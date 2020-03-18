"""A small utility for transferring features to/from the webserver.

Example usage:
To fetch features for LSMDC, run the following in the project root folder:
python misc/sync_experts.py --dataset MSRVTT
"""
import os
import time
import subprocess
import argparse
from pathlib import Path


def upload_to_server(web_dir, dataset, webserver, root_feat_dir, refresh):
    server_dir = Path(web_dir) / "data" / "features-v2"
    subprocess.call(["ssh", webserver, "mkdir -p", str(server_dir)])
    compressed_file = f"{dataset}-experts.tar.gz"
    compressed_path = Path("data") / dataset / "webserver-files" / compressed_file
    if not compressed_path.parent.exists():
        compressed_path.parent.mkdir(exist_ok=True, parents=True)
    tar_include = Path("misc") / "datasets" / dataset.lower() / "tar_include.txt"
    if not Path(compressed_path).exists() or refresh["compression"]:
        compression_args = (f"tar --dereference --create --verbose"
                            f" --file={str(compressed_path)}"
                            f" --use-compress-program=pigz"
                            f" --files-from={tar_include}")
        print(f"running command {compression_args}")
        tic = time.time()
        os.system(compression_args)
        duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - tic))
        print(f"Finished compressing features in {duration}")
    else:
        print(f"Found existing compressed file at {compressed_path}, skipping....")

    dest = f"{webserver}:{str(server_dir / compressed_file)}"
    rsync_args = ["rsync", "-av", "--progress", str(compressed_path), dest]
    if not refresh["server"]:
        rsync_args.insert(1, "--ignore-existing")
    tic = time.time()
    subprocess.call(rsync_args)
    duration = time.strftime('%Hh%Mm%Ss', time.gmtime(time.time() - tic))
    print(f"Finished transferring features in {duration}")


def fetch_from_server(dataset, root_url, refresh, purge_tar_file):
    local_data_dir = Path("data") / dataset
    symlinked_feats_dir = local_data_dir / "symlinked-feats"
    if symlinked_feats_dir.exists() and not refresh["symlinked-feats"]:
        print(f"Found symlinked feats at {symlinked_feats_dir}, skipping")
        return

    local_data_dir.mkdir(exist_ok=True, parents=True)
    archive_name = f"{dataset}-experts.tar.gz"
    local_archive = local_data_dir / archive_name
    if not local_archive.exists():
        src_url = f"{root_url}/features-v2/{archive_name}"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MSRVTT",
                        choices=["MSRVTT", "LSMDC", "MSVD", "didemo", "activity-net"])
    parser.add_argument("--action", default="fetch", choices=["upload", "fetch"])
    parser.add_argument("--webserver", default="login.robots.ox.ac.uk")
    parser.add_argument("--refresh_compression", action="store_true")
    parser.add_argument("--refresh_server", action="store_true")
    parser.add_argument("--refresh_symlinked_feats", action="store_true")
    parser.add_argument("--purge_tar_file", action="store_true")
    parser.add_argument("--web_dir",
                        default="/projects/vgg/vgg/WWW/research/collaborative-experts")
    parser.add_argument(
        "--root_url",
        default="http://www.robots.ox.ac.uk/~vgg/research/collaborative-experts/data",
    )
    args = parser.parse_args()

    server_root_feat_dir = Path("data") / args.dataset / "symlinked-feats"
    refresh_targets = {
        "server": args.refresh_server,
        "compression": args.refresh_compression,
        "symlinked-feats": args.refresh_symlinked_feats,
    }

    if args.action == "upload":
        upload_to_server(
            web_dir=args.web_dir,
            dataset=args.dataset,
            refresh=refresh_targets,
            webserver=args.webserver,
            root_feat_dir=server_root_feat_dir,
        )
    elif args.action == "fetch":
        fetch_from_server(
            dataset=args.dataset,
            root_url=args.root_url,
            refresh=refresh_targets,
            purge_tar_file=args.purge_tar_file,
        )
    else:
        raise ValueError(f"unknown action: {args.action}")
