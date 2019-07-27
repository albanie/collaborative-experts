"""A small utility for transferring features to/from the webserver.
"""
import os
import time
import subprocess
import argparse
from pathlib import Path


def upload_to_server(web_dir, dataset, webserver, root_feat_dir, refresh):
    # NOTE: The compression step will take a long time. The last runs took:
    # MSRVTT -> 00h29m43s, LSMDC -> 0022m21s, MSVD -> 00h03m28s
    # DiDeMo -> 00h04m20s, ActivityNet ->  00h08m00s
    server_dir = Path(web_dir) / "data" / "features"
    subprocess.call(["ssh", webserver, "mkdir -p", str(server_dir)])
    compressed_file = f"{dataset}-experts.tar.gz"
    compressed_path = Path("data") / dataset / "webserver-files" / compressed_file
    if not compressed_path.parent.exists():
        compressed_path.parent.mkdir(exist_ok=True, parents=True)
    tar_include = Path("misc") / "datasets" / dataset.lower() / "tar_include.txt"
    compression_args = (f"tar --dereference --create --verbose"
                        f" --file={str(compressed_path)}"
                        f" --gzip  --files-from={tar_include}")
    print(f"running command {compression_args}")

    if not Path(compressed_path).exists() or refresh["compression"]:
        tic = time.time()
        # TODO(Samuel): Figure out why using subprocess introduces tarring problems
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="MSRVTT",
                        choices=["MSRVTT", "LSMDC", "MSVD", "didemo", "activity-net"])
    parser.add_argument("--action", default="upload", choices=["upload", "fetch"])
    parser.add_argument("--webserver", default="login.robots.ox.ac.uk")
    parser.add_argument("--refresh_compression", action="store_true")
    parser.add_argument("--refresh_server", action="store_true")
    parser.add_argument("--web_dir",
                        default="/projects/vgg/vgg/WWW/research/collaborative-experts")
    args = parser.parse_args()

    server_root_feat_dir = Path("data") / args.dataset / "symlinked-feats"
    refresh_targets = {
        "compression": args.refresh_compression,
        "server": args.refresh_server,
    }

    if args.action == "upload":
        upload_to_server(
            web_dir=args.web_dir,
            dataset=args.dataset,
            webserver=args.webserver,
            root_feat_dir=server_root_feat_dir,
            refresh=refresh_targets,
        )
    elif args.action == "fetch":
        raise NotImplementedError("TODO(Samuel): Implement the fetcher")
        fetch_from_server(
            refresh=refresh,
            web_dir=args.web_dir,
            save_dir=args.save_dir,
            webserver=args.webserver,
            expertiments_path=args.expertiments_path,
        )
    else:
        raise ValueError(f"unknown action: {args.action}")