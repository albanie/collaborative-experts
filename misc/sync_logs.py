"""A small utility for transferring training logs to the webserver.
"""
import json
import subprocess
import argparse
from pathlib import Path


def sync_files(expertiments_path, save_dir, webserver, webdir):

    filetypes = {
        "log": ["info.log"],
        "models": ["trained_model.pth", "config.json"]
    }
    with open(expertiments_path, "r") as f:
        experiments = json.load(f)
    for key, rel_dir in experiments.items():

        # copy experiment artifacts
        for filetype, fnames in filetypes.items():
            for fname in fnames:
                rel_path = Path(rel_dir) / fname
                local_path = Path(save_dir) / filetype / key / rel_path
                server_path = Path(webdir).expanduser() / filetype / key / rel_path
                dest = f"{webserver}:{str(server_path)}"
                print(f"{key} -> {webserver} [{local_path} -> {server_path}]")
                subprocess.call(["ssh", webserver, "mkdir -p", str(server_path.parent)])
                scp_args = ["scp", str(local_path), dest]
                print(f"running command {' '.join(scp_args)}")
                subprocess.call(scp_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expertiments_path", default="misc/experiments.json")
    parser.add_argument("--save_dir", default="data/saved")
    parser.add_argument("--webserver", default="login.robots.ox.ac.uk")
    parser.add_argument("--web_dir", default="~WWW/data/collaborative-experts")
    args = parser.parse_args()

    sync_files(
        webdir=args.web_dir,
        save_dir=args.save_dir,
        webserver=args.webserver,
        expertiments_path=args.expertiments_path,
    )