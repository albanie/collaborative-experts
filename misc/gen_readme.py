"""A small utility for filling in the README paths to experiment artifacts.

The template contains tags of the form {{filetype.experiment_name}}, which are then
replaced with the urls for each resource.
"""
import re
import json
import argparse
from pathlib import Path
from itertools import zip_longest


def generate_url(root_url, target, exp_name, experiments):
    path_store = {
        "log": {"parent": "log", "fname": "info.log"},
        "config": {"parent": "models", "fname": "config.json"},
        "model": {"parent": "models", "fname": "trained_model.pth"}
    }
    paths = path_store[target]
    timestamp = experiments[exp_name]
    return Path(root_url) / paths["parent"] / exp_name / timestamp / paths["fname"]


def generate_readme(expertiments_path, readme_template, root_url, readme_dest):
    with open(expertiments_path, "r") as f:
        experiments = json.load(f)

    with open(readme_template, "r") as f:
        readme = f.read().splitlines()

    generated = []
    for row in readme:
        edits = []
        regex = r"\{\{(.*?)\}\}"
        for match in re.finditer(regex, row):
            groups = match.groups()
            assert len(groups) == 1, "expected single group"
            exp_name, target = groups[0].split(".")
            url = generate_url(root_url, target, exp_name, experiments=experiments)
            edits.append((match.span(), str(url)))
        if edits:
            # invert the spans
            spans = [(None, 0)] + [x[0] for x in edits] + [(len(row), None)]
            inverse_spans = [(x[1], y[0]) for x, y in zip(spans, spans[1:])]
            tokens = [row[start:stop] for start, stop in inverse_spans]
            urls = [str(x[1]) for x in edits]
            new_row = ""
            for token, url in zip_longest(tokens, urls, fillvalue=""):
                new_row += token + url
            row = new_row

        generated.append(row)

    with open(readme_dest, "w") as f:
        f.write("\n".join(generated))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expertiments_path", default="misc/experiments.json")
    parser.add_argument("--readme_template", default="misc/README-template.md")
    parser.add_argument("--readme_dest", default="README.md")
    parser.add_argument("--root_url",
                        default="www.robots.ox.ac.uk/~albanie/data/collaborative-experts")
    args = parser.parse_args()

    generate_readme(
        root_url=args.root_url,
        readme_template=args.readme_template,
        readme_dest=args.readme_dest,
        expertiments_path=args.expertiments_path,
    )
