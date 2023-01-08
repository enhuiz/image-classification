import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(paths, args):
    dfs = []

    for path in paths:
        with open(path, "r") as f:
            text = f.read()

        rows = []

        pattern = r"(\{.+?\})"

        for row in re.findall(pattern, text, re.DOTALL):
            try:
                row = json.loads(row)
            except Exception as e:
                continue

            if "global_step" in row:
                rows.append(row)

        df = pd.DataFrame(rows)

        if "name" in df:
            df["name"] = df["name"].fillna("train")
        else:
            df["name"] = "train"

        df["group"] = str(path.parents[args.group_level])
        df["group"] = df["group"] + "/" + df["name"]

        dfs.append(df)

    df = pd.concat(dfs)

    for group_name, gdf in df.groupby("group"):
        for y in args.ys:
            gdf = gdf.sort_values("global_step")
            print(gdf)

            gdf.plot(
                x="global_step",
                y=y,
                label=f"{group_name}/{y}",
                ax=plt.gca(),
                marker="x" if len(gdf) < 100 else None,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ys", nargs="+")
    parser.add_argument("--log-dir", default="logs", type=Path)
    parser.add_argument("--out-path", default="out.png")
    parser.add_argument("--filename", default="log.txt")
    parser.add_argument("--max-x", default=None)
    parser.add_argument("--group-level", default=1)
    args = parser.parse_args()

    paths = args.log_dir.rglob(f"**/{args.filename}")
    plot(paths, args)
    plt.savefig(args.out_path)


if __name__ == "__main__":
    main()
