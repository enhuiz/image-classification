import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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

    dfs = []

    for path in paths:
        with open(path, "r") as f:
            text = f.read()

        rows = []

        pattern = r"(\{.+?\})"
        for record in re.findall(pattern, text, re.DOTALL):
            if "name" not in record and ("global_step" in record):
                rows.append(json.loads(record))

        df = pd.DataFrame(rows)
        df["group"] = path.parents[args.group_level]

        dfs.append(df)

    df = pd.concat(dfs)

    for group_name, gdf in df.groupby("group"):
        for y in args.ys:
            gdf = gdf.sort_values("global_step")
            gdf.plot(x="global_step", y=y, label=f"{group_name}/{y}", ax=plt.gca())

    plt.savefig(args.out_path)


if __name__ == "__main__":
    main()
