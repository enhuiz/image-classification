import argparse
from pathlib import Path

import pandas as pd


def _display(df, max, dup, name):
    df = df.sort_values(max, ascending=False)
    df = df.drop_duplicates(dup)
    df[name] = df["value"]
    del df["value"]
    assert isinstance(df, pd.DataFrame)
    print(df.to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs", type=Path)
    parser.add_argument("--filename", default="top1_acc.txt")
    args = parser.parse_args()

    paths = args.log_dir.rglob(f"**/{args.filename}")

    rows = []
    for path in paths:
        try:
            with open(path, "r") as f:
                value = float(f.read())
            rows.append(
                dict(
                    path=path.parents[2],
                    split=path.parents[0].stem,
                    step=int(path.parents[1].stem),
                    value=value,
                )
            )
        except:
            pass

    df = pd.DataFrame(rows)

    for split, grp in df.groupby("split"):
        print(split)
        _display(grp, max="value", dup=["path"], name=args.filename.split(".")[0])


if __name__ == "__main__":
    main()
