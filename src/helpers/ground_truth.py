import os

from core.data.bodmas import Bodmas
from util import config
from util.logger import Logger


def create_ground_truth_bodmas(bodmas_meta_csv: str):
    import pandas as pd
    config.load_env()

    df = pd.read_csv(bodmas_meta_csv, index_col="sha")
    df["md5"] = ""
    df["sha256"] = ""
    for i, sample in enumerate(Bodmas.get_samples()):
        sha = os.path.basename(sample.filepath).split(".")[0]
        df.loc[sha, "md5"] = sample.md5
        df.loc[sha, "sha256"] = sample.sha256
        if i % 1000 == 0:
            Logger.info(f"Processed {i} samples up to {sha}")

    df.index.rename("filename(original sha256)", inplace=True)
    df.to_csv("./BODMAS_ground_truth.csv")


if __name__ == "__main__":
    create_ground_truth_bodmas("/opt/work/bd/BODMAS/bodmas_metadata.csv")
