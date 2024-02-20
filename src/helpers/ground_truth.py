import json
import os

import matplotlib.pyplot as plt

from core.data.bodmas import Bodmas
from util import config
from util.logger import Logger

config.load_env()

BODMAS_METADATA_CSV = "/opt/work/bd/BODMAS/bodmas_metadata.csv"


def read_bodmas_metadata(bodmas_meta_csv: str):
    import pandas as pd
    df = pd.read_csv(bodmas_meta_csv, index_col="sha")
    df["md5"] = ""
    df["sha256"] = ""
    return df


def create_ground_truth_bodmas(bodmas_meta_csv: str):
    df = read_bodmas_metadata(bodmas_meta_csv)
    for i, sample in enumerate(Bodmas.get_samples()):
        sha = os.path.basename(sample.filepath).split(".")[0]
        df.loc[sha, "md5"] = sample.md5
        df.loc[sha, "sha256"] = sample.sha256
        if i % 1000 == 0:
            Logger.info(f"Processed {i} samples up to {sha}")

    df.index.rename("filename(original sha256)", inplace=True)
    df.to_csv("./BODMAS_ground_truth.csv")


def get_ground_truth_distribution(bodmas_meta_csv: str):
    df = read_bodmas_metadata(bodmas_meta_csv)
    ds_families = df["family"].dropna().str.lower()
    value_counts = ds_families.value_counts()
    total_samples_having_family = value_counts.sum()
    buff = ""
    for min_freq in [10, 50, 100, 500, 1000]:
        gt_min = value_counts >= min_freq
        having_gt_min = value_counts[gt_min]
        pct = f"{having_gt_min.sum() / total_samples_having_family:.2%}"
        buff += f"{len(having_gt_min):>5} families with at least {min_freq:>5} samples ({having_gt_min.sum()} total samples - {pct})\n"

    plt.title(
        f"BODMAS Family Distribution ({ds_families.nunique()} unique families, {total_samples_having_family} samples)\n{buff}")

    value_counts.plot(kind="barh", figsize=(10, 60), logx=True)
    plt.savefig("BODMAS_ground_truth_family_distribution.pdf", bbox_inches='tight')

    family_dict = {f: i for i, f in enumerate(sorted(list(ds_families.unique())))}
    with open("BODMAS_ground_truth_family_dict.json", "w") as f:
        json.dump(family_dict, f)


def tmp_rename_png_files_to_contain_family(png_dir):
    import pandas as pd
    df = pd.read_csv("./BODMAS_ground_truth.csv")
    df.set_index("md5", inplace=True)
    for i, file in enumerate(os.listdir(png_dir)):
        md5 = file.split("_")[0]
        family = df.loc[md5, "family"]
        new_name = f"{family}_{file}"
        os.rename(os.path.join(png_dir, file), os.path.join(png_dir, new_name))
        if i % 1000 == 0:
            Logger.info(f"Processed {i} files up to {file}")


if __name__ == "__main__":
    # create_ground_truth_bodmas(BODMAS_METADATA_CSV)
    get_ground_truth_distribution(BODMAS_METADATA_CSV)