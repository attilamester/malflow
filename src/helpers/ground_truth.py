import os

from core.data.bodmas import Bodmas
from util import config
from util.logger import Logger

config.load_env()


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
    create_ground_truth_bodmas("/opt/work/bd/BODMAS/bodmas_metadata.csv")
