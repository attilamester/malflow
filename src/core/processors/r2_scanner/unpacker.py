import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from typing import Type

from core.data import DatasetProvider
from core.data.bodmas import Bodmas, BodmasUnpacked, BodmasArmed
from core.model import CallGraph
from core.model.sample import Sample
from core.processors.r2_scanner.create_dfs import create_callgraph_dfs
from core.processors.r2_scanner.paths import get_path_image
from core.processors.r2_scanner.scan_samples import scan_sample
from core.processors.util import process_samples, decorator_callgraph_processor, decorator_sample_processor
from helpers.ground_truth import BODMAS_GROUND_TRUTH_CSV
from util import config
from util.logger import Logger


def is_sample_packed(sample: Sample):
    dset = Bodmas
    die_info_path = os.path.join(dset.get_dir_info(), "die", os.path.basename(sample.filepath) + ".csv")

    if not os.path.isfile(die_info_path):
        Logger.info(f"No die info found for sample {sample.md5} {sample.sha256}")
        return False, None

    packed = False
    with open(die_info_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("Packer;"):
                tokens = line.split(";")
                packer_name = tokens[1].strip()
                packer_info = tokens[4].replace("Packer:", "").strip()
                return True, packer_name
    if not packed:
        return False, None


ARM_INFO = "/opt/work/bd/BODMAS/arm.txt"
BUFF = ""


def create_csv_for_arming_sample(dset: Type[DatasetProvider], sample: Sample):
    global BUFF

    packed, packer_name = is_sample_packed(sample)
    if not packed:
        return
    Logger.info(f"Sample packed with {packer_name} {sample.filepath}")
    BUFF += f"{sample.filepath}\n"


def unpack_sample(sample: Sample):
    packed, packer_name = is_sample_packed(sample)
    dset_unpacked = BodmasUnpacked
    dest_path = os.path.join(dset_unpacked.get_dir_samples(), f"unpacked_{os.path.basename(sample.filepath)}")

    if not packed:
        return

    if os.path.isfile(dest_path):
        return

    Logger.info(f"Unpacking sample [{packer_name}]: {sample.md5} {sample.sha256} {sample.filepath}")

    packer_name = packer_name.lower()
    upx_args = ["upx", "-d", "-o", dest_path, sample.filepath]
    unipacker_args = ["unipacker", "--dest", dset_unpacked.get_dir_samples(), sample.filepath]
    for unpacker_args in [upx_args]:
        try:
            subprocess.run(unpacker_args)
            if os.path.isfile(dest_path):
                break
        except Exception as e:
            Logger.error(f"Error while unpacking with {unpacker_args[0]} {sample.filepath}: {e}")

    if not os.path.isfile(dest_path):
        Logger.error(f"Unpacking failed: {sample.md5} {sample.sha256} {sample.filepath}")


@decorator_sample_processor(Bodmas)
def create_csv_for_arming_samples(dset: Type[DatasetProvider], sample: Sample):
    create_csv_for_arming_sample(dset, sample)


@decorator_sample_processor(BodmasUnpacked)
def scan_unpacked_bodmas_sample(dset: Type[DatasetProvider], sample: Sample):
    scan_sample(dset, sample)


@decorator_callgraph_processor(BodmasUnpacked, skip_load_if=lambda dset, md5: os.path.isfile(
    get_path_image(dset, md5, (300, 300), True, True)))
def create_dfs(dset: Type[DatasetProvider], cg: CallGraph):
    create_callgraph_dfs(dset, cg, img_dims=[(30, 30), (100, 100), (224, 224), (300, 300)])


def __arm_samples():
    process_samples(Bodmas, create_csv_for_arming_samples, batch_size=1000, max_batches=None,
                    pool=ThreadPoolExecutor(max_workers=8))
    with open(ARM_INFO, "w") as f:
        f.write(BUFF)


def __unpack_samples():
    Logger.set_file_logging("unpacker.log")
    process_samples(BodmasArmed, unpack_sample, batch_size=1000, max_batches=None, pool=None)

    import pandas as pd
    index_col_sha = "filename(original sha256)"
    df = pd.read_csv(BODMAS_GROUND_TRUTH_CSV, index_col=index_col_sha)
    for sample in BodmasUnpacked.get_samples():
        original_sha = BodmasUnpacked.sha256_from_filename(os.path.basename(sample.filepath))

        df.loc[original_sha, "unpacked-md5"] = sample.md5
        df.loc[original_sha, "unpacked-sha256"] = sample.sha256
    df.to_csv(BODMAS_GROUND_TRUTH_CSV)

    # rename_png_files_to_contain_family(BodmasUnpacked.get_dir_images() + "/images_300x300_with_families", md5_index_key="unpacked-md5")


def tmp_plot_unpacked_samples_comparison():
    import pandas as pd
    from PIL import Image
    import matplotlib.pyplot as plt

    df = pd.read_csv(BODMAS_GROUND_TRUTH_CSV, index_col="md5")
    unique_unpacked_md5s = df["unpacked-md5"].dropna().unique()
    for unpacked_md5 in unique_unpacked_md5s:
        rows_having_this_unpacked_md5 = df[df["unpacked-md5"] == unpacked_md5]

        Logger.info(f"Processing {unpacked_md5} - {len(rows_having_this_unpacked_md5)} samples")
        fig = plt.figure(figsize=(20, 20))
        plt.subplot(2, len(rows_having_this_unpacked_md5), 1)
        plt.title(f"Unpacked image {unpacked_md5}")
        image = Image.open(get_path_image(BodmasUnpacked, unpacked_md5, (300, 300), True, True))
        plt.imshow(image)

        i = 0
        for index, row in rows_having_this_unpacked_md5.iterrows():
            i += 1
            md5 = row.name
            image = Image.open(get_path_image(Bodmas, md5, (300, 300), True, True))
            plt.subplot(2, len(rows_having_this_unpacked_md5), len(rows_having_this_unpacked_md5) + i)
            plt.title(f"BODMAS: {md5}\n{row['family']}")
            plt.imshow(image)

        families = rows_having_this_unpacked_md5["family"].dropna().unique()
        family_count = rows_having_this_unpacked_md5["family"].value_counts()
        family_info = "-OR-".join([f"{f}:{family_count[f]}" for f in families])
        plt.savefig(os.path.join(BodmasUnpacked.get_dir_analysis_custom("unpacked_comparison"),
                                 f"unpacked_{unpacked_md5}_{len(rows_having_this_unpacked_md5)}_bodmas_files_{family_info}.pdf"))
        plt.close(fig)


if __name__ == "__main__":
    config.load_env()
    # __arm_samples()
    # __unpack_samples()
    # process_samples(BodmasUnpacked, scan_unpacked_bodmas_sample, batch_size=1000, max_batches=None,
    #                 pool=ThreadPoolExecutor(max_workers=8))
