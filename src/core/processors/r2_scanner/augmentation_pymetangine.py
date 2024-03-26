import os.path
import random
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor
from typing import Type

import pandas as pd

from core.data import DatasetProvider
from core.data.bodmas import Bodmas, BodmasPymetangined
from core.model.sample import Sample
from core.processors.cg_image_classification.paths import get_cg_image_classification_env
from core.processors.r2_scanner.scan_samples import scan_sample
from core.processors.util import decorator_sample_processor
from helpers.ground_truth import BODMAS_GROUND_TRUTH_CSV
from util import config
from util.logger import Logger

config.load_env(get_cg_image_classification_env())

from core.processors.cg_image_classification.dataset.preprocess import df_filter_having_at_column_min_occurencies

BODMAS_GROUND_TRUTH_AUGMENTATION_PYMETANGINE_CSV = os.path.join(
    os.path.dirname(BODMAS_GROUND_TRUTH_CSV), "BODMAS_ground_truth_augmentation_pymetangine.csv")


def get_augmented_filename(dirpath, filename: str):
    fname, ext = os.path.splitext(os.path.basename(filename))
    i = 0
    while True:
        new_filename = f"{fname}_augm{i}{ext}"
        new_path = os.path.join(dirpath, new_filename)
        if not os.path.exists(new_path):
            return new_path
        i += 1


def augment_pymetangine(input_path: str, output_path: str):
    input_dir = os.path.dirname(input_path)
    input_name = os.path.basename(input_path)
    output_dir = os.path.dirname(output_path)
    output_name = os.path.basename(output_path)
    subprocess.run(["docker", "run", "--rm",
                    "-v", f"{input_dir}:/input",
                    "-v", f"{output_dir}:/output",
                    "attilamester/pymetangine",
                    "-i", os.path.join("/input", input_name),
                    "-o", os.path.join("/output", output_name)])


def augment_pymetangine_sample(sample: Sample):
    input_path = sample.filepath
    output_path = get_augmented_filename(BodmasPymetangined.get_dir_samples(), os.path.basename(input_path))
    augment_pymetangine(input_path, output_path)
    if os.path.isfile(output_path):
        augmented_sample = Sample(filepath=output_path, sha256=None, check_hashes=False)
        if augmented_sample.sha256 == sample.sha256:
            Logger.warning(f"Augmenting sample failed: {sample.sha256} has the same sha256")
            os.remove(output_path)
            return None, None
        Logger.info(f"Augmenting sample OK: {sample.sha256} -> {augmented_sample.sha256}")
        return sample, augmented_sample
    else:
        return None, None


def create_augmentation_with_pymetangine(pct=0.2, original_min_occurencies: int = 100):
    df = pd.read_csv(BODMAS_GROUND_TRUTH_CSV, index_col="sha256")
    df_filtered = df_filter_having_at_column_min_occurencies(df, "family", min_occurencies=original_min_occurencies)
    value_counts = df_filtered["family"].value_counts()
    value_counts.sort_values(ascending=True, inplace=True)

    max_occurrence = value_counts.max()

    for family, count in value_counts.items():
        augm_needed = int(((1 + pct) * max_occurrence / count))

        rows_with_family = df_filtered[df_filtered["family"] == family]
        ids_not_packed = rows_with_family[rows_with_family["packer"].isna()][
            "filename(original sha256)"]
        ids_successfully_unpacked = rows_with_family[~ rows_with_family["unpacked-md5"].isna()][
            "filename(original sha256)"]

        Logger.info(f"Family {family} | Augmentation needed: {augm_needed}\n"
                    f"\tsamples:            {count:>5}\n"
                    f"\tsamples not packed: {len(ids_not_packed):>5}\n"
                    f"\tsamples unpacked:   {len(ids_successfully_unpacked):>5}")

        if len(ids_not_packed) == 0:
            continue

        ids_to_augment = random.choices(ids_not_packed, k=augm_needed)
        samples_to_augment = [Bodmas.get_sample(sha256=id_) for id_ in ids_to_augment]

        with ProcessPoolExecutor(max_workers=8) as executor:
            results = executor.map(augment_pymetangine_sample, samples_to_augment)
            for sample, augmented_sample in results:
                if sample is None:
                    continue


def create_augmentation_ground_truth():
    df = pd.read_csv(BODMAS_GROUND_TRUTH_CSV, index_col="filename(original sha256)")
    augm_data = []
    for filename in os.listdir(BodmasPymetangined.get_dir_samples()):
        if "_augm" in filename:
            original_sha256 = filename.split("_")[0]
            orig_sample = Bodmas.get_sample(sha256=original_sha256)
            augm_sample = Sample(filepath=os.path.join(BodmasPymetangined.get_dir_samples(), filename),
                                 sha256=None, check_hashes=False)
            shutil.copyfile(augm_sample.filepath, os.path.join(BodmasPymetangined.get_dir_samples(),
                                                               BodmasPymetangined.filename_from_sha256(
                                                                   augm_sample.sha256)))
            family = df.loc[original_sha256, "family"]
            augm_data.append([family, augm_sample.md5, augm_sample.sha256, orig_sample.md5, orig_sample.sha256])
    df_augm = pd.DataFrame(data=augm_data,
                           columns=["family", "md5", "sha256", "augmentation_of_md5", "augmentation_of_sha256"])
    df_augm.sort_values(by="family", inplace=True)
    df_augm.to_csv(BODMAS_GROUND_TRUTH_AUGMENTATION_PYMETANGINE_CSV, index=False)


@decorator_sample_processor(BodmasPymetangined)
def scan_pymetangine_sample(dset: Type[DatasetProvider], sample: Sample):
    scan_sample(dset, sample)


if __name__ == "__main__":
    pass
    # create_augmentation_with_pymetangine()
    # create_augmentation_ground_truth()

    # process_samples(BodmasPymetangined, scan_pymetangine_sample, batch_size=1000, max_batches=None,
    #                 pool=ThreadPoolExecutor(max_workers=8))
