import os.path
import random
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from core.data.bodmas import Bodmas, BodmasAugmented
from core.model.sample import Sample
from core.processors.cg_image_classification.paths import get_cg_image_classification_env
from helpers.ground_truth import BODMAS_GROUND_TRUTH_CSV, BODMAS_GROUND_TRUTH_WITH_AUGM_CSV
from helpers.ground_truth import (BODMAS_GT_COL0,
                                  BODMAS_GT_COL1_ts,
                                  BODMAS_GT_COL2_fam,
                                  BODMAS_GT_COL3_md5,
                                  BODMAS_GT_COL4_sha,
                                  BODMAS_GT_COL5_up_md5,
                                  BODMAS_GT_COL6_up_sha,
                                  BODMAS_GT_COL7_packer,
                                  BODMAS_GT_COL8_augmof_md5,
                                  BODMAS_GT_COL9_augmof_sha)
from util import config
from util.logger import Logger
from util.misc import ensure_dir

config.load_env(get_cg_image_classification_env())

from core.processors.cg_image_classification.dataset.preprocess import df_filter_having_at_column_min_occurencies


def get_augmented_filename(dirpath, filename: str):
    fname, ext = os.path.splitext(os.path.basename(filename))
    i = 0
    while True:
        new_filename = f"{fname}_augm{i}{ext}"
        new_path = os.path.join(dirpath, new_filename)
        if not os.path.exists(new_path):
            return new_path
        i += 1


def augment_pymetangine(input_path: str, output_path: str, random: bool = True):
    input_dir = os.path.dirname(input_path)
    input_name = os.path.basename(input_path)
    output_dir = os.path.dirname(output_path)
    output_name = os.path.basename(output_path)
    subprocess.run(["docker", "run", "--rm",
                    "-v", f"{input_dir}:/input",
                    "-v", f"{output_dir}:/output",
                    "attilamester/pymetangine",
                    "-i", os.path.join("/input", input_name),
                    "-o", os.path.join("/output", output_name),
                    "--random", "y" if random else "n"  # if `n`, then each mutable instruction will be mutated
                    # "--debug" # uncomment for debug info
                    # e.g.
                    # [-] Mutating instruction (0x4039a8): mov eax, edx                        89d0 --> 89d0                mov eax, edx
                    # [*] Mutating instruction (0x4039aa): xor edx, edx                        33d2 --> 29d2                sub edx, edx
                    ])


def augment_pymetangine_sample(sample: Sample):
    input_path = sample.filepath
    output_path = get_augmented_filename(BodmasAugmented.get_dir_samples(), os.path.basename(input_path))

    augmented = False
    for trial in range(2):
        # 3 trials should be enough. But, some files are slow to scan so after the first error, we try with force
        # If the sample has no possible mutation options, then it does not matter how many times we try
        # But if it has N mutation options,
        #  then the chance that we did not mutate after 3 trials is
        #  at most (0.5^N) ^3  (at most - bc. each mutation has more than one mutation options and one original)
        # - 1 mutation option: 12.5%
        # - 2 mutation options: 1.5%
        # - 3 mutation options: 0.2% [...]

        augment_pymetangine(input_path, output_path, random=(trial == 0))
        if not os.path.isfile(output_path):
            continue

        augmented = True

        augmented_sample = Sample(filepath=output_path, sha256=None, check_hashes=False)
        if augmented_sample.sha256 == sample.sha256:
            os.remove(output_path)
            continue
        Logger.info(f"Augmenting sample OK   : {sample.filepath} SHA256: {sample.sha256} -> {augmented_sample.sha256}")
        return sample, augmented_sample

    if not augmented:
        Logger.error(f"Augmenting sample ERROR: {sample.filepath} could not scan")
    else:
        Logger.warning(f"Augmenting sample WARN : {sample.filepath} had no mutation option with pymetangine")
    return None, None


def create_augmentation_with_pymetangine(pct=0.2, original_min_occurencies: int = 100):
    df = pd.read_csv(BODMAS_GROUND_TRUTH_CSV, index_col="sha256")
    df_filtered = df_filter_having_at_column_min_occurencies(df, "family", min_occurencies=original_min_occurencies)
    value_counts = df_filtered["family"].value_counts()
    value_counts.sort_values(ascending=True, inplace=True)

    max_occurrence = value_counts.max()
    min_occurrence = value_counts.min()
    min_augmentations = int(min_occurrence * (1 + pct))
    total = 0
    for family, count in value_counts.items():
        # augm_needed = int(((1 + pct) * max_occurrence / count))
        augm_needed = int(min_augmentations * (min_occurrence / count))

        rows_with_family = df_filtered[df_filtered["family"] == family]
        ids_not_packed = rows_with_family[rows_with_family["packer"].isna()][
            BODMAS_GT_COL0]
        ids_successfully_unpacked = rows_with_family[~ rows_with_family["unpacked-md5"].isna()][
            BODMAS_GT_COL0]

        Logger.info(f"Family {family} | Augmentation needed: {augm_needed}\n"
                    f"\tsamples:            {count:>5}\n"
                    f"\tsamples not packed: {len(ids_not_packed):>5}\n"
                    f"\tsamples unpacked:   {len(ids_successfully_unpacked):>5}")

        if len(ids_not_packed) == 0:
            continue

        total += augm_needed

        ids_to_augment = random.choices(ids_not_packed, k=augm_needed)
        samples_to_augment = [Bodmas.get_sample(sha256=id_) for id_ in ids_to_augment]

        with ProcessPoolExecutor(max_workers=8) as executor:
            results = executor.map(augment_pymetangine_sample, samples_to_augment)
            for sample, augmented_sample in results:
                if sample is None:
                    continue

    Logger.info(f"Total augmentation needed: {total}")


def create_augmentation_ground_truth():
    df = pd.read_csv(BODMAS_GROUND_TRUTH_CSV, index_col=BODMAS_GT_COL0)
    augm_data = []
    for filename in os.listdir(BodmasAugmented.get_dir_samples()):
        if "_augm" in filename:
            original_sha256 = filename.split("_")[0]
            orig_sample = Bodmas.get_sample(sha256=original_sha256)
            augm_sample = Sample(filepath=os.path.join(BodmasAugmented.get_dir_samples(), filename),
                                 sha256=None, check_hashes=False)
            shutil.copyfile(augm_sample.filepath, os.path.join(BodmasAugmented.get_dir_samples(),
                                                               BodmasAugmented.filename_from_sha256(
                                                                   augm_sample.sha256)))
            family = df.loc[original_sha256, "family"]
            augm_data.append([original_sha256, None, family,
                              augm_sample.md5, augm_sample.sha256,
                              None, None, None,
                              orig_sample.md5, orig_sample.sha256])

    column_order = [BODMAS_GT_COL0, BODMAS_GT_COL1_ts, BODMAS_GT_COL2_fam,
                    BODMAS_GT_COL3_md5, BODMAS_GT_COL4_sha,
                    BODMAS_GT_COL5_up_md5, BODMAS_GT_COL6_up_sha, BODMAS_GT_COL7_packer,
                    BODMAS_GT_COL8_augmof_md5, BODMAS_GT_COL9_augmof_sha]
    df_augm = pd.DataFrame(data=augm_data, columns=column_order)
    df_augm.sort_values(by=BODMAS_GT_COL2_fam, inplace=True)

    df.reset_index(inplace=True)
    df[BODMAS_GT_COL8_augmof_md5] = ""
    df[BODMAS_GT_COL9_augmof_sha] = ""
    df_augm = pd.concat([df, df_augm], ignore_index=True)
    df_augm = df_augm[column_order]
    df_augm.to_csv(BODMAS_GROUND_TRUTH_WITH_AUGM_CSV, index=False)


def complete_image_collection(augm_images_dir, dim):
    df = pd.read_csv(BODMAS_GROUND_TRUTH_WITH_AUGM_CSV)
    df.set_index("md5", inplace=True)

    subdir = f"images_{dim[0]}x{dim[1]}"
    original_images_dir = os.path.join(Bodmas.get_dir_images(), subdir + "_with_augm")
    ensure_dir(original_images_dir)

    n = 0
    for index, row in df.iterrows():
        if pd.notna(row[BODMAS_GT_COL8_augmof_md5]):
            n += 1
            file_to_copy = f"{index}_{dim[0]}x{dim[1]}_True_True.png"
            path_to_copy_to = os.path.join(original_images_dir, file_to_copy)
            path_to_copy_from = os.path.join(augm_images_dir, subdir, file_to_copy)

            if not os.path.isfile(path_to_copy_from):
                print(f"{n} Error: File not found: {path_to_copy_from}")
                continue

            shutil.copy(path_to_copy_from, path_to_copy_to)

            print(f"{n} Copied \n\t{path_to_copy_from} --> \n\t{path_to_copy_to}")


if __name__ == "__main__":
    pass
    # create_augmentation_with_pymetangine()
    # create_augmentation_ground_truth()

    # process_samples(BodmasAugmented, scan_sample, batch_size=1000, max_batches=None,
    #                 pool=ThreadPoolExecutor(max_workers=8))

    # complete_image_collection(BodmasAugmented.get_dir_images(), (30, 30))
    # complete_image_collection(BodmasAugmented.get_dir_images(), (100, 100))
    # complete_image_collection(BodmasAugmented.get_dir_images(), (224, 224))
