import os

from core.data import DatasetProvider
from core.model.sample import Sample
from util.logger import Logger
from util.validators import HashValidator


class Bodmas(DatasetProvider):
    @staticmethod
    def get_samples():
        bodmas_samples_dir = Bodmas.get_dir_samples()
        filenames = os.listdir(bodmas_samples_dir)
        for filename in filenames:
            filepath = os.path.join(bodmas_samples_dir, filename)
            sha256 = filename.split(".")[0]

            if not HashValidator.is_sha256(sha256):
                Logger.warning(f"Skipping invalid BODMAS filename: {filename}")
                continue

            yield Sample(filepath, md5=None, sha256=sha256, check_hashes=False)

    @staticmethod
    def get_sample(md5: str = None, sha256: str = None) -> Sample:
        bodmas_samples_dir = Bodmas.get_dir_samples()
        filepath = os.path.join(bodmas_samples_dir, f"{sha256}.exe")
        if not os.path.isfile(filepath):
            raise Exception(f"Cannot find BODMAS file: {filepath}")
        return Sample(filepath=filepath, sha256=sha256, check_hashes=False)

    @staticmethod
    def is_existing_analysis(md5: str):
        return os.path.isfile(os.path.join(os.environ["BODMAS_DIR_R2_SCANS"], f"{md5}.info"))

    @staticmethod
    def get_dir_samples():
        return os.environ["BODMAS_DIR_SAMPLES"]

    @staticmethod
    def get_dir_r2_scans():
        return os.environ["BODMAS_DIR_R2_SCANS"]
