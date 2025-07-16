import os

from malflow.core.data.bodmas import Bodmas
from malflow.core.model.sample import Sample
from malflow.util.logger import Logger
from malflow.util.validators import HashValidator


class MalImg(Bodmas):

    @classmethod
    def get_samples(cls):
        for sub in os.listdir(cls.get_dir_samples()):
            folder = os.path.join(cls.get_dir_samples(), sub)
            if not os.path.isdir(folder):
                continue

            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                md5 = cls.sha256_from_filename(filename)

                if not HashValidator.is_md5(md5):
                    Logger.warning(f"Skipping invalid MalImg filename: {filename}")
                    continue

                yield Sample(filepath, md5=md5, sha256=None, check_hashes=False)

    @classmethod
    def get_sample(cls, md5: str = None, sha256: str = None) -> Sample:
        filepath = os.path.join(cls.get_dir_samples(), cls.filename_from_sha256(md5))
        if not os.path.isfile(filepath):
            raise Exception(f"Cannot find MalImg file: {filepath}")
        return Sample(filepath=filepath, md5=md5, check_hashes=False)

    @classmethod
    def sha256_from_filename(cls, filename):
        return filename.split(".")[0]

    @classmethod
    def filename_from_sha256(cls, sha256):
        return f"{sha256}.bender"

    @classmethod
    def get_dir_samples(cls):
        return os.environ["MALIMG_DIR_SAMPLES"]

    @classmethod
    def get_dir_analysis(cls):
        return os.environ["MALIMG_DIR_ANALYSIS"]
