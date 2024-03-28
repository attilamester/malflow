import os

from core.data import DatasetProvider
from core.model.sample import Sample
from util.logger import Logger
from util.misc import ensure_dir
from util.validators import HashValidator


class Bodmas(DatasetProvider):

    @classmethod
    def get_samples(cls):
        bodmas_samples_dir = cls.get_dir_samples()
        filenames = os.listdir(bodmas_samples_dir)
        for filename in filenames:
            filepath = os.path.join(bodmas_samples_dir, filename)
            sha256 = cls.sha256_from_filename(filename)

            if not HashValidator.is_sha256(sha256):
                Logger.warning(f"Skipping invalid BODMAS filename: {filename}")
                continue

            yield Sample(filepath, md5=None, sha256=sha256, check_hashes=False)

    @classmethod
    def get_sample(cls, md5: str = None, sha256: str = None) -> Sample:
        bodmas_samples_dir = cls.get_dir_samples()
        filepath = os.path.join(bodmas_samples_dir, cls.filename_from_sha256(sha256))
        if not os.path.isfile(filepath):
            raise Exception(f"Cannot find BODMAS file: {filepath}")
        return Sample(filepath=filepath, sha256=sha256, check_hashes=False)

    @classmethod
    def sha256_from_filename(cls, filename):
        return filename.split(".")[0]

    @classmethod
    def filename_from_sha256(cls, sha256):
        return f"{sha256}.exe"

    @classmethod
    def is_existing_analysis(cls, md5: str):
        return os.path.isfile(os.path.join(cls.get_dir_info(), f"{md5}.info"))

    @classmethod
    def get_dir_samples(cls):
        return os.environ["BODMAS_DIR_SAMPLES"]

    @classmethod
    def get_dir_analysis(cls):
        return os.environ["BODMAS_DIR_R2_SCANS"]

    @classmethod
    def get_dir_callgraphs(cls):
        path = os.path.join(cls.get_dir_analysis(), "callgraphs")
        ensure_dir(path)
        return path

    @classmethod
    def get_dir_images(cls):
        path = os.path.join(cls.get_dir_analysis(), "images")
        ensure_dir(path)
        return path

    @classmethod
    def get_dir_instructions(cls):
        path = os.path.join(cls.get_dir_analysis(), "instructions")
        ensure_dir(path)
        return path

    @classmethod
    def get_dir_info(cls):
        path = os.path.join(cls.get_dir_analysis(), "info")
        ensure_dir(path)
        return path


class BodmasArmed(Bodmas):

    @classmethod
    def get_dir_samples(cls):
        return os.path.join(os.path.dirname(Bodmas.get_dir_samples()), "armed")

    @classmethod
    def get_dir_analysis(cls):
        raise Exception("BodmasArmed does not have an analysis directory")


class BodmasUnpacked(BodmasArmed):

    @classmethod
    def get_dir_samples(cls):
        return os.path.join(os.path.dirname(BodmasArmed.get_dir_samples()),
                            os.path.basename(BodmasArmed.get_dir_samples()) + "_unpacked")

    @classmethod
    def get_dir_analysis(cls):
        return os.path.join(os.path.dirname(Bodmas.get_dir_analysis()),
                            os.path.basename(Bodmas.get_dir_analysis()) + "_unpacked")

    @classmethod
    def sha256_from_filename(cls, filename):
        return filename.split(".")[0].replace("unpacked_", "")

    @classmethod
    def filename_from_sha256(cls, sha256):
        return f"unpacked_{sha256}.exe"


class BodmasAugmented(Bodmas):

    @classmethod
    def get_dir_samples(cls):
        return os.path.join(os.path.dirname(Bodmas.get_dir_samples()), "augmented")

    @classmethod
    def get_dir_analysis(cls):
        return os.path.join(os.path.dirname(Bodmas.get_dir_analysis()),
                            os.path.basename(Bodmas.get_dir_analysis()) + "_augmented")
