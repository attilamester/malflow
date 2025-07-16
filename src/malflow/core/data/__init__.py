import os
from abc import abstractmethod, ABC
from typing import Generator

from malflow.core.model.sample import Sample
from malflow.util.misc import ensure_dir


class DatasetProvider(ABC):
    @classmethod
    @abstractmethod
    def get_samples(cls) -> Generator[Sample, None, None]:
        pass

    @classmethod
    @abstractmethod
    def get_sample(cls, md5: str = None, sha256: str = None) -> Sample:
        pass

    @classmethod
    @abstractmethod
    def get_dir_samples(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def get_dir_analysis(cls) -> str:
        pass

    @classmethod
    def get_dir_analysis_custom(cls, custom: str) -> str:
        path = os.path.join(cls.get_dir_info(), custom)
        ensure_dir(path)
        return path

    @classmethod
    @abstractmethod
    def get_dir_callgraphs(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def get_dir_images(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def get_dir_instructions(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def get_dir_info(cls) -> str:
        pass
