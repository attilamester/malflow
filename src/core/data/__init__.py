from abc import abstractmethod, ABC
from typing import Generator

from core.model.sample import Sample


class DatasetProvider(ABC):
    @staticmethod
    @abstractmethod
    def get_samples() -> Generator[Sample, None, None]:
        pass

    @staticmethod
    @abstractmethod
    def get_sample(md5: str = None, sha256: str = None) -> Sample:
        pass

    @staticmethod
    @abstractmethod
    def get_dir_samples() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_dir_analysis() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_dir_callgraphs() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_dir_images() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_dir_instructions() -> str:
        pass

    @staticmethod
    @abstractmethod
    def get_dir_info() -> str:
        pass
