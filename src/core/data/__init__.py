from abc import abstractmethod, ABC
from typing import Generator

from core.model.sample import Sample


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
