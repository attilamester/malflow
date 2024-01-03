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
