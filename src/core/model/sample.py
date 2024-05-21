import hashlib

from core.model import CallGraph
from util.misc import display_size


class Sample:
    filepath: str
    md5: str
    sha256: str
    call_graph: CallGraph
    content: bytes
    size_fmt: str

    def __init__(self, filepath: str, md5: str = None, sha256: str = None, check_hashes: bool = True):
        self.filepath = filepath

        with open(filepath, "rb") as f:
            self.content = f.read()
            self.md5 = hashlib.md5(self.content).hexdigest()
            self.sha256 = hashlib.sha256(self.content).hexdigest()
            self.size_fmt = ""
            self.size_fmt = self.get_size_fmt()

        if check_hashes:
            if md5 and self.md5 != md5:
                raise Exception(f"Incorrect md5 provided for {filepath}")

            if sha256 and self.sha256 != sha256:
                raise Exception(f"Incorrect sha256 provided for {filepath}")

    def get_size_fmt(self):
        if not self.size_fmt:
            self.size_fmt = display_size(len(self.content))
        return self.size_fmt

    def get_ember_features(self):
        import numpy as np
        np.int = np.int32
        np.float = np.float64
        np.bool = np.bool_

        from ember.features import PEFeatureExtractor
        ex = PEFeatureExtractor(2, print_feature_warning=False)
        return ex.feature_vector(self.content)

    def __str__(self):
        return (f"""Sample(  {self.filepath}
    md5={self.md5}
    sha256={self.sha256})
    size={self.get_size_fmt()})""")
