import hashlib
from enum import Enum


class CGFingerprint:
    class Type(Enum):
        FUNCTION = "function"
        CALL = "call"

    md5: str
    type: Type
    value: str
    hash: str
    tf: int

    def __init__(self, md5, type: Type, value):
        self.md5 = md5
        self.type = type
        self.value = value
        self.hash = hashlib.md5(value.encode("ascii")).hexdigest()
        self.tf = 1
