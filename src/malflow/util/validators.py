import re
from typing import Tuple, Type, List, Union, Callable


class HashValidator:
    RE_MD5 = re.compile(r"^[a-f0-9]{32}$")
    RE_SHA256 = re.compile(r"^[0-9a-f]{64}$")

    @staticmethod
    def is_md5(value):
        return isinstance(value, str) and HashValidator.RE_MD5.match(value)

    @staticmethod
    def is_sha256(value):
        return isinstance(value, str) and HashValidator.RE_SHA256.match(value)


class Validator:
    @staticmethod
    def validate_bool(e: str) -> bool:
        if isinstance(e, str):
            e = e.lower()
        if e in ("t", "y", "yes", "on", "true", "1", 1, True):
            return True
        elif e in ("f", "n", "no", "off", "false", "0", 0, False):
            return False
        else:
            raise ValueError(f"Cannot convert {e!r} to bool")

    @staticmethod
    def validate_int(e: str) -> int:
        return int(e)

    @staticmethod
    def validate_shape(e: str) -> Tuple[int, int]:
        if not isinstance(e, str):
            raise ValueError(f"Cannot convert {e!r} to shape")
        tokens = e.split(",")
        if len(tokens) != 2:
            raise ValueError(f"Cannot convert {e!r} to shape")

        return int(tokens[0]), int(tokens[1])

    @staticmethod
    def validate_list(e: str, type_: Union[Type, Callable]) -> List:
        if not isinstance(e, str):
            raise ValueError(f"Cannot convert {e!r} to list")
        tokens = e.split(",")
        return list(set(type_(t.strip()) for t in tokens))
