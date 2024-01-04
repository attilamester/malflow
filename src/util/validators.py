import re


class HashValidator:
    RE_MD5 = re.compile(r"^[a-f0-9]{32}$")
    RE_SHA256 = re.compile(r"^[0-9a-f]{64}$")

    @staticmethod
    def is_md5(value):
        return isinstance(value, str) and HashValidator.RE_MD5.match(value)

    @staticmethod
    def is_sha256(value):
        return isinstance(value, str) and HashValidator.RE_SHA256.match(value)
