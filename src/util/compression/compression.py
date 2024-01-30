import bz2
import lzma
import zlib

import brotli


class Compressor:

    def compress(self, content: bytes) -> bytes:
        raise Exception("Not implemented")

    def decompress(self, compressed: bytes) -> bytes:
        raise Exception("Not implemented")

    @staticmethod
    def get_decompressor(compressed: bytes) -> "Compressor":
        if compressed.startswith(b"BZh"):
            return Bzip2Compressor()
        elif compressed.startswith(b"x\x9c"):
            return ZLibCompressor()
        elif compressed.startswith(b"\xfd"):
            return LzmaCompressor()
        elif compressed.startswith(b"BROT"):
            return BrotliCompressor(11)
        raise Exception("Unknown compression algorithm")

    def __str__(self):
        raise Exception("Not implemented")


class BrotliCompressor(Compressor):

    def __init__(self, quality: int):
        super().__init__()
        self.__quality = quality

    def compress(self, content: bytes) -> bytes:
        return b"BROT" + brotli.compress(content, quality=self.__quality)

    def decompress(self, compressed: bytes) -> bytes:
        if not compressed.startswith(b"BROT"):
            raise Exception("Invalid compressed content")
        return brotli.decompress(compressed[4:])

    def __str__(self):
        return f"BrotliCompressor(q={self.__quality})"


class ZLibCompressor(Compressor):

    def compress(self, content: bytes) -> bytes:
        return zlib.compress(content)

    def decompress(self, compressed: bytes) -> bytes:
        return zlib.decompress(compressed)

    def __str__(self):
        return "ZLibCompressor"


class Bzip2Compressor(Compressor):

    def compress(self, content: bytes) -> bytes:
        return bz2.compress(content)

    def decompress(self, compressed: bytes) -> bytes:
        return bz2.decompress(compressed)

    def __str__(self):
        return "Bzip2Compressor"


class LzmaCompressor(Compressor):

    def compress(self, content: bytes) -> bytes:
        return lzma.compress(content)

    def decompress(self, compressed: bytes) -> bytes:
        return lzma.decompress(compressed)

    def __str__(self):
        return "LzmaCompressor"
