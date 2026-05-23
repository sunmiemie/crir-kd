import struct
import tempfile
from pathlib import Path
import unittest
import zlib

from crir_kd.image_headers import image_dimensions


def png_header(width, height):
    data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + data + struct.pack(">I", zlib.crc32(b"IHDR" + data))


class ImageHeaderTests(unittest.TestCase):
    def test_png_and_jpeg_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image"
            for content, size in [(png_header(32, 24), (32, 24)),
                                  (b"\xff\xd8\xff\xc0\x00\x0b\x08\x00\x18\x00\x20\x01\x01\x11\x00", (32, 24))]:
                path.write_bytes(content)
                result = image_dimensions(path)
                self.assertEqual((result["width"], result["height"]), size)
                self.assertFalse(result["full_decode_verified"])

    def test_corrupt_headers_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image"
            for content in [b"", b"not an image", png_header(0, 4), png_header(4, 4)[:-1], b"\xff\xd8\xff\xda"]:
                path.write_bytes(content)
                with self.assertRaises(ValueError):
                    image_dimensions(path)
