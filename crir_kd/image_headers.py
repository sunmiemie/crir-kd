"""Read PNG/JPEG dimensions; header inspection is not a full decode check."""

from pathlib import Path
import struct
import zlib


def _read(stream, count):
    data = stream.read(count)
    if len(data) != count:
        raise ValueError("truncated image header")
    return data


def image_dimensions(path):
    with Path(path).open("rb") as stream:
        signature = _read(stream, 2)
        if signature == b"\x89P":
            if signature + _read(stream, 6) != b"\x89PNG\r\n\x1a\n":
                raise ValueError("invalid PNG signature")
            length, kind = struct.unpack(">I4s", _read(stream, 8))
            if length != 13 or kind != b"IHDR":
                raise ValueError("PNG must begin with a 13-byte IHDR")
            data = _read(stream, length)
            crc, = struct.unpack(">I", _read(stream, 4))
            if zlib.crc32(kind + data) != crc:
                raise ValueError("PNG IHDR checksum mismatch")
            width, height = struct.unpack(">II", data[:8])
            format_name = "PNG"
        elif signature == b"\xff\xd8":
            while True:
                if _read(stream, 1) != b"\xff":
                    raise ValueError("invalid JPEG marker")
                marker = _read(stream, 1)[0]
                while marker == 0xFF:
                    marker = _read(stream, 1)[0]
                if marker in (0xD9, 0xDA, 0x00):
                    raise ValueError("JPEG has no supported frame header before image data")
                if marker in (0x01, *range(0xD0, 0xD8)):
                    continue
                length, = struct.unpack(">H", _read(stream, 2))
                if length < 2:
                    raise ValueError("invalid JPEG segment length")
                data = _read(stream, length - 2)
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    if len(data) < 6 or len(data) != 6 + 3 * data[5]:
                        raise ValueError("invalid JPEG frame header length")
                    height, width = struct.unpack(">HH", data[1:5])
                    format_name = "JPEG"
                    break
        else:
            raise ValueError("supported image headers are PNG and JPEG")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return {"width": width, "height": height, "format": format_name, "full_decode_verified": False}
