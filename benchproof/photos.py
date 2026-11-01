import io
from dataclasses import dataclass
from hashlib import sha256

from PIL import Image, ImageOps

# Phone photos are 12 MP or more; the models see a downscaled copy, which keeps each check fast and cheap.
MAX_SIDE = 1600
JPEG_QUALITY = 92


@dataclass(frozen=True)
class Photo:
    """One capture as every model sees it: same pixels, same size, same orientation."""

    jpeg: bytes
    width: int
    height: int
    sha256: str


def open_upright(image_bytes: bytes) -> Image.Image:
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except Image.DecompressionBombError as e:
        raise ValueError("photo is too large; lower the camera resolution and retake it") from e


def pixel_digest(image: Image.Image) -> str:
    dimensions = image.width.to_bytes(8, "big") + image.height.to_bytes(8, "big")
    return sha256(dimensions + image.tobytes()).hexdigest()


def prepare(image_bytes: bytes) -> Photo:
    image = open_upright(image_bytes)
    image.thumbnail((MAX_SIDE, MAX_SIDE))

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return Photo(jpeg=buffer.getvalue(), width=image.width, height=image.height, sha256=pixel_digest(image))
