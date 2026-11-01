import io

import pytest
from PIL import Image

from benchproof import photos
from benchproof.photos import MAX_SIDE, prepare


def encoded(image: Image.Image, fmt: str = "PNG", **kwargs) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, **kwargs)
    return buffer.getvalue()


def test_photo_is_a_jpeg_of_the_reported_size():
    photo = prepare(encoded(Image.new("RGB", (80, 60), "white")))

    with Image.open(io.BytesIO(photo.jpeg)) as image:
        assert image.format == "JPEG"
        assert image.size == (photo.width, photo.height) == (80, 60)


def test_large_photo_is_scaled_down_keeping_its_shape():
    photo = prepare(encoded(Image.new("RGB", (MAX_SIDE * 2, MAX_SIDE), "white")))
    assert (photo.width, photo.height) == (MAX_SIDE, MAX_SIDE // 2)


def test_exif_rotation_is_applied_before_the_models_see_it():
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise when displayed
    sideways = encoded(Image.new("RGB", (80, 60), "white"), "JPEG", exif=exif)

    photo = prepare(sideways)

    assert (photo.width, photo.height) == (60, 80)


def test_same_pixels_in_different_formats_share_a_digest():
    image = Image.new("RGB", (8, 6), (10, 20, 30))
    assert prepare(encoded(image, "PNG")).sha256 == prepare(encoded(image, "BMP")).sha256


def test_different_pixels_have_different_digests():
    first = prepare(encoded(Image.new("RGB", (8, 6), "white")))
    second = prepare(encoded(Image.new("RGB", (8, 6), "black")))
    assert first.sha256 != second.sha256


def test_oversized_image_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 10)
    with pytest.raises(ValueError, match="too large"):
        prepare(encoded(Image.new("RGB", (8, 6))))


def test_non_image_bytes_are_rejected():
    with pytest.raises(Image.UnidentifiedImageError):
        photos.open_upright(b"not an image")
