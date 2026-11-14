"""Image files -> one image element, normalized to RGB PNG."""

import io

from PIL import Image, ImageOps

from ingest.elements import Element, ParsedDocument

MAX_STORED_SIDE = 2048


def to_png(image: Image.Image, max_side: int = MAX_STORED_SIDE) -> bytes:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def parse_image(data: bytes) -> ParsedDocument:
    with Image.open(io.BytesIO(data)) as probe:
        probe.verify()
    with Image.open(io.BytesIO(data)) as image:
        png = to_png(image)
    return ParsedDocument(elements=[Element(kind="image", image=png, standalone=True)])
