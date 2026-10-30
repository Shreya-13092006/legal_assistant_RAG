"""Optional controlled OCR experiment (SDD 3.2.10 / 7.7). Output goes to data/ocr_experiment/ only and
is never loaded into the production graph."""
from __future__ import annotations

import random
import re
from pathlib import Path

from lawrag.errors import ConfigError


def levenshtein(a: list | str, b: list | str) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()


def cer(ref: str, hyp: str) -> float:
    r, h = _clean(ref), _clean(hyp)
    return levenshtein(r, h) / max(len(r), 1)


def wer(ref: str, hyp: str) -> float:
    r, h = _clean(ref).split(), _clean(hyp).split()
    return levenshtein(r, h) / max(len(r), 1)


def section_number_accuracy(ref_numbers: list[str], hyp_numbers: list[str]) -> float:
    if not ref_numbers:
        return 1.0
    hyp = set(hyp_numbers)
    return sum(1 for n in ref_numbers if n in hyp) / len(ref_numbers)


def boundary_f1(ref: list[str], hyp: list[str]) -> float:
    r, h = set(ref), set(hyp)
    if not r and not h:
        return 1.0
    p = len(r & h) / len(h) if h else 0.0
    q = len(r & h) / len(r) if r else 0.0
    return 2 * p * q / (p + q) if p + q else 0.0


def rasterize(pdf: Path, page: int, out: Path, dpi: int = 300, noise: float = 0.0, skew: float = 0.0,
              jpeg: int = 95, seed: int = 13) -> Path:
    """Render a native page and degrade it deterministically (seeded noise, skew, JPEG)."""
    try:
        import fitz
        from PIL import Image, ImageFilter
    except ImportError as exc:
        raise ConfigError("OCR experiment needs pymupdf and Pillow (pip install lawrag[ocr,ingest])") from exc
    rng = random.Random(seed)
    with fitz.open(pdf) as doc:
        pix = doc[page - 1].get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples).convert("L")
    if skew:
        img = img.rotate(skew, expand=True, fillcolor=255)
    if noise:
        px = img.load()
        for _ in range(int(noise * img.width * img.height * 0.05)):
            x, y = rng.randrange(img.width), rng.randrange(img.height)
            px[x, y] = 0 if rng.random() < 0.5 else 255
        img = img.filter(ImageFilter.GaussianBlur(radius=noise * 2))
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "JPEG", quality=jpeg)
    return out


def ocr_image(path: Path) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ConfigError("pytesseract is not installed; the OCR experiment is optional (report it as not run)") from exc
    try:
        import cv2
        import numpy as np
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        coords = np.column_stack(np.where(img < 128))
        if len(coords):
            angle = cv2.minAreaRect(coords)[-1]
            angle = -(90 + angle) if angle < -45 else -angle
            h, w = img.shape
            m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return pytesseract.image_to_string(img, lang="eng")
    except ImportError:
        return pytesseract.image_to_string(Image.open(path), lang="eng")
    except pytesseract.TesseractNotFoundError as exc:
        raise ConfigError("Tesseract binary not found on PATH; install it or skip the OCR experiment") from exc
