"""
Shared OCR utility functions used by main.py and benchmark.py.
"""

import base64
import io

import pymupdf
from PIL import Image


def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[Image.Image]:
    """PDF 파일을 페이지별 PIL Image 리스트로 변환"""
    images = []
    zoom = dpi / 72  # 72 is default PDF DPI
    matrix = pymupdf.Matrix(zoom, zoom)

    with pymupdf.open(pdf_path) as doc:
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(matrix=matrix)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
            print(f"  페이지 {page_num + 1}/{len(doc)} 변환 완료 ({pix.width}x{pix.height})")

    return images


def image_to_base64(img: Image.Image) -> str:
    """PIL Image를 base64 문자열로 변환"""
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_pdf_page_count(pdf_path: str) -> int:
    """PDF 파일의 총 페이지 수를 반환"""
    with pymupdf.open(pdf_path) as doc:
        return len(doc)
