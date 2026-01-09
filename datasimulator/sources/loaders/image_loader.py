"""
Image OCR loader.

Extracts text from images using Tesseract OCR:
- JPEG
- PNG
- TIFF
- BMP
- GIF
"""

import logging
from pathlib import Path
from typing import Optional

from ..base_loader import BaseLoader, LoaderException

logger = logging.getLogger(__name__)


class ImageLoader(BaseLoader):
    """
    Load and extract text from images using OCR.

    Requires:
    - pytesseract: Python wrapper for Tesseract
    - Pillow: Image processing library
    - Tesseract OCR: System dependency (must be installed separately)
    """

    SUPPORTED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif']

    def __init__(
        self,
        source: str,
        language: str = 'eng',
        tesseract_cmd: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize image loader.

        Args:
            source: Path to image file
            language: OCR language code (default: 'eng' for English)
            tesseract_cmd: Path to tesseract executable (if not in PATH)
            **kwargs: Additional options
        """
        super().__init__(source, **kwargs)
        self.language = language
        self.tesseract_cmd = tesseract_cmd

    def load(self) -> str:
        """
        Load text from image using OCR.

        Returns:
            Extracted text content
        """
        path = Path(self.source)
        self._validate_file_exists(path)

        try:
            import pytesseract
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "Required libraries not installed. "
                "Install with: pip install pytesseract Pillow\n"
                "Also install Tesseract OCR:\n"
                "  Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
                "  macOS: brew install tesseract\n"
                "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
            )

        logger.info(f"Loading image for OCR: {path}")

        try:
            # Set tesseract command if provided
            if self.tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd

            # Open and process image
            image = Image.open(path)

            # Convert to RGB if needed (some formats like PNG with alpha)
            if image.mode not in ('RGB', 'L'):
                image = image.convert('RGB')

            # Extract text using OCR
            content = pytesseract.image_to_string(
                image,
                lang=self.language,
                config='--psm 3'  # Assume single column of text
            )

            # Clean up extracted text
            content = content.strip()

            # Store metadata
            self.metadata = {
                'file_path': str(path),
                'file_size': path.stat().st_size,
                'image_size': image.size,
                'image_format': image.format,
                'image_mode': image.mode,
                'char_count': len(content),
                'language': self.language,
                'method': 'tesseract_ocr'
            }

            logger.info(
                f"OCR extracted {len(content)} chars from "
                f"{image.size[0]}x{image.size[1]} image"
            )

            return content

        except pytesseract.TesseractNotFoundError:
            raise LoaderException(
                "Tesseract OCR not found. Please install:\n"
                "  Ubuntu/Debian: sudo apt-get install tesseract-ocr\n"
                "  macOS: brew install tesseract\n"
                "  Windows: https://github.com/UB-Mannheim/tesseract/wiki"
            )
        except Exception as e:
            raise LoaderException(f"Error performing OCR on image: {e}")
