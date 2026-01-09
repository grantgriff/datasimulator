"""
PDF document loader.

Supports multiple PDF libraries for robust text extraction:
- PyPDF2: Fast, good for simple PDFs
- pdfplumber: Better for complex layouts, tables
"""

import logging
from pathlib import Path
from typing import Optional

from ..base_loader import BaseLoader, LoaderException

logger = logging.getLogger(__name__)


class PDFLoader(BaseLoader):
    """
    Load and extract text from PDF files.

    Tries multiple methods for best results:
    1. pdfplumber (if available) - best for complex layouts
    2. PyPDF2 (fallback) - faster for simple PDFs
    """

    SUPPORTED_EXTENSIONS = ['.pdf']

    def __init__(self, source: str, method: str = "auto", **kwargs):
        """
        Initialize PDF loader.

        Args:
            source: Path to PDF file
            method: Extraction method ("auto", "pdfplumber", "pypdf2")
            **kwargs: Additional options
        """
        super().__init__(source, **kwargs)
        self.method = method

    def load(self) -> str:
        """
        Load text from PDF.

        Returns:
            Extracted text content
        """
        path = Path(self.source)
        self._validate_file_exists(path)

        # Try methods in order of preference
        if self.method == "auto":
            try:
                return self._load_with_pdfplumber(path)
            except (ImportError, Exception) as e:
                logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
                try:
                    return self._load_with_pypdf2(path)
                except Exception as e2:
                    raise LoaderException(
                        f"All PDF extraction methods failed. "
                        f"Install pdfplumber or PyPDF2: pip install pdfplumber PyPDF2"
                    )

        elif self.method == "pdfplumber":
            return self._load_with_pdfplumber(path)

        elif self.method == "pypdf2":
            return self._load_with_pypdf2(path)

        else:
            raise LoaderException(f"Unknown PDF method: {self.method}")

    def _load_with_pdfplumber(self, path: Path) -> str:
        """Extract text using pdfplumber (best quality)."""
        try:
            import pdfplumber
        except ImportError:
            raise ImportError(
                "pdfplumber not installed. Install with: pip install pdfplumber"
            )

        logger.info(f"Loading PDF with pdfplumber: {path}")

        text_parts = []
        page_count = 0

        with pdfplumber.open(path) as pdf:
            page_count = len(pdf.pages)

            for i, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                        logger.debug(f"Extracted page {i}/{page_count}")
                except Exception as e:
                    logger.warning(f"Error extracting page {i}: {e}")

        content = "\n\n".join(text_parts)

        # Store metadata
        self.metadata = {
            'file_path': str(path),
            'file_size': path.stat().st_size,
            'page_count': page_count,
            'char_count': len(content),
            'method': 'pdfplumber'
        }

        logger.info(f"Extracted {page_count} pages, {len(content)} chars")
        return content

    def _load_with_pypdf2(self, path: Path) -> str:
        """Extract text using PyPDF2 (fallback)."""
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError(
                "PyPDF2 not installed. Install with: pip install PyPDF2"
            )

        logger.info(f"Loading PDF with PyPDF2: {path}")

        text_parts = []

        with open(path, 'rb') as f:
            pdf_reader = PdfReader(f)
            page_count = len(pdf_reader.pages)

            for i, page in enumerate(pdf_reader.pages, 1):
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                        logger.debug(f"Extracted page {i}/{page_count}")
                except Exception as e:
                    logger.warning(f"Error extracting page {i}: {e}")

        content = "\n\n".join(text_parts)

        # Store metadata
        self.metadata = {
            'file_path': str(path),
            'file_size': path.stat().st_size,
            'page_count': page_count,
            'char_count': len(content),
            'method': 'pypdf2'
        }

        logger.info(f"Extracted {page_count} pages, {len(content)} chars")
        return content
