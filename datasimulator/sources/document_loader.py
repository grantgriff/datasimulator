"""
Unified document loader with automatic format detection.

Automatically detects and loads documents from:
- Plain text files (.txt, .md)
- PDF files (.pdf)
- Word documents (.docx, .doc)
- Images (.jpg, .png, etc.) via OCR
- Web pages (http://, https://)
- Google Docs (Google Docs URLs or IDs)
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from urllib.parse import urlparse

from .base_loader import BaseLoader, LoaderException, TextLoader
from .loaders.pdf_loader import PDFLoader
from .loaders.word_loader import WordLoader
from .loaders.image_loader import ImageLoader
from .loaders.web_scraper import WebScraperLoader, JavaScriptWebLoader
from .loaders.google_docs_loader import GoogleDocsLoader

logger = logging.getLogger(__name__)


class DocumentLoader:
    """
    Unified document loader with automatic format detection.

    Automatically chooses the appropriate loader based on:
    - File extension (for local files)
    - URL scheme (for web content)
    - Google Docs URL pattern (for Google Docs)
    """

    # Map file extensions to loader classes
    LOADER_MAP = {
        # Text files
        '.txt': TextLoader,
        '.md': TextLoader,
        '.markdown': TextLoader,
        '.text': TextLoader,

        # PDFs
        '.pdf': PDFLoader,

        # Word documents
        '.docx': WordLoader,
        '.doc': WordLoader,

        # Images
        '.jpg': ImageLoader,
        '.jpeg': ImageLoader,
        '.png': ImageLoader,
        '.tiff': ImageLoader,
        '.tif': ImageLoader,
        '.bmp': ImageLoader,
        '.gif': ImageLoader,
    }

    def __init__(self, source: str, **loader_options):
        """
        Initialize document loader.

        Args:
            source: Path to file, URL, or Google Doc ID/URL
            **loader_options: Options to pass to the specific loader
        """
        self.source = source
        self.loader_options = loader_options
        self.loader = self._select_loader()

    def _select_loader(self) -> BaseLoader:
        """Select appropriate loader based on source."""

        # Check if it's a URL
        if self.source.startswith(('http://', 'https://')):
            return self._select_web_loader()

        # Check if it's a Google Docs URL or ID
        if 'docs.google.com' in self.source or self._looks_like_google_doc_id():
            logger.info("Detected Google Docs URL")
            return GoogleDocsLoader(self.source, **self.loader_options)

        # Check if it's a local file
        path = Path(self.source)
        if path.exists() and path.is_file():
            return self._select_file_loader(path)

        # If nothing matches, raise error
        raise LoaderException(
            f"Could not determine loader for source: {self.source}\n"
            f"Supported:\n"
            f"  - Local files (.txt, .pdf, .docx, .jpg, .png, etc.)\n"
            f"  - Web URLs (http://, https://)\n"
            f"  - Google Docs (URLs or document IDs)"
        )

    def _select_file_loader(self, path: Path) -> BaseLoader:
        """Select loader based on file extension."""
        extension = path.suffix.lower()

        loader_class = self.LOADER_MAP.get(extension)

        if loader_class:
            logger.info(f"Selected {loader_class.__name__} for {extension} file")
            return loader_class(self.source, **self.loader_options)
        else:
            raise LoaderException(
                f"Unsupported file format: {extension}\n"
                f"Supported formats: {', '.join(self.LOADER_MAP.keys())}"
            )

    def _select_web_loader(self) -> BaseLoader:
        """Select appropriate web loader."""
        # Check if JavaScript rendering is needed
        use_javascript = self.loader_options.pop('javascript', False)

        if use_javascript:
            logger.info("Using JavaScript-capable web loader (Playwright)")
            return JavaScriptWebLoader(self.source, **self.loader_options)
        else:
            logger.info("Using standard web scraper (BeautifulSoup)")
            return WebScraperLoader(self.source, **self.loader_options)

    def _looks_like_google_doc_id(self) -> bool:
        """Check if source looks like a Google Doc ID."""
        # Google Doc IDs are typically 44 characters, alphanumeric with dashes/underscores
        return (
            len(self.source) > 20 and
            len(self.source) < 100 and
            all(c.isalnum() or c in '-_' for c in self.source)
        )

    def load(self) -> str:
        """
        Load content from source.

        Returns:
            Extracted text content
        """
        return self.loader.load()

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the loaded document.

        Returns:
            Dictionary with metadata
        """
        return self.loader.get_metadata()


def load_document(source: str, **options) -> str:
    """
    Convenience function to load a document in one call.

    Args:
        source: Path to file, URL, or Google Doc ID/URL
        **options: Loader-specific options

    Returns:
        Extracted text content

    Example:
        ```python
        # Load PDF
        text = load_document("document.pdf")

        # Load web page
        text = load_document("https://example.com")

        # Load Google Doc
        text = load_document("https://docs.google.com/document/d/DOC_ID/edit")

        # Load image with OCR
        text = load_document("image.jpg", language='eng')

        # Load JavaScript-heavy website
        text = load_document("https://spa-app.com", javascript=True)
        ```
    """
    loader = DocumentLoader(source, **options)
    return loader.load()
