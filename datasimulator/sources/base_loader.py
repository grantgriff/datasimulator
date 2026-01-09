"""
Base classes for document loaders.

All loaders inherit from BaseLoader and implement the load() method.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class LoaderException(Exception):
    """Base exception for loader errors."""
    pass


class BaseLoader(ABC):
    """
    Abstract base class for all document loaders.

    Subclasses must implement the load() method to extract text from their
    specific document format.
    """

    def __init__(self, source: str, **kwargs):
        """
        Initialize loader.

        Args:
            source: Path to file or URL
            **kwargs: Additional loader-specific options
        """
        self.source = source
        self.options = kwargs
        self.metadata: Dict[str, Any] = {}

    @abstractmethod
    def load(self) -> str:
        """
        Load and extract text from the source.

        Returns:
            Extracted text content

        Raises:
            LoaderException: If loading fails
        """
        pass

    def _validate_file_exists(self, path: Path) -> bool:
        """Validate that a file exists."""
        if not path.exists():
            raise LoaderException(f"File not found: {path}")
        if not path.is_file():
            raise LoaderException(f"Not a file: {path}")
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the loaded document.

        Returns:
            Dictionary with metadata (pages, file_size, etc.)
        """
        return self.metadata.copy()


class TextLoader(BaseLoader):
    """Loader for plain text files."""

    SUPPORTED_EXTENSIONS = ['.txt', '.md', '.markdown', '.text']

    def load(self) -> str:
        """Load text from file."""
        try:
            path = Path(self.source)
            self._validate_file_exists(path)

            encoding = self.options.get('encoding', 'utf-8')

            with open(path, 'r', encoding=encoding) as f:
                content = f.read()

            # Store metadata
            self.metadata = {
                'file_path': str(path),
                'file_size': path.stat().st_size,
                'encoding': encoding,
                'char_count': len(content),
                'line_count': content.count('\n') + 1
            }

            logger.info(f"Loaded text file: {path} ({len(content)} chars)")
            return content

        except UnicodeDecodeError as e:
            raise LoaderException(f"Encoding error reading {self.source}: {e}")
        except Exception as e:
            raise LoaderException(f"Error loading text file {self.source}: {e}")
