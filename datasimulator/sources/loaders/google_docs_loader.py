"""
Google Docs loader.

Loads documents from Google Docs using the Google Docs API.

Setup:
1. Enable Google Docs API in Google Cloud Console
2. Create OAuth credentials or Service Account
3. Download credentials JSON file
4. Set GOOGLE_APPLICATION_CREDENTIALS env variable
"""

import logging
import os
import re
from typing import Optional, List, Dict, Any

from ..base_loader import BaseLoader, LoaderException

logger = logging.getLogger(__name__)


class GoogleDocsLoader(BaseLoader):
    """
    Load and extract text from Google Docs.

    Requires:
    - google-auth
    - google-api-python-client

    Authentication:
    - Set GOOGLE_APPLICATION_CREDENTIALS to path of credentials JSON
    - Or provide credentials_path parameter
    """

    def __init__(
        self,
        source: str,
        credentials_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize Google Docs loader.

        Args:
            source: Google Doc URL or document ID
            credentials_path: Path to Google credentials JSON file
            **kwargs: Additional options
        """
        super().__init__(source, **kwargs)
        self.credentials_path = credentials_path or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
        self.document_id = self._extract_doc_id(source)

    def _extract_doc_id(self, source: str) -> str:
        """
        Extract document ID from Google Docs URL or return as-is if already an ID.

        Supported formats:
        - https://docs.google.com/document/d/DOCUMENT_ID/edit
        - DOCUMENT_ID
        """
        # Check if it's a URL
        if source.startswith('http'):
            # Extract ID from URL
            match = re.search(r'/document/d/([a-zA-Z0-9-_]+)', source)
            if match:
                return match.group(1)
            else:
                raise LoaderException(f"Could not extract document ID from URL: {source}")
        else:
            # Assume it's already a document ID
            return source

    def load(self) -> str:
        """
        Load text from Google Doc.

        Returns:
            Extracted text content
        """
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "Google API libraries not installed. "
                "Install with: pip install google-auth google-api-python-client"
            )

        if not self.credentials_path:
            raise LoaderException(
                "Google credentials not configured. "
                "Set GOOGLE_APPLICATION_CREDENTIALS environment variable "
                "or provide credentials_path parameter."
            )

        logger.info(f"Loading Google Doc: {self.document_id}")

        try:
            # Load credentials
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path,
                scopes=['https://www.googleapis.com/auth/documents.readonly']
            )

            # Build service
            service = build('docs', 'v1', credentials=credentials)

            # Retrieve document
            document = service.documents().get(documentId=self.document_id).execute()

            # Extract text from document structure
            content = self._extract_text_from_document(document)

            # Store metadata
            self.metadata = {
                'document_id': self.document_id,
                'title': document.get('title', ''),
                'char_count': len(content),
                'method': 'google_docs_api'
            }

            logger.info(
                f"Loaded Google Doc '{document.get('title', 'Untitled')}': "
                f"{len(content)} chars"
            )

            return content

        except Exception as e:
            raise LoaderException(f"Error loading Google Doc: {e}")

    def _extract_text_from_document(self, document: Dict[str, Any]) -> List[str]:
        """
        Extract text from Google Docs API document structure.

        The document structure contains nested elements.
        """
        text_parts = []

        # Get document content
        content = document.get('body', {}).get('content', [])

        for element in content:
            text = self._read_structural_elements(element)
            if text:
                text_parts.append(text)

        return '\n\n'.join(text_parts)

    def _read_structural_elements(self, elements: Dict[str, Any]) -> str:
        """
        Recursively read structural elements from Google Doc.

        Adapted from Google's documentation examples.
        """
        text = ''

        if 'paragraph' in elements:
            para_elements = elements.get('paragraph', {}).get('elements', [])
            for elem in para_elements:
                text_run = elem.get('textRun')
                if text_run:
                    text += text_run.get('content', '')

        elif 'table' in elements:
            # Extract text from tables
            table = elements.get('table', {})
            for row in table.get('tableRows', []):
                for cell in row.get('tableCells', []):
                    cell_content = cell.get('content', [])
                    for element in cell_content:
                        text += self._read_structural_elements(element)
                text += ' | '
            text += '\n'

        elif 'tableOfContents' in elements:
            # Skip table of contents
            pass

        return text
