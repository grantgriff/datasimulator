"""
Web scraper loader.

Extracts text content from web pages using:
- BeautifulSoup: HTML parsing
- requests: HTTP client
"""

import logging
from typing import Optional, List
from urllib.parse import urlparse

from ..base_loader import BaseLoader, LoaderException

logger = logging.getLogger(__name__)


class WebScraperLoader(BaseLoader):
    """
    Load and extract text from web pages.

    Supports:
    - HTTP/HTTPS URLs
    - Automatic HTML cleaning
    - Removes scripts, styles, and navigation elements
    """

    def __init__(
        self,
        source: str,
        timeout: int = 30,
        headers: Optional[dict] = None,
        **kwargs
    ):
        """
        Initialize web scraper.

        Args:
            source: URL to scrape
            timeout: Request timeout in seconds
            headers: Optional HTTP headers
            **kwargs: Additional options
        """
        super().__init__(source, **kwargs)
        self.timeout = timeout
        self.headers = headers or {
            'User-Agent': 'DataSimulator/0.1.0 (Web Content Loader)'
        }

    def load(self) -> str:
        """
        Load text from web page.

        Returns:
            Extracted text content
        """
        # Validate URL
        parsed = urlparse(self.source)
        if not parsed.scheme in ('http', 'https'):
            raise LoaderException(f"Invalid URL scheme: {parsed.scheme}")

        try:
            import requests
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError(
                "Required libraries not installed. "
                "Install with: pip install requests beautifulsoup4"
            )

        logger.info(f"Scraping web page: {self.source}")

        try:
            # Fetch page
            response = requests.get(
                self.source,
                timeout=self.timeout,
                headers=self.headers
            )
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()

            # Extract text from main content areas
            # Try to find main content first
            main_content = soup.find('main') or soup.find('article') or soup.find('body')

            if main_content:
                # Get text from paragraphs and headings
                text_elements = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                paragraphs = [elem.get_text(strip=True) for elem in text_elements]
            else:
                # Fallback: get all text
                paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]

            # Filter out empty paragraphs
            paragraphs = [p for p in paragraphs if p]

            # Combine paragraphs
            content = "\n\n".join(paragraphs)

            # Store metadata
            title = soup.find('title')
            self.metadata = {
                'url': self.source,
                'title': title.get_text(strip=True) if title else None,
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type'),
                'char_count': len(content),
                'paragraph_count': len(paragraphs),
                'method': 'beautifulsoup'
            }

            logger.info(
                f"Scraped {len(paragraphs)} paragraphs, {len(content)} chars from {self.source}"
            )

            return content

        except requests.RequestException as e:
            raise LoaderException(f"Error fetching URL {self.source}: {e}")
        except Exception as e:
            raise LoaderException(f"Error scraping web page: {e}")


class JavaScriptWebLoader(BaseLoader):
    """
    Load content from JavaScript-heavy websites using Playwright.

    Use this for single-page applications (SPAs) and dynamic content.
    """

    def __init__(
        self,
        source: str,
        wait_for: Optional[str] = None,
        timeout: int = 30000,
        **kwargs
    ):
        """
        Initialize JavaScript web loader.

        Args:
            source: URL to load
            wait_for: CSS selector to wait for before extracting
            timeout: Page load timeout in milliseconds
            **kwargs: Additional options
        """
        super().__init__(source, **kwargs)
        self.wait_for = wait_for
        self.timeout = timeout

    def load(self) -> str:
        """
        Load text from JavaScript-rendered page.

        Returns:
            Extracted text content
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise ImportError(
                "Playwright not installed. Install with:\n"
                "  pip install playwright\n"
                "  playwright install chromium"
            )

        logger.info(f"Loading JavaScript page: {self.source}")

        try:
            with sync_playwright() as p:
                # Launch browser
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()

                # Navigate to page
                page.goto(self.source, timeout=self.timeout)

                # Wait for specific element if provided
                if self.wait_for:
                    page.wait_for_selector(self.wait_for, timeout=self.timeout)
                else:
                    page.wait_for_load_state('networkidle')

                # Get page content
                content = page.inner_text('body')

                # Get title
                title = page.title()

                browser.close()

            # Store metadata
            self.metadata = {
                'url': self.source,
                'title': title,
                'char_count': len(content),
                'method': 'playwright'
            }

            logger.info(f"Loaded {len(content)} chars from {self.source}")

            return content

        except Exception as e:
            raise LoaderException(f"Error loading JavaScript page: {e}")
