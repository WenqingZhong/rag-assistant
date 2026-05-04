import requests
import time
import logging
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from typing import Optional
import xml.etree.ElementTree as ET
from src.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """
    A source-agnostic container for document metadata.
    
    The 'extra' field is a flexible dict for source-specific stuff:
    - arXiv: {"categories": ["cs.AI"], "arxiv_url": "..."}
    - Legal docs later: {"case_number": "...", "jurisdiction": "..."}
    """
    id: str
    source: str
    title: str
    authors: list[str]
    abstract: str
    published_date: Optional[datetime]
    url: str
    extra: dict


class ArxivFetcher:
    """
    Fetches paper metadata from the arXiv Atom XML API.
    Returns generic DocumentMetadata objects so the rest of the
    pipeline doesn't know or care that the source is arXiv.
    """

    # arXiv's API returns XML where every tag belongs to the Atom namespace.
    # Python's XML parser prepends this namespace URL to every tag name
    # internally, so we must include it when searching for tags.
    #
    # Example: <title> in the XML becomes "{http://...Atom}title" in Python.
    # We store it as a constant so we don't repeat this long string everywhere.
    
    NAMESPACE = "{http://www.w3.org/2005/Atom}"

    def __init__(self):
        self.settings = get_settings().arxiv
        # Track when we last made a request so we can enforce rate limiting.
        # arXiv asks developers to wait 3 seconds between requests.
        self._last_request_time = 0.0

    def _rate_limit(self):
        """
        Enforce a minimum delay between API requests.
        
        arXiv's terms of service require polite API usage.
        If we just hammered them with requests, they'd block us.
        
        This calculates how much time has passed since our last request,
        and sleeps for the remainder if we're under the required delay.
        """
        elapsed = time.time() - self._last_request_time
        if elapsed < self.settings.rate_limit_delay:
            time.sleep(self.settings.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def fetch_recent(self, days_back: int = 1) -> list[DocumentMetadata]:
        """
        Public entry point: fetch papers from the last N days.
        Delegates to fetch_by_date_range with calculated dates.
        """
        to_date = datetime.now(timezone.utc)
        from_date = to_date - timedelta(days=days_back)
        return self.fetch_by_date_range(from_date, to_date)

    def fetch_by_date_range(self, from_date: datetime, to_date: datetime) -> list[DocumentMetadata]:
        """
        Build the arXiv API query URL and fetch results.
        
        The arXiv API is a simple HTTP GET with query parameters. No authentication needed.
        Returns a list of DocumentMetadata parsed from the XML response.
        """
        self._rate_limit()

        # Build the query string. 'cat:cs.AI' means "category = AI papers".
        # sortBy=submittedDate gets us the newest papers first.
        query = (
            f"search_query=cat:{self.settings.search_category}"
            f"&start=0"
            f"&max_results={self.settings.max_results}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )

        url = f"{self.settings.base_url}?{query}"
        logger.info(f"Fetching arXiv papers: {url}")

        try:
            response = requests.get(url, timeout=self.settings.timeout_seconds)
            # raise_for_status() throws an exception if HTTP status is 4xx or 5xx.
            response.raise_for_status()
            return self._parse_response(response.text)
        except Exception as e:
            logger.error(f"Failed to fetch from arXiv: {e}")
            return []  # return empty list instead of crashing the pipeline

    def _parse_response(self, xml_text: str) -> list[DocumentMetadata]:
        """
        Parse the Atom XML response from arXiv into DocumentMetadata objects.
        
        The XML structure looks like:
        <feed>               ← root
            <entry>          ← one paper
                <id>...</id>
                <title>...</title>
                <author><name>...</name></author>
                <summary>...</summary>   ← this is the abstract
                <published>...</published>
                <link title="pdf" href="..."/>
                <category term="cs.AI"/>
            </entry>
            <entry>...</entry>
            ...
        </feed>
        
        ET.fromstring() parses the XML string into a tree of Element objects.
        """
        root = ET.fromstring(xml_text)
        results = []

        # Find all <entry> elements under root.
        # Note the NAMESPACE prefix — without it, findall("entry") returns nothing.
        for entry in root.findall(f"{self.NAMESPACE}entry"):
            try:
                # entry.find() returns the first matching child element.
                # .text gets the text content between the tags.
                # e.g. <id>http://arxiv.org/abs/2301.07041v1</id> → .text is the full URL
                raw_id = entry.find(f"{self.NAMESPACE}id").text
                # We only want "2301.07041", not the full URL
                arxiv_id = raw_id.split("/abs/")[-1]

                title = entry.find(f"{self.NAMESPACE}title").text.strip()
                # <summary> is what arXiv calls the abstract
                abstract = entry.find(f"{self.NAMESPACE}summary").text.strip()

                # There can be multiple <author> tags — collect all of them.
                authors = [
                    author.find(f"{self.NAMESPACE}name").text
                    for author in entry.findall(f"{self.NAMESPACE}author")
                ]

                # Parse the ISO 8601 date string into a Python datetime object.
                # e.g. "2023-01-17T18:15:23Z" → datetime(2023, 1, 17, 18, 15, 23, utc)
                published_str = entry.find(f"{self.NAMESPACE}published").text
                published_date = datetime.fromisoformat(published_str.replace("Z", "+00:00"))

                # Find the PDF link specifically — there are multiple <link> tags
                # (one for the abstract page, one for the PDF).
                # We want the one with title="pdf".
                pdf_url = None
                for link in entry.findall(f"{self.NAMESPACE}link"):
                    if link.get("title") == "pdf":   # .get() reads an XML attribute
                        pdf_url = link.get("href")

                # Collect all category tags (a paper can belong to multiple)
                categories = [
                    tag.get("term")
                    for tag in entry.findall(f"{self.NAMESPACE}category")
                ]

                results.append(DocumentMetadata(
                    id=arxiv_id,
                    source="arxiv",
                    title=title,
                    authors=authors,
                    abstract=abstract,
                    published_date=published_date,
                    url=pdf_url or raw_id,  # fall back to abstract URL if no PDF link
                    extra={
                        "categories": categories,
                        "arxiv_url": raw_id,
                    },
                ))

            except Exception as e:
                # If one paper fails to parse, log it and continue.
                # We don't want one bad entry to crash the whole batch.
                logger.warning(f"Failed to parse entry: {e}")
                continue

        logger.info(f"Fetched {len(results)} papers from arXiv")
        return results