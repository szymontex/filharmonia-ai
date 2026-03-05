"""
Scraper for filharmonia.pl concert programs.
Fetches concert details (pieces, performers) for a given date.
"""

import logging
import re
import time
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel

logger = logging.getLogger(__name__)

BASE_URL = "https://filharmonia.pl"


class ProgramPiece(BaseModel):
    composer: str = ""
    title: str = ""
    duration_min: Optional[int] = None
    annotation: Optional[str] = None
    is_break: bool = False


class ConcertProgram(BaseModel):
    title: str
    date: str  # YYYY-MM-DD
    time: str  # HH:MM
    venue: str = ""
    conductor: Optional[str] = None
    soloists: list[str] = []
    orchestra: Optional[str] = None
    pieces: list[ProgramPiece] = []
    url: str


class FilharmoniaScraper:
    def __init__(self):
        self._cache: dict[str, tuple[float, object]] = {}
        self._cache_ttl = 3600  # 1 hour
        self._last_request_time = 0.0
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "pl,en;q=0.9",
        })

    def _rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < 1.0:
            time.sleep(1.0 - elapsed)
        self._last_request_time = time.time()

    def _get_cached(self, key: str):
        if key in self._cache:
            ts, val = self._cache[key]
            if time.time() - ts < self._cache_ttl:
                return val
            del self._cache[key]
        return None

    def _set_cached(self, key: str, val):
        self._cache[key] = (time.time(), val)

    def _fetch(self, url: str) -> Optional[BeautifulSoup]:
        full_url = url if url.startswith("http") else BASE_URL + url
        self._rate_limit()
        try:
            resp = self._session.get(full_url, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            logger.error(f"Failed to fetch {full_url}: {e}")
            return None

    def search_concerts(self, date: str) -> list[ConcertProgram]:
        """Search concerts for a given date (YYYY-MM-DD format)."""
        cache_key = f"search:{date}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        try:
            dt = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {date}")
            return []

        # The calendar page uses day-preview divs with id="d-prev-YYYYMMDD"
        date_id = dt.strftime("%Y%m%d")

        # Calculate UTC midnight timestamp for the first day of the month
        from calendar import timegm
        ts = timegm((dt.year, dt.month, 1, 0, 0, 0))

        # Try: current page first, then with explicit ts, then archive
        concerts = []
        for page_url in ["/repertuar", f"/repertuar,ts:{ts}", f"/repertuar,m:archive,ts:{ts}"]:
            soup = self._fetch(page_url)
            if not soup:
                continue

            # Find the day-preview div for our date
            day_div = soup.find("div", id=f"d-prev-{date_id}")
            if not day_div:
                continue

            # Find all concert links within this day
            links = day_div.find_all("a", class_="event-list-chocolate")
            for link in links:
                href = link.get("href", "")
                if not href or not href.startswith("/repertuar/"):
                    continue

                # Extract basic info from the listing
                title_el = link.find("div", class_="title-attr")
                title = title_el.get_text(strip=True) if title_el else ""

                date_el = link.find("div", class_="event-date")
                date_text = date_el.find("span").get_text(strip=True) if date_el and date_el.find("span") else ""

                time_el = link.find("span", class_="time")
                time_text = time_el.get_text(strip=True) if time_el else ""

                # Get full details from the concert page
                details = self.get_concert_details(href)
                if details:
                    concerts.append(details)
                else:
                    # Fallback: basic info only
                    concerts.append(ConcertProgram(
                        title=title,
                        date=date,
                        time=time_text,
                        url=BASE_URL + href,
                    ))

            if concerts:
                break  # Found concerts, no need to check archive

        self._set_cached(cache_key, concerts)
        return concerts

    def get_concert_details(self, url: str) -> Optional[ConcertProgram]:
        """Get full concert details from a concert page URL."""
        full_url = url if url.startswith("http") else BASE_URL + url
        cache_key = f"detail:{full_url}"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        soup = self._fetch(url)
        if not soup:
            return None

        try:
            result = self._parse_concert_page(soup, full_url)
            if result:
                self._set_cached(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Failed to parse concert page {full_url}: {e}")
            return None

    def _parse_concert_page(self, soup: BeautifulSoup, url: str) -> Optional[ConcertProgram]:
        # Title
        title_el = soup.find("h2", class_="title-in-sidebar")
        title = title_el.get_text(strip=True) if title_el else ""

        # Date and time from event-meta-date-full
        meta_date = soup.find("div", class_="event-meta-date-full")
        date_str = ""
        time_str = ""
        if meta_date:
            date_el = meta_date.find("div", class_="event-date")
            if date_el and date_el.find("span"):
                date_str = date_el.find("span").get_text(strip=True)  # e.g., "6.03"

            time_el = meta_date.find("span", class_="time")
            if time_el:
                time_str = time_el.get_text(strip=True)  # e.g., "19:30"

        # Try to get year from ticket link (bilety24 has full date like 6-03-2026)
        year_hint = None
        ticket_link = soup.find("a", class_="buy-tickets")
        if ticket_link:
            ticket_href = ticket_link.get("href", "")
            year_match = re.search(r'-(\d{1,2})-(\d{2})-(\d{4})-', ticket_href)
            if year_match:
                year_hint = int(year_match.group(3))

        # Convert date_str (D.MM) to YYYY-MM-DD
        iso_date = self._parse_date(date_str, url, year_hint=year_hint)

        # Venue
        venue_el = soup.find("div", class_="venue-str")
        venue = venue_el.get_text(strip=True) if venue_el else ""

        # Performers
        performers_section = soup.find("div", class_="event-meta-performers")
        conductor = None
        soloists = []
        orchestra = None

        if performers_section:
            artist_links = performers_section.find_all("a", class_="artist-list")
            for artist_link in artist_links:
                name_el = artist_link.find("div", class_="artist-name")
                role_el = artist_link.find("div", class_="artist-role")
                name = name_el.get_text(strip=True) if name_el else ""
                role = role_el.get_text(strip=True) if role_el else ""

                if not name:
                    continue

                role_lower = role.lower()
                if "dyrygent" in role_lower or "conductor" in role_lower:
                    conductor = name
                elif not role and ("orkiestr" in name.lower() or "sinfoni" in name.lower() or "philharmon" in name.lower() or "filharmon" in name.lower()):
                    orchestra = name
                elif role:
                    soloists.append(f"{name} ({role})")
                else:
                    # No role - could be orchestra or ensemble
                    if orchestra is None:
                        orchestra = name
                    else:
                        soloists.append(name)

        # Program pieces
        pieces = self._parse_pieces(soup)

        return ConcertProgram(
            title=title,
            date=iso_date,
            time=time_str,
            venue=venue,
            conductor=conductor,
            soloists=soloists,
            orchestra=orchestra,
            pieces=pieces,
            url=url,
        )

    def _parse_date(self, date_str: str, url: str, year_hint: Optional[int] = None) -> str:
        """Convert 'D.MM' to 'YYYY-MM-DD'. Infer year from URL or current year."""
        if not date_str:
            # Try to extract from URL (e.g., ...-20260323)
            match = re.search(r'(\d{4})(\d{2})(\d{2})$', url.rstrip('/'))
            if match:
                return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            return ""

        parts = date_str.split(".")
        if len(parts) != 2:
            return ""

        day = parts[0].strip()
        month = parts[1].strip()

        # Infer year: from hint (ticket link), from URL YYYYMMDD, or current year
        year = year_hint or datetime.now().year
        if not year_hint:
            url_year_match = re.search(r'-(\d{4})(\d{2})(\d{2})(?:[^/\d]|$)', url)
            if url_year_match:
                year = int(url_year_match.group(1))

        try:
            return f"{year}-{int(month):02d}-{int(day):02d}"
        except ValueError:
            return ""

    def _parse_pieces(self, soup: BeautifulSoup) -> list[ProgramPiece]:
        """Parse program pieces from the tracks-wrapper section."""
        tracks_wrapper = soup.find("div", class_="tracks-wrapper")
        if not tracks_wrapper:
            return []

        pieces = []
        track_lists = tracks_wrapper.find_all("div", class_="track-list")

        for track in track_lists:
            # Check if this is a break (przerwa)
            comp_title_el = track.find("div", class_="composition-title")
            if not comp_title_el:
                continue

            comp_text = comp_title_el.get_text(strip=True)

            # Check for break/intermission
            if "przerwa" in comp_text.lower():
                duration = self._extract_duration(comp_title_el)
                pieces.append(ProgramPiece(
                    title="Przerwa",
                    is_break=True,
                    duration_min=duration,
                ))
                continue

            # Normal piece: composer from artist-name, title from composition-title
            composer_el = track.find("div", class_="artist-name")
            composer = composer_el.get_text(strip=True) if composer_el else ""

            # Get title text (excluding the duration span)
            duration = self._extract_duration(comp_title_el)

            # Get clean title (remove duration span text)
            time_span = comp_title_el.find("span", class_="time")
            if time_span:
                time_span.decompose()

            title = comp_title_el.get_text(strip=True)
            # Clean up HTML entities
            title = title.replace("\xa0", " ").strip()

            # Extract annotation from parentheses at the end
            annotation = None
            ann_match = re.search(r'\(([^)]+)\)\s*$', title)
            if ann_match:
                annotation = ann_match.group(1)
                title = title[:ann_match.start()].strip()

            pieces.append(ProgramPiece(
                composer=composer,
                title=title,
                duration_min=duration,
                annotation=annotation,
            ))

        return pieces

    def _extract_duration(self, el) -> Optional[int]:
        """Extract duration in minutes from a span.time element like [36']."""
        time_span = el.find("span", class_="time")
        if not time_span:
            return None
        text = time_span.get_text(strip=True)
        match = re.search(r"\[(\d+)['\u2032]?\]", text)
        if match:
            return int(match.group(1))
        return None


# Singleton
_instance: Optional[FilharmoniaScraper] = None


def get_filharmonia_scraper() -> FilharmoniaScraper:
    global _instance
    if _instance is None:
        _instance = FilharmoniaScraper()
    return _instance
