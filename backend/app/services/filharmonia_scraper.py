"""
Scraper for filharmonia.pl concert programs.
Fetches concert details (pieces, performers) for a given date.
Supports both HTML program pages and PDF brochures (common for children's concerts).
"""

import io
import logging
import re
import time
from datetime import datetime
from typing import Optional

import pdfplumber
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

    def _fetch_bytes(self, url: str) -> Optional[bytes]:
        full_url = url if url.startswith("http") else BASE_URL + url
        self._rate_limit()
        try:
            resp = self._session.get(full_url, timeout=30)
            resp.raise_for_status()
            return resp.content
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

        # Program pieces: try HTML first, fallback to PDF brochure
        pieces = self._parse_pieces(soup)
        if not pieces:
            pdf_url = self._find_pdf_url(soup)
            if pdf_url:
                logger.info(f"No HTML program found, trying PDF: {pdf_url}")
                pieces = self._parse_pdf_program(pdf_url)

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

    def _find_pdf_url(self, soup: BeautifulSoup) -> Optional[str]:
        """Find a program PDF link on the concert page."""
        downloads = soup.find("div", class_="item-downloads")
        if downloads:
            for link in downloads.find_all("a", href=True):
                if link["href"].endswith(".pdf"):
                    return link["href"]
        # Fallback: any PDF link with "program" in text or URL
        for link in soup.find_all("a", href=True):
            href = link["href"]
            if href.endswith(".pdf") and "program" in (href + link.get_text()).lower():
                return href
        return None

    def _parse_pdf_program(self, pdf_url: str) -> list[ProgramPiece]:
        """Download a PDF program brochure and extract pieces from it."""
        pdf_bytes = self._fetch_bytes(pdf_url)
        if not pdf_bytes:
            return []

        try:
            return self._extract_pieces_from_pdf(pdf_bytes)
        except Exception as e:
            logger.error(f"Failed to parse PDF program {pdf_url}: {e}")
            return []

    def _extract_pieces_from_pdf(self, pdf_bytes: bytes) -> list[ProgramPiece]:
        """Parse program pieces from PDF text content."""
        all_text = ""
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text or "Program" not in text:
                    continue

                # Detect two-column layout and re-extract in correct order
                if self._is_two_column_page(page):
                    text = self._extract_two_column_text(page)

                all_text += text + "\n"

        if not all_text:
            return []

        # Find the "Program" section
        program_match = re.search(r'^Program\s*$', all_text, re.MULTILINE)
        if not program_match:
            return []

        program_text = all_text[program_match.end():]

        # Cut off footer text (invitations to next concert, info about latecomers, etc.)
        for cutoff in [
            r'Na następn',
            r'Uprzejmie informujemy',
            r'Dyrektor\s+(Naczelna|Artystyczny)',
            r'opracowanie muzyczne',
            r'opracowanie wszystkich',
            r'[Ww]\s*opracowaniu',
            r'[Ww]szystkie utwory',
            r'●\s*utwory',
        ]:
            cutoff_match = re.search(cutoff, program_text)
            if cutoff_match:
                program_text = program_text[:cutoff_match.start()]

        # Pre-process lines: clean bullets, merge split composer names
        program_text = self._clean_pdf_text(program_text)

        return self._parse_pdf_lines(program_text)

    def _is_two_column_page(self, page) -> bool:
        """Detect two-column layout by analyzing word x-positions for a gap in the middle."""
        words = page.extract_words()
        if len(words) < 10:
            return False

        w = page.width
        # Check if there's a clear vertical gap in the middle third of the page
        mid_left = w * 0.35
        mid_right = w * 0.65

        # Count words whose center falls in the left, middle, and right zones
        left_count = 0
        right_count = 0
        mid_count = 0
        for word in words:
            cx = (word["x0"] + word["x1"]) / 2
            if cx < mid_left:
                left_count += 1
            elif cx > mid_right:
                right_count += 1
            else:
                mid_count += 1

        # Two-column if both sides have significant text and the middle gap is sparse
        has_both_sides = left_count >= 5 and right_count >= 5
        gap_is_sparse = mid_count < (left_count + right_count) * 0.15
        return has_both_sides and gap_is_sparse

    def _extract_two_column_text(self, page) -> str:
        """Extract text from a two-column page by reading left column then right column."""
        words = page.extract_words()
        if not words:
            return page.extract_text() or ""

        # Find the split point: largest horizontal gap between word clusters
        x_starts = sorted(set(round(w["x0"]) for w in words))
        w = page.width
        best_gap_x = w / 2
        best_gap_size = 0

        # Only look for gaps in the middle 30-70% of the page
        for j in range(len(x_starts) - 1):
            gap_start = x_starts[j]
            gap_end = x_starts[j + 1]
            gap_center = (gap_start + gap_end) / 2
            if w * 0.3 < gap_center < w * 0.7:
                gap_size = gap_end - gap_start
                if gap_size > best_gap_size:
                    best_gap_size = gap_size
                    best_gap_x = gap_center

        # Crop at the split point and extract each column
        left_crop = page.crop((0, 0, best_gap_x, page.height))
        right_crop = page.crop((best_gap_x, 0, w, page.height))
        left_text = left_crop.extract_text() or ""
        right_text = right_crop.extract_text() or ""
        return left_text + "\n" + right_text

    def _clean_pdf_text(self, text: str) -> str:
        """Pre-process PDF text: remove bullet markers, merge split composer lines."""
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            # Remove standalone bullet markers (• used as decorative separators)
            line = line.strip()
            if line in ("•", "·", "∙"):
                continue
            # Remove leading/trailing bullets
            line = re.sub(r'^[•·∙]\s*', '', line)
            line = re.sub(r'\s*[•·∙]$', '', line)
            if line:
                cleaned.append(line)

        # Merge composer lines split across line breaks (ending with "/")
        merged = []
        i = 0
        while i < len(cleaned):
            line = cleaned[i]
            if line.rstrip().endswith("/") and i + 1 < len(cleaned):
                # Merge with next line (multi-composer credit split across lines)
                merged.append(line.rstrip() + " " + cleaned[i + 1].strip())
                i += 2
            else:
                merged.append(line)
                i += 1

        return "\n".join(merged)

    def _parse_pdf_lines(self, text: str) -> list[ProgramPiece]:
        """Parse cleaned program text into ProgramPiece objects.

        Uses strict alternation: composer line, then title line(s), repeat.
        Multi-line titles are detected via continuation patterns (lowercase start,
        parenthetical annotations). Multiple titles under one composer are detected
        via musical terminology in what would otherwise be the next composer line.
        Special handling for "(wybór)" (selection) titles where sub-fragments follow.
        """
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        pieces: list[ProgramPiece] = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Break / intermission
            if "przerwa" in line.lower():
                pieces.append(ProgramPiece(title="Przerwa", is_break=True))
                i += 1
                continue

            # Footer-like lines to skip
            if self._is_pdf_footer_line(line):
                i += 1
                continue

            # Step 1: This line is a composer
            composer = line
            i += 1

            # Step 2: Read title(s) for this composer
            while i < len(lines):
                if "przerwa" in lines[i].lower():
                    break
                if self._is_pdf_footer_line(lines[i]):
                    i += 1
                    continue

                # Read one title (first line + continuations)
                title_parts = [lines[i]]
                i += 1

                # Collect continuation lines (lowercase start or parenthetical)
                while i < len(lines):
                    nl = lines[i]
                    if self._is_title_continuation(nl):
                        title_parts.append(nl)
                        i += 1
                    else:
                        break

                full_title = " ".join(title_parts)

                # Extract annotation
                annotation = None
                ann_match = re.search(r'\(([^)]+)\)\s*$', full_title)
                if ann_match:
                    potential_ann = ann_match.group(1)
                    if any(kw in potential_ann.lower() for kw in [
                        'fragment', 'oprac', 'wersja', 'arr.', 'wybór',
                    ]):
                        annotation = potential_ann
                        full_title = full_title[:ann_match.start()].strip()

                # Clean up ● markers
                full_title = full_title.replace(" ●", "").replace("●", "").strip()

                # If title is empty (e.g. only annotation was present), the
                # "composer" line was actually the title of a piece without composer
                if not full_title:
                    full_title = composer
                    composer = ""

                # Check if this is a "selection" title (wybór/Fragmenty)
                # If so, following lines are sub-fragment titles by the same composer
                is_selection = (
                    annotation and "wybór" in annotation.lower()
                ) or "fragmenty" in full_title.lower()

                pieces.append(ProgramPiece(
                    composer=composer,
                    title=full_title,
                    annotation=annotation,
                ))

                if is_selection:
                    # Consume sub-fragment titles until we hit a new composer,
                    # break, or footer
                    i = self._consume_selection_fragments(
                        lines, i, composer, full_title, pieces,
                    )

                # Check if next line is another title by same composer
                # (has musical terminology) or a new composer
                if i < len(lines) and "przerwa" not in lines[i].lower():
                    if self._is_musical_title(lines[i]):
                        continue  # Another title by same composer
                    else:
                        break  # New composer

        return pieces

    def _consume_selection_fragments(
        self,
        lines: list[str],
        i: int,
        composer: str,
        parent_title: str,
        pieces: list[ProgramPiece],
    ) -> int:
        """After a '(wybór)' title, consume sub-fragment titles as separate pieces."""
        while i < len(lines):
            line = lines[i]
            if "przerwa" in line.lower():
                break
            if self._is_pdf_footer_line(line):
                i += 1
                continue
            # A new composer is detected by lookahead: this line is followed by
            # a title containing strong indicators (z baletu, z filmu, Fragmenty, etc.)
            if self._looks_like_new_composer(lines, i):
                break
            # This is a sub-fragment title
            # Collect any continuation lines
            title_parts = [line]
            i += 1
            while i < len(lines):
                nl = lines[i]
                if self._is_title_continuation(nl):
                    title_parts.append(nl)
                    i += 1
                else:
                    break

            frag_title = " ".join(title_parts)
            # Extract annotation from fragment title
            annotation = None
            ann_match = re.search(r'\(([^)]+)\)\s*$', frag_title)
            if ann_match:
                potential_ann = ann_match.group(1)
                if any(kw in potential_ann.lower() for kw in [
                    'fragment', 'oprac', 'wersja', 'arr.',
                ]):
                    annotation = potential_ann
                    frag_title = frag_title[:ann_match.start()].strip()

            frag_title = frag_title.replace(" ●", "").replace("●", "").strip()
            if frag_title:
                pieces.append(ProgramPiece(
                    composer=composer,
                    title=f"{parent_title}: {frag_title}",
                    annotation=annotation,
                ))
        return i

    def _looks_like_new_composer(self, lines: list[str], idx: int) -> bool:
        """Check if lines[idx] is a new composer by looking ahead at lines[idx+1]."""
        line = lines[idx]
        # Lines with ":" are likely fragment/excerpt titles, not composers
        if ":" in line:
            return False
        # Lines with "(fragment)" or "(wybór)" are titles
        if "(" in line and any(kw in line.lower() for kw in ['fragment', 'wybór']):
            return False
        if idx + 1 >= len(lines):
            return False
        next_line = lines[idx + 1]
        next_lower = next_line.lower()
        # Strong title indicators in the following line suggest current line is a composer
        strong_indicators = [
            'z baletu', 'z filmu', 'z opery', 'z suity', 'ze zbioru',
            'z musicalu', 'z cyklu', 'z muzyki', 'fragmenty', '(wybór)',
            'op.', 'kv ', 'bwv ',
        ]
        if any(ind in next_lower for ind in strong_indicators):
            return True
        # Also check if the following line starts with a musical form word
        if self._is_musical_title(next_line):
            return True
        return False

    def _is_title_continuation(self, line: str) -> bool:
        """Check if a line is a continuation of the previous title."""
        if not line:
            return False
        # Lines starting with "(" are annotations
        if line.startswith("("):
            return True
        # Lines starting with lowercase are continuations
        if line[0].islower():
            return True
        return False

    def _is_musical_title(self, line: str) -> bool:
        """Check if a line contains musical terminology, indicating it's a
        piece title rather than a composer name."""
        lower = line.lower()
        musical_terms = [
            'op.', ' nr ', 'kv ', 'bwv ', 'cz.', ' k ',
            '-dur', '-moll',
            'z filmu', 'z baletu', 'z opery', 'z suity', 'ze zbioru',
            'z musicalu', 'z cyklu', 'z muzyki',
            'sł.', 'oprac.',
        ]
        if any(term in lower for term in musical_terms):
            return True
        # Musical form words at the start
        form_words = [
            'sonata', 'sonatina', 'kwintet', 'kwartet', 'trio', 'duet',
            'koncert', 'symfoni', 'uwertura', 'suita', 'nokturn',
            'preludium', 'fuga', 'walc', 'mazur', 'polone', 'taniec',
            'marsz', 'pieśń', 'pieśn', 'aria', 'romans', 'etiuda',
            'fantazja', 'rapsod', 'scherzo', 'rondo', 'menuet',
            'variazioni', 'allegro', 'adagio', 'andante',
        ]
        first_word = lower.split()[0] if lower.split() else ""
        if any(first_word.startswith(fw) for fw in form_words):
            return True
        return False

    def _is_pdf_footer_line(self, line: str) -> bool:
        """Check if a line is a footer/metadata line to skip."""
        lower = line.lower()
        return any(kw in lower for kw in [
            'opracowanie muzyczne utworów',
            'opracowanie wszystkich',
            'w opracowaniu',
            'wszystkie utwory',
            'na następn',
            'uprzejmie informujemy',
        ])

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
