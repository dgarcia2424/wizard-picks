"""
odds_historical_pull.py
=======================
One-time Playwright scraper for OddsPortal historical MLB odds (2023-2025).

Setup:
    pip install playwright pandas pyarrow
    playwright install chromium

Usage:
    python odds_historical_pull.py

Outputs per year (written to ./statcast_data/):
    odds_historical_{year}.parquet
    odds_historical_{year}_checkpoint.parquet  (saved every 50 games)
"""

import asyncio
import json
import logging
import re
import time
from datetime import datetime, date
from pathlib import Path
from typing import Optional

import pandas as pd
from playwright.async_api import async_playwright, Page, BrowserContext, TimeoutError as PlaywrightTimeoutError

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("./statcast_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2023, 2024, 2025]
CHECKPOINT_INTERVAL = 50
PAGE_SLEEP_SECONDS = 2.5
NAVIGATION_TIMEOUT = 60_000  # ms — OddsPortal can be slow; increased for Cloudflare challenge
# MLB regular season: ~2430 games / ~18 per page = ~135 pages. Cap at 200 for safety.
MAX_PAGES_PER_YEAR = 200

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "odds_historical_pull.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Team name normalisation
# ---------------------------------------------------------------------------
TEAM_ABBR_MAP: dict[str, str] = {
    # Standard 30 MLB teams
    "arizona diamondbacks": "ARI",
    "diamondbacks": "ARI",
    "arizona": "ARI",
    "atlanta braves": "ATL",
    "braves": "ATL",
    "atlanta": "ATL",
    "baltimore orioles": "BAL",
    "orioles": "BAL",
    "baltimore": "BAL",
    "boston red sox": "BOS",
    "red sox": "BOS",
    "boston": "BOS",
    "chicago cubs": "CHC",
    "cubs": "CHC",
    "chicago white sox": "CWS",
    "white sox": "CWS",
    "cincinnati reds": "CIN",
    "reds": "CIN",
    "cincinnati": "CIN",
    "cleveland guardians": "CLE",
    "guardians": "CLE",
    "cleveland": "CLE",
    "cleveland indians": "CLE",
    "colorado rockies": "COL",
    "rockies": "COL",
    "colorado": "COL",
    "detroit tigers": "DET",
    "tigers": "DET",
    "detroit": "DET",
    "houston astros": "HOU",
    "astros": "HOU",
    "houston": "HOU",
    "kansas city royals": "KC",
    "royals": "KC",
    "kansas city": "KC",
    "los angeles angels": "LAA",
    "angels": "LAA",
    "la angels": "LAA",
    "anaheim angels": "LAA",
    "los angeles dodgers": "LAD",
    "dodgers": "LAD",
    "la dodgers": "LAD",
    "miami marlins": "MIA",
    "marlins": "MIA",
    "miami": "MIA",
    "florida marlins": "MIA",
    "milwaukee brewers": "MIL",
    "brewers": "MIL",
    "milwaukee": "MIL",
    "minnesota twins": "MIN",
    "twins": "MIN",
    "minnesota": "MIN",
    "new york mets": "NYM",
    "mets": "NYM",
    "ny mets": "NYM",
    "new york yankees": "NYY",
    "yankees": "NYY",
    "ny yankees": "NYY",
    "new york": "NYY",  # fallback; disambiguate by context if needed
    "athletics": "OAK",
    "oakland athletics": "OAK",
    "oakland": "OAK",
    "las vegas athletics": "OAK",
    "philadelphia phillies": "PHI",
    "phillies": "PHI",
    "philadelphia": "PHI",
    "pittsburgh pirates": "PIT",
    "pirates": "PIT",
    "pittsburgh": "PIT",
    "san diego padres": "SD",
    "padres": "SD",
    "san diego": "SD",
    "san francisco giants": "SF",
    "giants": "SF",
    "san francisco": "SF",
    "seattle mariners": "SEA",
    "mariners": "SEA",
    "seattle": "SEA",
    "st. louis cardinals": "STL",
    "st louis cardinals": "STL",
    "cardinals": "STL",
    "st. louis": "STL",
    "st louis": "STL",
    "tampa bay rays": "TB",
    "rays": "TB",
    "tampa bay": "TB",
    "texas rangers": "TEX",
    "rangers": "TEX",
    "texas": "TEX",
    "toronto blue jays": "TOR",
    "blue jays": "TOR",
    "toronto": "TOR",
    "washington nationals": "WSH",
    "nationals": "WSH",
    "washington": "WSH",
    # OddsPortal-specific variants
    "chi. cubs": "CHC",
    "chi. white sox": "CWS",
    "chi cubs": "CHC",
    "chi white sox": "CWS",
    "n.y. mets": "NYM",
    "n.y. yankees": "NYY",
    "ny. mets": "NYM",
    "ny. yankees": "NYY",
    "l.a. angels": "LAA",
    "l.a. dodgers": "LAD",
    "k.c. royals": "KC",
    "s.d. padres": "SD",
    "s.f. giants": "SF",
    "t.b. rays": "TB",
    "oak. athletics": "OAK",
}


def normalize_team(raw: str) -> Optional[str]:
    """Convert a raw OddsPortal team string to 3-letter MLB abbreviation."""
    cleaned = raw.strip().lower()
    # Direct lookup
    if cleaned in TEAM_ABBR_MAP:
        return TEAM_ABBR_MAP[cleaned]
    # Suffix lookup (e.g. "New York Yankees" → try longer keys first)
    for key, abbr in sorted(TEAM_ABBR_MAP.items(), key=lambda x: -len(x[0])):
        if key in cleaned:
            return abbr
    log.warning("Unknown team name: %r", raw)
    return raw.upper()[:3]


# ---------------------------------------------------------------------------
# Date normalisation
# ---------------------------------------------------------------------------
MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def normalize_date(raw: str, year: int) -> Optional[str]:
    """
    Convert OddsPortal date strings to YYYY-MM-DD.
    Handles: 'Today', 'Yesterday', '11 Apr 2024', '11 Apr', 'Apr 11, 2024'.
    """
    raw = raw.strip()
    today = date.today()

    if raw.lower() == "today":
        return today.strftime("%Y-%m-%d")
    if raw.lower() == "yesterday":
        from datetime import timedelta
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")

    # Try various explicit patterns
    patterns = [
        r"(\d{1,2})\s+([A-Za-z]{3,})\s+(\d{4})",  # 11 Apr 2024
        r"([A-Za-z]{3,})\s+(\d{1,2}),?\s+(\d{4})",  # Apr 11, 2024
        r"(\d{1,2})\s+([A-Za-z]{3,})",              # 11 Apr (year inferred)
        r"(\d{2})/(\d{2})/(\d{4})",                 # 04/11/2024
        r"(\d{4})-(\d{2})-(\d{2})",                 # 2024-04-11 (already normalised)
    ]

    for pat in patterns:
        m = re.search(pat, raw)
        if not m:
            continue
        groups = m.groups()
        try:
            if len(groups) == 3:
                g0, g1, g2 = groups
                # Detect format
                if g0.isdigit() and not g1.isdigit() and g2.isdigit() and len(g2) == 4:
                    # DD Mon YYYY
                    month = MONTH_MAP.get(g1[:3].lower())
                    if month:
                        return f"{g2}-{month:02d}-{int(g0):02d}"
                elif not g0.isdigit() and g1.isdigit() and g2.isdigit() and len(g2) == 4:
                    # Mon DD YYYY
                    month = MONTH_MAP.get(g0[:3].lower())
                    if month:
                        return f"{g2}-{month:02d}-{int(g1):02d}"
                elif "/" in raw:
                    # MM/DD/YYYY
                    return f"{g2}-{int(g0):02d}-{int(g1):02d}"
                elif "-" in raw and len(g0) == 4:
                    return f"{g0}-{g1}-{g2}"
            elif len(groups) == 2:
                # DD Mon — infer year
                g0, g1 = groups
                if g0.isdigit():
                    month = MONTH_MAP.get(g1[:3].lower())
                    if month:
                        return f"{year}-{month:02d}-{int(g0):02d}"
        except (ValueError, TypeError):
            continue

    log.warning("Could not parse date: %r (year=%d)", raw, year)
    return None


# ---------------------------------------------------------------------------
# Odds parsing helpers
# ---------------------------------------------------------------------------
def parse_american_odds(raw: str) -> Optional[int]:
    """Parse American odds string like '+120', '-110', 'EV' to int."""
    if not raw:
        return None
    raw = raw.strip().replace(",", "").replace(" ", "")
    if raw in ("", "-", "N/A", "n/a", "?"):
        return None
    if raw.upper() in ("EV", "EVEN"):
        return 100
    try:
        return int(raw)
    except ValueError:
        return None


def parse_score(raw: str) -> tuple[Optional[int], Optional[int]]:
    """Parse OddsPortal score '2:5' → (home_score=2, away_score=5)."""
    if not raw or ":" not in raw:
        return None, None
    parts = raw.split(":")
    if len(parts) != 2:
        return None, None
    try:
        return int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        return None, None


def parse_total(raw: str) -> Optional[float]:
    """Parse total like 'o8.5', 'u8.5', '8.5' to float."""
    if not raw:
        return None
    raw = raw.strip().lstrip("ouOU")
    try:
        return float(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Core scraper
# ---------------------------------------------------------------------------
async def dismiss_cookie_banner(page: Page) -> None:
    """Attempt to click common cookie accept buttons."""
    selectors = [
        "button#onetrust-accept-btn-handler",
        "button.accept-cookies",
        "[data-testid='cookie-accept']",
        "button:has-text('Accept')",
        "button:has-text('Accept All')",
        "button:has-text('I Accept')",
        "button:has-text('Agree')",
    ]
    for sel in selectors:
        try:
            btn = page.locator(sel).first
            if await btn.is_visible(timeout=2000):
                await btn.click()
                log.info("Dismissed cookie banner (%s)", sel)
                await page.wait_for_timeout(800)
                return
        except Exception:
            continue


async def scrape_year(context: BrowserContext, year: int) -> list[dict]:
    """Scrape all OddsPortal result pages for a single MLB season."""
    base_url = f"https://www.oddsportal.com/baseball/usa/mlb-{year}/results/"
    games: list[dict] = []
    page_num = 1
    checkpoint_path = OUTPUT_DIR / f"odds_historical_{year}_checkpoint.parquet"

    log.info("=== Starting year %d ===", year)

    # Load + deduplicate existing checkpoint so we don't re-scrape pages
    seen_keys: set[tuple] = set()
    if checkpoint_path.exists():
        try:
            ckpt_df = pd.read_parquet(checkpoint_path)
            before = len(ckpt_df)
            ckpt_df = ckpt_df.drop_duplicates(subset=["game_date", "home_team", "away_team"])
            after = len(ckpt_df)
            log.info("Checkpoint loaded: %d rows → %d unique after dedup (removed %d dupes)",
                     before, after, before - after)
            # Save deduped checkpoint back
            ckpt_df.to_parquet(checkpoint_path, engine="pyarrow", index=False)
            games = ckpt_df.to_dict("records")
            seen_keys = {
                (str(r.get("game_date", ""))[:10], r.get("home_team", ""), r.get("away_team", ""))
                for r in games
            }
            log.info("Resuming with %d unique games already in checkpoint; "
                     "duplicate pages will be skipped automatically", len(games))
        except Exception as exc:
            log.warning("Could not load checkpoint (%s) — starting fresh", exc)

    consecutive_duplicate_pages = 0

    page = await context.new_page()
    await page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})

    try:
        while True:
            # Hard cap — MLB season is ~135 pages, 200 is generous ceiling
            if page_num > MAX_PAGES_PER_YEAR:
                log.info("Year %d | Reached MAX_PAGES_PER_YEAR (%d) — stopping",
                         year, MAX_PAGES_PER_YEAR)
                break

            if page_num == 1:
                url = base_url
                log.info("Year %d | Page %d | URL: %s", year, page_num, url)
                try:
                    await page.goto(url, timeout=NAVIGATION_TIMEOUT, wait_until="domcontentloaded")
                except PlaywrightTimeoutError:
                    log.warning("Timeout on page 1 — retrying")
                    await asyncio.sleep(3)
                    try:
                        await page.goto(url, timeout=NAVIGATION_TIMEOUT, wait_until="domcontentloaded")
                    except PlaywrightTimeoutError:
                        log.error("Second timeout on page 1 — stopping")
                        break
            else:
                # Try clicking the pagination link first (fast path for pages 2-9
                # where data-number buttons are visible in the pagination bar).
                # For pages 10+ OddsPortal only shows 1-9 + "..." so the link
                # won't exist; we fall back to direct URL navigation which always works.
                log.info("Year %d | Page %d | navigating", year, page_num)
                clicked = await page.evaluate(f"""
                    () => {{
                        const link = document.querySelector(
                            'a.pagination-link[data-number="{page_num}"]'
                        );
                        if (link) {{ link.click(); return true; }}
                        return false;
                    }}
                """)
                if not clicked:
                    # Direct URL navigation — works for any page number
                    url = f"{base_url}#/page/{page_num}/"
                    log.info("Year %d | Page %d | no pagination button found, "
                             "navigating directly to %s", year, page_num, url)
                    try:
                        await page.goto(url, timeout=NAVIGATION_TIMEOUT,
                                        wait_until="domcontentloaded")
                    except PlaywrightTimeoutError:
                        log.error("Timeout navigating to page %d — stopping", page_num)
                        break
                # Give Vue time to fetch data and re-render game rows
                await page.wait_for_timeout(3000)

            # Dismiss cookie banner on first page only
            if page_num == 1:
                # Give Cloudflare challenge time to complete (5 s) before interacting
                await page.wait_for_timeout(5000)
                await dismiss_cookie_banner(page)
                await page.wait_for_timeout(1500)

            # Wait for results table — use a longer timeout for URL-navigated pages
            # (direct navigation may trigger Cloudflare JS challenge or slow Vue hydration)
            try:
                await page.wait_for_selector(
                    "div.eventRow, .result-ok, [class*='eventRow']",
                    timeout=30_000
                )
            except PlaywrightTimeoutError:
                log.info("No results table on page %d for year %d — ending pagination", page_num, year)
                break

            # Extract all game rows
            rows_scraped = await extract_game_rows(page, year)
            if not rows_scraped:
                log.info("No rows extracted on page %d — stopping", page_num)
                break

            # Duplicate-page detection: build key set for this page's games
            page_keys = {
                (r.get("game_date", ""), r.get("home_team", ""), r.get("away_team", ""))
                for r in rows_scraped
            }
            new_keys = page_keys - seen_keys

            if not new_keys:
                # Every game on this page was already seen — OddsPortal is looping
                consecutive_duplicate_pages += 1
                log.info("Year %d | Page %d | All %d games already seen (duplicate page %d)",
                         year, page_num, len(rows_scraped), consecutive_duplicate_pages)
                if consecutive_duplicate_pages >= 2:
                    log.info("Year %d | 2 consecutive duplicate pages — stopping pagination", year)
                    break
            else:
                consecutive_duplicate_pages = 0
                seen_keys.update(new_keys)
                # Only add genuinely new games
                new_rows = [r for r in rows_scraped
                            if (r.get("game_date", ""), r.get("home_team", ""), r.get("away_team", ""))
                            in new_keys]
                games.extend(new_rows)
                log.info("Year %d | Page %d | %d new games (%d total unique)",
                         year, page_num, len(new_rows), len(games))

            # Checkpoint every CHECKPOINT_INTERVAL unique games
            if len(games) > 0 and len(games) % CHECKPOINT_INTERVAL < len(rows_scraped):
                _save_checkpoint(games, checkpoint_path, year)

            # Check for next page using only reliable selectors (NOT page-number text match)
            has_next = await check_next_page(page, page_num)
            if not has_next:
                log.info("Year %d | No more pages after page %d", year, page_num)
                break

            page_num += 1
            await asyncio.sleep(PAGE_SLEEP_SECONDS)

    finally:
        await page.close()

    return games


async def extract_game_rows(page: Page, year: int) -> list[dict]:
    """
    Parse game rows from an OddsPortal results page.

    OddsPortal (2024+ layout) uses Vue.js-rendered eventRow divs.
    Key fix: dates appear ONLY in the first eventRow of each date group (as a
    section header). Subsequent game rows on the same date have no date text.
    We carry `currentDate` across rows so games are not silently dropped.
    """
    pull_ts = datetime.utcnow()

    raw_data = await page.evaluate("""
    () => {
        const results = [];
        const eventRows = document.querySelectorAll('[class*="eventRow"]');

        // Carry the last seen date across rows — OddsPortal only emits the date
        // in the first row of each date group; all subsequent rows on the same
        // date have no date text and would be silently skipped without this.
        let currentDate = '';

        eventRows.forEach(row => {
            try {
                const fullText = row.innerText;

                // Update currentDate whenever a "DD Mon YYYY" header appears
                const dateMatch = fullText.match(/(\\d{1,2}\\s+[A-Za-z]{3}\\s+\\d{4})/);
                if (dateMatch) {
                    currentDate = dateMatch[1];
                }

                // Team names ONLY from <p class*="participant"> — never from anchors
                const participantEls = row.querySelectorAll('p[class*="participant"]');
                if (participantEls.length < 2) return;  // date-header row — skip

                const homeTeam = participantEls[0].innerText.trim();
                const awayTeam = participantEls[1].innerText.trim();
                if (!homeTeam || !awayTeam) return;

                // Score: digits around en-dash or hyphen — "6 – 7" or "6-7"
                const scoreMatch = fullText.match(/(\\d+)\\s*[\\u2013\\-]\\s*(\\d+)/);
                const homeScore = scoreMatch ? parseInt(scoreMatch[1]) : null;
                const awayScore = scoreMatch ? parseInt(scoreMatch[2]) : null;

                // ML odds: ±NNN or ±NNNN — exclude 4-digit years (2023-2026)
                const oddsMatches = [];
                const oddsRe = /([+\\-]\\d{2,4})/g;
                let m;
                while ((m = oddsRe.exec(fullText)) !== null) {
                    const n = parseInt(m[1]);
                    if (Math.abs(n) >= 100 && Math.abs(n) <= 5000 &&
                        !(n >= 2020 && n <= 2030)) {
                        oddsMatches.push(m[1]);
                    }
                }

                results.push({
                    homeTeam, awayTeam,
                    dateStr: currentDate,   // use carried date
                    homeScore, awayScore,
                    odds: oddsMatches
                });
            } catch(e) {}
        });
        return results;
    }
    """)

    games = []
    seen: set = set()

    for item in raw_data:
        home_raw = item.get("homeTeam", "")
        away_raw = item.get("awayTeam", "")
        date_str = item.get("dateStr", "")

        home_team = normalize_team(home_raw)
        away_team = normalize_team(away_raw)

        if not home_team or not away_team:
            continue

        game_date = normalize_date(date_str, year) if date_str else None
        if not game_date:
            continue

        key = (game_date, home_team, away_team)
        if key in seen:
            continue
        seen.add(key)

        odds = item.get("odds", [])
        # OddsPortal listing shows: home ML, away ML (open/close not separated)
        # Take first two valid odds as close_ml_home / close_ml_away
        ml_home = parse_american_odds(odds[0]) if len(odds) > 0 else None
        ml_away = parse_american_odds(odds[1]) if len(odds) > 1 else None

        games.append({
            "game_date":       game_date,
            "game_pk":         None,
            "home_team":       home_team,
            "away_team":       away_team,
            "open_ml_home":    ml_home,
            "close_ml_home":   ml_home,
            "open_ml_away":    ml_away,
            "close_ml_away":   ml_away,
            "open_total":      None,   # not on listing page
            "close_total":     None,   # not on listing page
            "runline_home":    -1.5,
            "runline_home_odds": None,
            "public_pct_home": None,
            "public_pct_over": None,
            "source":          "oddsportal",
            "pull_timestamp":  pull_ts,
            "_home_score":     item.get("homeScore"),
            "_away_score":     item.get("awayScore"),
        })

    return games


def _build_game_dict_structured(item: dict, year: int, pull_ts: datetime) -> Optional[dict]:
    """Build a standardised odds dict from structured OddsPortal row data."""
    home_raw = item.get("home", "")
    away_raw = item.get("away", "")
    if not home_raw or not away_raw:
        return None

    game_date = normalize_date(item.get("date", ""), year)
    home_team = normalize_team(home_raw)
    away_team = normalize_team(away_raw)

    score_raw = item.get("score", "")
    # Scores occasionally appear as "W 5:2" or just "5:2"
    score_match = re.search(r"(\d+)\s*:\s*(\d+)", score_raw)
    home_score = int(score_match.group(1)) if score_match else None
    away_score = int(score_match.group(2)) if score_match else None

    odds_list = item.get("odds", [])
    # OddsPortal main page typically shows: [open_home, close_home, open_away, close_away]
    # Totals are usually not on the main page
    open_ml_home = parse_american_odds(odds_list[0]) if len(odds_list) > 0 else None
    close_ml_home = parse_american_odds(odds_list[1]) if len(odds_list) > 1 else None
    open_ml_away = parse_american_odds(odds_list[2]) if len(odds_list) > 2 else None
    close_ml_away = parse_american_odds(odds_list[3]) if len(odds_list) > 3 else None

    # Attempt to extract total from allCells if present
    open_total = None
    close_total = None
    for cell in item.get("allCells", []):
        t = parse_total(cell)
        if t and 4.0 <= t <= 20.0:
            if open_total is None:
                open_total = t
            elif close_total is None:
                close_total = t
                break

    return {
        "game_date": game_date,
        "game_pk": None,
        "home_team": home_team,
        "away_team": away_team,
        "open_ml_home": open_ml_home,
        "close_ml_home": close_ml_home,
        "open_ml_away": open_ml_away,
        "close_ml_away": close_ml_away,
        "open_total": open_total,
        "close_total": close_total,
        "runline_home": -1.5,
        "runline_home_odds": None,
        "public_pct_home": None,
        "public_pct_over": None,
        "source": "oddsportal",
        "pull_timestamp": pull_ts,
        # Extra for debugging
        "_home_score": home_score,
        "_away_score": away_score,
    }


def _build_game_dict_raw(item: dict, year: int, pull_ts: datetime, fallback_date: Optional[str]) -> Optional[dict]:
    """Build a game dict from the raw (unstructured) JS extraction."""
    teams = item.get("teams", [])
    if len(teams) < 2:
        return None

    raw_date = item.get("date", "") or ""
    game_date = normalize_date(raw_date, year) if raw_date else fallback_date

    home_raw, away_raw = teams[0], teams[1]
    home_team = normalize_team(home_raw)
    away_team = normalize_team(away_raw)

    odds_list = item.get("odds", [])
    open_ml_home = parse_american_odds(odds_list[0]) if len(odds_list) > 0 else None
    close_ml_home = parse_american_odds(odds_list[1]) if len(odds_list) > 1 else None
    open_ml_away = parse_american_odds(odds_list[2]) if len(odds_list) > 2 else None
    close_ml_away = parse_american_odds(odds_list[3]) if len(odds_list) > 3 else None

    score_raw = item.get("score", "")
    score_match = re.search(r"(\d+)\s*:\s*(\d+)", score_raw)
    home_score = int(score_match.group(1)) if score_match else None
    away_score = int(score_match.group(2)) if score_match else None

    open_total = None
    close_total = None

    return {
        "game_date": game_date,
        "game_pk": None,
        "home_team": home_team,
        "away_team": away_team,
        "open_ml_home": open_ml_home,
        "close_ml_home": close_ml_home,
        "open_ml_away": open_ml_away,
        "close_ml_away": close_ml_away,
        "open_total": open_total,
        "close_total": close_total,
        "runline_home": -1.5,
        "runline_home_odds": None,
        "public_pct_home": None,
        "public_pct_over": None,
        "source": "oddsportal",
        "pull_timestamp": pull_ts,
        "_home_score": home_score,
        "_away_score": away_score,
    }


async def check_next_page(page: Page, current_page: int) -> bool:
    """
    Return True if OddsPortal has a page beyond current_page.

    OddsPortal (2024+ Vue.js layout) uses:
      <a class="pagination-link" data-number="N">N</a>
    The pagination bar only shows ~9 page buttons at a time, then "..." and
    the final page number. We check BOTH the data-number buttons AND any
    standalone page-count text (e.g. "Page 5 of 137") to find the true max.
    """
    try:
        max_page = await page.evaluate("""
            () => {
                // Primary: data-number links (cover visible pages 1-9 in bar)
                const links = document.querySelectorAll('a.pagination-link[data-number]');
                const nums = Array.from(links)
                    .map(a => parseInt(a.getAttribute('data-number')))
                    .filter(n => !isNaN(n) && n > 0);
                let maxFromLinks = nums.length ? Math.max(...nums) : 0;

                // Secondary: look for any text like "of NNN" or last numeric
                // button after the ellipsis (may not have data-number attr)
                const allPageBtns = document.querySelectorAll(
                    'a.pagination-link, button.pagination-link, [class*="pagination"] a, [class*="pagination"] button'
                );
                const allNums = Array.from(allPageBtns)
                    .map(el => parseInt(el.innerText.trim()))
                    .filter(n => !isNaN(n) && n > 0 && n < 500);
                const maxFromText = allNums.length ? Math.max(...allNums) : 0;

                return Math.max(maxFromLinks, maxFromText) || null;
            }
        """)
        if max_page and isinstance(max_page, (int, float)) and int(max_page) > current_page:
            return True
        # If we got exactly 0 or None from JS (Cloudflare challenge page, etc.),
        # assume there's no next page rather than looping forever.
        return False
    except Exception:
        return False


def _save_checkpoint(games: list[dict], path: Path, year: int) -> None:
    """Save a checkpoint parquet of games scraped so far."""
    try:
        df = _games_to_dataframe(games)
        df.to_parquet(path, engine="pyarrow", index=False)
        log.info("Checkpoint saved: %s (%d rows)", path.name, len(df))
    except Exception as exc:
        log.warning("Failed to save checkpoint: %s", exc)


def _games_to_dataframe(games: list[dict]) -> pd.DataFrame:
    """Convert list of game dicts to standardised DataFrame."""
    SCHEMA_COLS = [
        "game_date", "game_pk", "home_team", "away_team",
        "open_ml_home", "close_ml_home", "open_ml_away", "close_ml_away",
        "open_total", "close_total", "runline_home", "runline_home_odds",
        "public_pct_home", "public_pct_over", "source", "pull_timestamp",
    ]
    df = pd.DataFrame(games)
    # Drop internal debug columns
    debug_cols = [c for c in df.columns if c.startswith("_")]
    df = df.drop(columns=debug_cols, errors="ignore")
    # Ensure all schema columns exist
    for col in SCHEMA_COLS:
        if col not in df.columns:
            df[col] = None
    df = df[SCHEMA_COLS]

    # Type coercions
    df["game_date"] = df["game_date"].astype(str)
    for int_col in ["game_pk", "open_ml_home", "close_ml_home", "open_ml_away", "close_ml_away",
                    "runline_home_odds"]:
        df[int_col] = pd.to_numeric(df[int_col], errors="coerce").astype("Int64")
    for float_col in ["open_total", "close_total", "runline_home", "public_pct_home", "public_pct_over"]:
        df[float_col] = pd.to_numeric(df[float_col], errors="coerce")
    df["pull_timestamp"] = pd.to_datetime(df["pull_timestamp"])

    return df


# ---------------------------------------------------------------------------
# Main async entrypoint
# ---------------------------------------------------------------------------
async def scrape_all_years() -> None:
    """Scrape OddsPortal historical MLB odds for all configured years."""
    async with async_playwright() as pw:
        # Use Firefox — its TLS fingerprint is not flagged by Cloudflare/OddsPortal
        # unlike headless Chromium which triggers ERR_CONNECTION_RESET (net_error -101).
        # headless=False further reduces bot detection (no headless UA hints, real GPU).
        browser = await pw.firefox.launch(
            headless=False,
        )
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) "
                "Gecko/20100101 Firefox/124.0"
            ),
            viewport={"width": 1280, "height": 900},
            locale="en-US",
        )

        for year in YEARS:
            output_path = OUTPUT_DIR / f"odds_historical_{year}.parquet"
            if output_path.exists():
                log.info("Skipping year %d — %s already exists", year, output_path.name)
                continue

            try:
                games = await scrape_year(context, year)
            except Exception as exc:
                log.error("Error scraping year %d: %s", year, exc, exc_info=True)
                continue

            if not games:
                log.warning("No games scraped for year %d", year)
                continue

            df = _games_to_dataframe(games)
            # Deduplicate
            df = df.drop_duplicates(subset=["game_date", "home_team", "away_team"], keep="first")
            df = df.sort_values(["game_date", "home_team"]).reset_index(drop=True)

            df.to_parquet(output_path, engine="pyarrow", index=False)
            log.info(
                "Saved %s | shape: %s | date range: %s → %s",
                output_path.name,
                df.shape,
                df["game_date"].min(),
                df["game_date"].max(),
            )
            print(f"[{year}] Saved {output_path.name} — shape: {df.shape}")

            # Remove checkpoint after successful full save
            checkpoint_path = OUTPUT_DIR / f"odds_historical_{year}_checkpoint.parquet"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                log.info("Removed checkpoint: %s", checkpoint_path.name)

            # Polite delay between years
            await asyncio.sleep(5)

        await context.close()
        await browser.close()


def main() -> None:
    """Synchronous entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Scrape OddsPortal historical MLB odds")
    parser.add_argument("--year", type=int, nargs="+", metavar="YYYY",
                        help="Specific year(s) to scrape (default: all in YEARS list)")
    args = parser.parse_args()

    if args.year:
        global YEARS
        YEARS = args.year

    asyncio.run(scrape_all_years())


if __name__ == "__main__":
    main()
