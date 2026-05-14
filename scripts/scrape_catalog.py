"""
Scrape SHL product catalog — Individual Test Solutions only, all pagination pages.

Uses Selenium (headless Chrome) because the catalog table is paginated client-side.
Pagination uses SHL's custom controls (e.g. `.pagination__item.-arrow.-next` / `.pagination__arrow`),
not DataTables. Pre-packaged Job Solutions pagination is skipped by DOM order (controls must
follow the Individual table in document order).

Usage:
  python scripts/scrape_catalog.py --out app/data/catalog.json

Requires:
  - Google Chrome installed (Selenium 4.6+ uses Selenium Manager for chromedriver)
  - Network access to https://www.shl.com

Optional:
  --no-details       Skip per-product detail pages (list columns only).
  --no-headless      Show browser window for debugging.
  --checkpoint PATH  Save merged catalog after each page (crash recovery).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import (
    ElementClickInterceptedException,
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

# Primary URL from SHL (also accepts legacy /solutions/products/... redirect).
CATALOG_URL = "https://www.shl.com/products/product-catalog/"
BASE = "https://www.shl.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

logger = logging.getLogger("shl_scraper")


@dataclass
class ListRow:
    name: str
    url: str
    remote_cell: str
    adaptive_cell: str
    test_type_cell: str


def _abs_url(href: str) -> str:
    return href if href.startswith("http") else urljoin(BASE, href)


def _tokenize_letters(cell: str) -> list[str]:
    return [ch for ch in cell if ch.isalpha() and ch.isupper()]


def _labels_from_codes(codes: list[str]) -> list[str]:
    labels: list[str] = []
    for c in codes:
        lab = TEST_TYPE_MAP.get(c)
        if lab and lab not in labels:
            labels.append(lab)
    return labels


def _find_individual_table(soup: BeautifulSoup) -> Any:
    """Return the HTML table whose header row identifies Individual Test Solutions (not Job Solutions)."""
    for table in soup.find_all("table"):
        first_row = table.find("tr")
        if not first_row:
            continue
        header_text = first_row.get_text(" ", strip=True)
        if "Individual Test Solutions" in header_text and "Remote Testing" in header_text:
            if "Pre-packaged" in header_text:
                continue
            return table
    return None


def _parse_list_rows(table: Any) -> list[ListRow]:
    rows: list[ListRow] = []
    for tr in table.find_all("tr")[1:]:
        tds = tr.find_all("td")
        if len(tds) < 4:
            continue
        a = tds[0].find("a", href=True)
        if not a:
            continue
        href = str(a.get("href"))
        if "/products/product-catalog/view/" not in href:
            continue
        name = a.get_text(" ", strip=True)
        url = _abs_url(href)
        remote = tds[1].get_text(" ", strip=True)
        adaptive = tds[2].get_text(" ", strip=True)
        test_type = tds[3].get_text(" ", strip=True)
        rows.append(
            ListRow(
                name=name,
                url=url,
                remote_cell=remote,
                adaptive_cell=adaptive,
                test_type_cell=test_type,
            )
        )
    return rows


def _extract_section(text: str, heading: str) -> str:
    pattern = rf"{re.escape(heading)}\s*(.+?)(?:\n[A-Z][^\n]*\n|$)"
    m = re.search(pattern, text, flags=re.S)
    if not m:
        return ""
    block = m.group(1).strip()
    block = re.sub(r"\s+", " ", block)
    return block.strip()


def _parse_detail_page(html: str) -> dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.find("h1")
    title = title_el.get_text(" ", strip=True) if title_el else ""

    text = soup.get_text("\n", strip=True)
    description = _extract_section(text, "Description")
    job_levels = _extract_section(text, "Job levels")
    languages = _extract_section(text, "Languages")
    length_m = re.search(r"Approximate Completion Time in minutes\s*=\s*(\d+)", text, re.I)
    duration = int(length_m.group(1)) if length_m else None
    tt = re.search(r"Test Type:\s*([A-Z])", text)
    test_code = tt.group(1).upper() if tt else ""
    remote_block = _extract_section(text, "Remote Testing")

    remote_supported: bool | None = None
    rl = remote_block.lower()
    if any(x in rl for x in ["yes", "supported", "available", "y"]):
        remote_supported = True
    elif any(x in rl for x in ["no", "not supported", "unavailable", "n"]):
        remote_supported = False

    skills: list[str] = []
    if description:
        skills.extend(
            [s.strip() for s in re.split(r"[,;]", description) if 5 <= len(s.strip()) <= 80][:8]
        )

    job_roles = [s.strip() for s in re.split(r",|\n", job_levels) if s.strip()]
    langs = [s.strip() for s in re.split(r",|\n", languages) if s.strip()]

    return {
        "title": title,
        "description": description,
        "job_roles": job_roles,
        "languages": langs,
        "duration_minutes": duration,
        "detail_test_code": test_code,
        "remote_testing_detail": remote_block,
        "remote_testing_supported": remote_supported,
        "skills": skills,
    }


def _http_get_with_retries(
    url: str,
    *,
    timeout: float = 45.0,
    max_attempts: int = 4,
    session: requests.Session | None = None,
) -> requests.Response:
    """GET with exponential backoff on transient failures."""
    sess = session or requests.Session()
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            r = sess.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            wait = min(8.0, 0.8 * (2 ** (attempt - 1))) + random.uniform(0, 0.4)
            logger.warning("HTTP GET failed (attempt %s/%s) for %s: %s — retry in %.1fs", attempt, max_attempts, url, exc, wait)
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def _build_chrome_driver(*, headless: bool, page_load_timeout: int) -> webdriver.Chrome:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument(f"--user-agent={HEADERS['User-Agent']}")
    opts.add_argument("--lang=en-US")
    # Reduce noisy automation flags for some CDNs
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])

    driver = webdriver.Chrome(service=ChromeService(), options=opts)
    driver.set_page_load_timeout(page_load_timeout)
    driver.implicitly_wait(0)
    return driver


def _dismiss_common_overlays(driver: webdriver.Chrome) -> None:
    """Best-effort: cookie wall / outdated browser banners that block clicks."""
    candidates = [
        (By.XPATH, "//button[contains(., 'I understand') or contains(., 'Continue') or contains(., 'Accept')]"),
        (By.CSS_SELECTOR, "button#onetrust-accept-btn-handler"),
        (By.XPATH, "//a[contains(., 'I understand and wish to continue')]"),
    ]
    for by, sel in candidates:
        try:
            els = driver.find_elements(by, sel)
            for el in els[:2]:
                if el.is_displayed():
                    driver.execute_script("arguments[0].click();", el)
                    time.sleep(0.3)
        except (WebDriverException, StaleElementReferenceException):
            continue


def _wait_for_individual_table(driver: webdriver.Chrome, timeout: int = 45) -> None:
    WebDriverWait(driver, timeout).until(
        lambda d: _find_individual_table(BeautifulSoup(d.page_source, "lxml")) is not None
    )


def _locate_individual_table(driver: webdriver.Chrome) -> Any | None:
    """Find the live Individual Test Solutions <table> (not Pre-packaged Job Solutions)."""
    for t in driver.find_elements(By.TAG_NAME, "table"):
        try:
            row1 = t.find_element(By.XPATH, ".//tr[1]")
            txt = row1.text or ""
        except NoSuchElementException:
            continue
        if (
            "Individual Test Solutions" in txt
            and "Remote Testing" in txt
            and "Pre-packaged" not in txt
        ):
            return t
    return None


# SHL catalog: custom pagination. Observed DOM:
#   .pagination__item.-arrow.-next
#   child click target: .pagination__arrow
_SHL_NEXT_ITEM_SELECTOR = ".pagination__item.-arrow.-next"
_SHL_NEXT_ARROW_SELECTOR = ".pagination__arrow"
_SHL_NEXT_ITEM_FALLBACK = "[class*='pagination__item'][class*='-arrow'][class*='-next']"


def _individual_table_y_min_for_pagination(table_el: Any) -> float:
    """Fallback: only consider Next controls below this Y (when DOM-order scan finds nothing)."""
    loc = table_el.location
    size = table_el.size
    return float(loc["y"]) + float(size["height"]) - 20.0


def _elements_following_table(driver: webdriver.Chrome, table_el: Any, css_selector: str, fallback_selector: str) -> list[Any]:
    """
    Return nodes matching css_selector that appear AFTER the Individual table in document order.
    Excludes Pre-packaged Job Solutions pagination, which precedes the Individual table in the DOM.
    """
    try:
        found = driver.execute_script(
            """
            var table = arguments[0];
            var sel = arguments[1];
            var fb = arguments[2];
            function list(s) {
                return Array.prototype.slice.call(document.querySelectorAll(s));
            }
            var nodes = list(sel);
            if (!nodes.length) nodes = list(fb);
            return nodes.filter(function(n) {
                var pos = table.compareDocumentPosition(n);
                return !!(pos & Node.DOCUMENT_POSITION_FOLLOWING);
            });
            """,
            table_el,
            css_selector,
            fallback_selector,
        )
        if not found:
            return []
        if not isinstance(found, list):
            return [found]
        return list(found)
    except WebDriverException as exc:
        logger.warning("[pagination] compareDocumentPosition scan failed: %s", exc)
        return []


def _shl_next_container_is_disabled(container: Any) -> bool:
    """SHL marks end-of-section Next via classes, aria-disabled, or parent modifiers."""
    try:
        bits = [
            container.get_attribute("class") or "",
            container.get_attribute("aria-disabled") or "",
            str(container.get_attribute("disabled") or ""),
        ]
        joined = " ".join(bits).lower()
        if any(
            token in joined
            for token in (
                "-disabled",
                "disabled",
                "is-disabled",
                "_disabled",
                "pagination__item--disabled",
                "pagination__item_disabled",
            )
        ):
            return True
        if container.get_attribute("aria-disabled") == "true":
            return True
        for _ in range(6):
            try:
                container = container.find_element(By.XPATH, "..")
            except Exception:
                break
            pcls = (container.get_attribute("class") or "").lower()
            if any(x in pcls for x in ("-disabled", "disabled", "is-disabled", "unavailable")):
                return True
    except StaleElementReferenceException:
        return True
    return False


def _read_active_catalog_page_hint(driver: webdriver.Chrome, table_el: Any) -> str:
    """Best-effort current page label from pagination controls that follow the Individual table."""
    hints: list[str] = []
    selectors = (
        ".pagination__item.-active",
        ".pagination__item.is-active",
        ".pagination__item[aria-current='true']",
        ".pagination__item.-current",
        ".pagination a[aria-current='page']",
        ".pagination__item[aria-selected='true']",
    )
    for sel in selectors:
        for el in driver.find_elements(By.CSS_SELECTOR, sel):
            try:
                if not el.is_displayed():
                    continue
                follows = driver.execute_script(
                    """
                    var t = arguments[0], e = arguments[1];
                    return !!(t.compareDocumentPosition(e) & Node.DOCUMENT_POSITION_FOLLOWING);
                    """,
                    table_el,
                    el,
                )
                if not follows:
                    continue
                t = (el.text or "").strip()
                if t and len(t) < 32:
                    hints.append(t)
            except (StaleElementReferenceException, WebDriverException, ValueError, TypeError):
                continue
    return hints[0] if hints else "?"


def _find_shl_individual_next_clickable(driver: webdriver.Chrome, table_el: Any) -> tuple[Any | None, Any | None]:
    """
    Return (click_target, container) for enabled Individual-section Next.

    Primary strategy: DOM order — only `.pagination__item.-arrow.-next` nodes that are
    DOCUMENT_POSITION_FOLLOWING the Individual table (skips Pre-packaged pagination).

    Fallback: Y-threshold below the table if the primary scan finds nothing.
    """
    try:
        y_min = _individual_table_y_min_for_pagination(table_el)
    except StaleElementReferenceException:
        logger.warning("[pagination] Stale table while reading geometry — cannot locate Next.")
        return None, None

    page_hint = _read_active_catalog_page_hint(driver, table_el)
    logger.info("[pagination] current page hint=%r", page_hint)

    containers = _elements_following_table(driver, table_el, _SHL_NEXT_ITEM_SELECTOR, _SHL_NEXT_ITEM_FALLBACK)
    using_dom_order = len(containers) > 0
    logger.info(
        "[pagination] Next containers after Individual table (document order): count=%s (dom_order=%s)",
        len(containers),
        using_dom_order,
    )

    if not containers:
        containers = driver.find_elements(By.CSS_SELECTOR, _SHL_NEXT_ITEM_SELECTOR)
        if not containers:
            containers = driver.find_elements(By.CSS_SELECTOR, _SHL_NEXT_ITEM_FALLBACK)
        logger.info(
            "[pagination] Fallback: all Next on page count=%s (will filter by y >= %.1f)",
            len(containers),
            y_min,
        )

    ranked: list[tuple[float, Any, Any]] = []
    for container in containers:
        try:
            if not container.is_displayed():
                logger.debug("[pagination] Next container not visible — skip")
                continue
            y = float(container.location["y"])
            ccls = container.get_attribute("class") or ""
            if not using_dom_order and y < y_min:
                logger.debug(
                    "[pagination] (y-fallback) skip Next y=%.1f < y_min classes=%r",
                    y,
                    ccls,
                )
                continue
            dis = _shl_next_container_is_disabled(container)
            logger.info("[pagination] Next container y=%.1f classes=%r disabled=%s", y, ccls, dis)
            if dis:
                continue
            arrows = container.find_elements(By.CSS_SELECTOR, _SHL_NEXT_ARROW_SELECTOR)
            if arrows:
                click_el = arrows[0]
                logger.info(
                    "[pagination] click target .pagination__arrow classes=%r",
                    click_el.get_attribute("class") or "",
                )
            else:
                click_el = container
                logger.warning("[pagination] no .pagination__arrow child; clicking container")
            ranked.append((y, click_el, container))
        except StaleElementReferenceException:
            continue

    if not ranked:
        logger.info("[pagination] Next button: NOT FOUND")
        return None, None

    ranked.sort(key=lambda x: x[0])
    _, click_el, _container = ranked[0]
    logger.info("[pagination] Next button: FOUND (first enabled control in chosen ordering)")
    return click_el, _container


def _parse_individual_rows_from_driver(driver: webdriver.Chrome) -> list[ListRow]:
    soup = BeautifulSoup(driver.page_source, "lxml")
    table = _find_individual_table(soup)
    if table is None:
        return []
    return _parse_list_rows(table)


def _click_next_page(driver: webdriver.Chrome, table_el: Any, settle_s: float) -> bool:
    """
    Click SHL's Individual-section Next (.pagination__item.-arrow.-next > .pagination__arrow).
    Waits until the first data row's product URL changes (explicit wait).
    """
    before_rows = _parse_individual_rows_from_driver(driver)
    if not before_rows:
        return False
    before_first_url = before_rows[0].url
    before_key = (len(before_rows), before_first_url)
    logger.info("[pagination] before click: first_row_url=%r row_count=%s", before_first_url, len(before_rows))

    click_el, _ = _find_shl_individual_next_clickable(driver, table_el)
    if click_el is None:
        logger.info("[pagination] stopping: no enabled Next (last page or controls missing).")
        return False

    logger.info("[pagination] moving to next page (click after scroll-into-view)")
    for attempt in range(1, 5):
        try:
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center', inline: 'nearest'});",
                click_el,
            )
            time.sleep(0.2)
            try:
                WebDriverWait(driver, 8).until(lambda d: click_el.is_displayed())
            except TimeoutException:
                logger.debug("[pagination] click target visibility wait skipped/expired")

            try:
                click_el.click()
            except ElementClickInterceptedException:
                logger.warning("[pagination] native click intercepted — using JS click")
                driver.execute_script("arguments[0].click();", click_el)
            break
        except StaleElementReferenceException:
            logger.warning("[pagination] stale click target on attempt %s — re-resolve", attempt)
            table_el = _locate_individual_table(driver)
            if table_el is None:
                return False
            click_el, _ = _find_shl_individual_next_clickable(driver, table_el)
            if click_el is None:
                return False
            time.sleep(0.45)
        except WebDriverException as exc:
            logger.warning("[pagination] click failed attempt %s: %s", attempt, exc)
            time.sleep(0.5)
    else:
        return False

    def first_row_url_changed(d: webdriver.Chrome) -> bool:
        rows_now = _parse_individual_rows_from_driver(d)
        if not rows_now:
            return False
        key = (len(rows_now), rows_now[0].url)
        return key != before_key

    try:
        WebDriverWait(driver, 35).until(first_row_url_changed)
        after = _parse_individual_rows_from_driver(driver)
        if after:
            logger.info(
                "[pagination] after click: first_row_url=%r row_count=%s (changed from previous)",
                after[0].url,
                len(after),
            )
    except TimeoutException:
        logger.warning(
            "[pagination] timeout waiting for first row URL to change (was %r) — stopping pagination.",
            before_first_url,
        )
        return False

    time.sleep(settle_s)
    return True


def _merge_catalog_item(existing_by_url: dict[str, dict[str, Any]], item: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    """Returns (is_duplicate, item_or_existing)."""
    url = item["url"]
    if url in existing_by_url:
        return True, existing_by_url[url]
    existing_by_url[url] = item
    return False, item


def _write_json_atomic(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def scrape_catalog(
    *,
    catalog_url: str = CATALOG_URL,
    fetch_details: bool = True,
    sleep_s: float = 0.08,
    headless: bool = True,
    max_pages: int = 80,
    page_load_timeout: int = 90,
    checkpoint_path: Path | None = None,
    detail_session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """
    Scrape all pages of Individual Test Solutions via Selenium, then optionally enrich
    each row with requests-based detail HTML (same schema as before).
    """
    sess = detail_session or requests.Session()
    existing_by_url: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    duplicates = 0

    driver = _build_chrome_driver(headless=headless, page_load_timeout=page_load_timeout)
    try:
        logger.info("Loading catalog: %s", catalog_url)
        for attempt in range(1, 4):
            try:
                driver.get(catalog_url)
                break
            except TimeoutException:
                logger.warning("Page load timeout (attempt %s) — retrying", attempt)
                time.sleep(2 * attempt)
        else:
            raise RuntimeError("Failed to load catalog after retries.")

        time.sleep(1.2)
        _dismiss_common_overlays(driver)
        _wait_for_individual_table(driver)

        for page_idx in range(1, max_pages + 1):
            logger.info("Scraping list page %s", page_idx)
            _wait_for_individual_table(driver, timeout=30)

            table_el = _locate_individual_table(driver)
            if table_el is None:
                raise RuntimeError("Could not locate Individual Test Solutions table in the live DOM.")

            logger.info(
                "[pagination] list page_index=%s dom_page_hint=%r",
                page_idx,
                _read_active_catalog_page_hint(driver, table_el),
            )
            rows = _parse_individual_rows_from_driver(driver)
            if not rows:
                raise RuntimeError(
                    "No rows parsed for Individual Test Solutions. "
                    "The site HTML may have changed — inspect the catalog page structure."
                )

            new_on_page = 0
            for row in rows:
                type_codes = _tokenize_letters(row.test_type_cell) or _tokenize_letters(row.remote_cell)
                labels = _labels_from_codes(type_codes)

                item: dict[str, Any] = {
                    "name": row.name,
                    "url": row.url,
                    "description": "",
                    "skills": [],
                    "test_type_codes": type_codes,
                    "test_type_labels": labels,
                    "duration_minutes": None,
                    "job_roles": [],
                    "remote_testing_supported": None,
                    "remote_testing_detail": row.remote_cell,
                    "languages": [],
                    "adaptive_irt": row.adaptive_cell,
                    "source_section": "individual_test_solutions",
                }

                is_dup, _ = _merge_catalog_item(existing_by_url, item)
                if is_dup:
                    duplicates += 1
                    logger.debug("Duplicate URL skipped: %s", row.url)
                    continue
                ordered.append(item)
                new_on_page += 1

            logger.info(
                "Scraped %s rows on page %s (%s new, %s duplicates this page)",
                len(rows),
                page_idx,
                new_on_page,
                len(rows) - new_on_page,
            )
            logger.info(
                "Running total unique assessments: %s (duplicates skipped overall: %s)",
                len(ordered),
                duplicates,
            )

            if checkpoint_path is not None:
                _write_json_atomic(checkpoint_path, ordered)
                logger.info("Checkpoint written: %s", checkpoint_path)

            if page_idx > 1 and new_on_page == 0:
                logger.warning(
                    "No new unique rows on page %s — stopping to avoid a pagination loop.",
                    page_idx,
                )
                break

            moved = _click_next_page(driver, table_el, settle_s=0.35)
            if not moved:
                logger.info("Finished pagination after %s page(s).", page_idx)
                break

        logger.info("List scrape complete: %s unique assessments.", len(ordered))

        # --- Detail enrichment (same behavior & schema as legacy scraper) ---
        if fetch_details:
            for i, item in enumerate(ordered, start=1):
                try:
                    r = _http_get_with_retries(item["url"], session=sess, timeout=45.0, max_attempts=4)
                    detail = _parse_detail_page(r.text)
                    if detail["description"]:
                        item["description"] = detail["description"]
                    if detail["skills"]:
                        item["skills"] = detail["skills"]
                    if detail["job_roles"]:
                        item["job_roles"] = detail["job_roles"]
                    if detail["languages"]:
                        item["languages"] = detail["languages"]
                    if detail["duration_minutes"] is not None:
                        item["duration_minutes"] = detail["duration_minutes"]
                    if detail["title"]:
                        item["name"] = detail["title"]
                    if detail["detail_test_code"]:
                        code = detail["detail_test_code"]
                        merged_codes: list[str] = []
                        for c in item["test_type_codes"] + [code]:
                            cu = c.upper()
                            if cu not in merged_codes:
                                merged_codes.append(cu)
                        item["test_type_codes"] = merged_codes
                        item["test_type_labels"] = _labels_from_codes(merged_codes)
                    if detail["remote_testing_supported"] is not None:
                        item["remote_testing_supported"] = detail["remote_testing_supported"]
                    if detail["remote_testing_detail"]:
                        item["remote_testing_detail"] = detail["remote_testing_detail"]
                except Exception as exc:  # noqa: BLE001
                    item["description"] = f"(detail fetch failed: {exc})"
                    logger.error("Detail fetch failed for %s: %s", item["url"], exc)

                if checkpoint_path is not None and i % 25 == 0:
                    _write_json_atomic(checkpoint_path, ordered)

                time.sleep(sleep_s)
                if i % 50 == 0:
                    logger.info("Detail enrichment progress: %s/%s", i, len(ordered))

    finally:
        driver.quit()

    return ordered


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="SHL Individual Test Solutions catalog scraper (all pages).")
    parser.add_argument("--out", type=Path, default=Path("app/data/catalog.json"))
    parser.add_argument("--url", type=str, default=CATALOG_URL, help="Catalog URL (default: SHL product catalog).")
    parser.add_argument("--no-details", action="store_true", help="List pages only (skip product detail HTTP fetches).")
    parser.add_argument("--sleep", type=float, default=0.08, help="Delay between detail page requests.")
    parser.add_argument("--no-headless", action="store_true", help="Show Chrome window.")
    parser.add_argument("--max-pages", type=int, default=80, help="Safety cap on pagination clicks.")
    parser.add_argument("--page-timeout", type=int, default=90, help="Selenium page load timeout (seconds).")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional path: write merged catalog after each list page (crash recovery).",
    )
    args = parser.parse_args()

    ckpt: Path | None = args.checkpoint

    args.out.parent.mkdir(parents=True, exist_ok=True)
    data = scrape_catalog(
        catalog_url=args.url,
        fetch_details=not args.no_details,
        sleep_s=args.sleep,
        headless=not args.no_headless,
        max_pages=args.max_pages,
        page_load_timeout=args.page_timeout,
        checkpoint_path=ckpt,
    )
    _write_json_atomic(args.out, data)
    logger.info("Wrote %s assessments to %s", len(data), args.out)


if __name__ == "__main__":
    main()
