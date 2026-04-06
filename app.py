import json
import inspect
import os
import re
import secrets
from datetime import datetime
from io import BytesIO
from typing import Callable, Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st
from PIL import ImageEnhance, ImageOps

try:
    import pytesseract  # type: ignore
except Exception:  # pragma: no cover
    pytesseract = None

from core_utils import (
    bytes_to_pdfplumber,
    dedupe_transactions,
    normalize_transactions,
    safe_float,
)

from maybank import parse_transactions_maybank
from public_bank import parse_transactions_pbb
from rhb import parse_transactions_rhb
from cimb import parse_transactions_cimb
from bank_islam import parse_bank_islam
from bank_rakyat import parse_bank_rakyat
from hong_leong import parse_hong_leong
from ambank import parse_ambank, extract_ambank_statement_totals
from bank_muamalat import parse_transactions_bank_muamalat
from affin_bank import parse_affin_bank, extract_affin_statement_totals
from agro_bank import parse_agro_bank
from ocbc import parse_transactions_ocbc
from gx_bank import parse_transactions_gx_bank
from mbsb import parse_transactions_mbsb

# ✅ UOB Bank parser
from uob import parse_transactions_uob

# ✅ Alliance Bank parser
from alliance import parse_transactions_alliance

# ✅ PDF password support
from pdf_security import is_pdf_encrypted, decrypt_pdf_bytes
from ui_components import (
    button_compat,
    close_tool_card,
    columns_compat,
    download_button_compat,
    form_submit_button_compat,
    inject_global_styles,
    render_app_hero,
    render_auth_shell,
    render_file_chips,
    render_metric_cards,
    render_parser_intro,
    render_progress_panel,
    render_section_header,
    render_status_card,
    render_steps_showcase,
    render_tool_card_header,
    render_top_bar,
    toggle_compat,
)


def _supports_streamlit_kwarg(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except Exception:
        return False




def require_basic_auth() -> None:
    """Gate the app behind credentials loaded from environment variables."""
    configured_user = os.getenv("BASIC_AUTH_USER")
    configured_pass = os.getenv("BASIC_AUTH_PASS")

    if not configured_user or not configured_pass:
        st.error(
            "Missing BASIC_AUTH_USER or BASIC_AUTH_PASS environment variables. "
            "Set both to use this app."
        )
        st.stop()

    if st.session_state.get("is_authenticated"):
        return

    render_auth_shell()
    auth_feedback = st.empty()

    with st.form("basic_auth_form"):
        entered_user = st.text_input("Username", placeholder="Enter your username")
        entered_pass = st.text_input("Password", type="password", placeholder="Enter your password")
        submitted = form_submit_button_compat("Sign in", primary=True, use_container_width=True)

    st.markdown(
        '<div class="auth-footer-note">Need access? Contact your administrator.</div>',
        unsafe_allow_html=True,
    )

    if submitted:
        is_valid = secrets.compare_digest(entered_user, configured_user) and secrets.compare_digest(
            entered_pass,
            configured_pass,
        )
        if is_valid:
            st.session_state.is_authenticated = True
            st.rerun()
        auth_feedback.error("Invalid username or password.")

    st.stop()


st.set_page_config(page_title="Bank Statement Parser", layout="wide")
if "ui_theme_light" not in st.session_state:
    st.session_state.ui_theme_light = False
st.session_state.ui_theme_mode = "Light" if st.session_state.ui_theme_light else "Dark"

inject_global_styles(st.session_state.ui_theme_mode)
render_top_bar()
require_basic_auth()
render_app_hero()
render_steps_showcase()
render_parser_intro()


# -----------------------------
# Session state init
# -----------------------------
if "status" not in st.session_state:
    st.session_state.status = "idle"

if "results" not in st.session_state:
    st.session_state.results = []

if "affin_statement_totals" not in st.session_state:
    st.session_state.affin_statement_totals = []

if "affin_file_transactions" not in st.session_state:
    st.session_state.affin_file_transactions = {}

if "ambank_statement_totals" not in st.session_state:
    st.session_state.ambank_statement_totals = []

if "ambank_file_transactions" not in st.session_state:
    st.session_state.ambank_file_transactions = {}

if "cimb_statement_totals" not in st.session_state:
    st.session_state.cimb_statement_totals = []

if "cimb_file_transactions" not in st.session_state:
    st.session_state.cimb_file_transactions = {}

if "rhb_statement_totals" not in st.session_state:
    st.session_state.rhb_statement_totals = []

if "rhb_file_transactions" not in st.session_state:
    st.session_state.rhb_file_transactions = {}

if "gx_statement_totals" not in st.session_state:
    st.session_state.gx_statement_totals = []

if "gx_file_transactions" not in st.session_state:
    st.session_state.gx_file_transactions = {}

if "bank_islam_file_month" not in st.session_state:
    st.session_state.bank_islam_file_month = {}

# ✅ password + company name tracking
if "pdf_password" not in st.session_state:
    st.session_state.pdf_password = ""

if "company_name_override" not in st.session_state:
    st.session_state.company_name_override = ""

if "file_company_name" not in st.session_state:
    st.session_state.file_company_name = {}

if "file_account_no" not in st.session_state:
    st.session_state.file_account_no = {}


_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_any_date_for_summary(x) -> pd.Timestamp:
    if x is None:
        return pd.NaT
    s = str(x).strip()
    if not s:
        return pd.NaT
    if _ISO_RE.match(s):
        return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce", dayfirst=True)


def _parse_with_pdfplumber(parser_func: Callable, pdf_bytes: bytes, filename: str) -> List[dict]:
    with bytes_to_pdfplumber(pdf_bytes) as pdf:
        return parser_func(pdf, filename)


# -----------------------------
# Company name extraction (FIXED)
# -----------------------------
# Strong signals
_COMPANY_NAME_PATTERNS = [
    r"(?:ACCOUNT\s+NAME|A\/C\s+NAME|CUSTOMER\s+NAME|NAMA\s+AKAUN|NAMA\s+PELANGGAN|NAMA)\s*[:\-]\s*(.+)",
    r"(?:ACCOUNT\s+HOLDER|PEMEGANG\s+AKAUN)\s*[:\-]\s*(.+)",
]

# Lines we should NOT treat as a company name
_EXCLUDE_LINE_REGEX = re.compile(
    r"(A\/C\s*NO|AC\s*NO|ACCOUNT\s*NO|ACCOUNT\s*NUMBER|NO\.?\s*AKAUN|NO\s+AKAUN|"
    r"STATEMENT\s+DATE|TARIKH\s+PENYATA|DATE\s+FROM|DATE\s+TO|CURRENCY|BRANCH|SWIFT|IBAN|PAGE\s+\d+)",
    re.IGNORECASE,
)

# If a candidate contains a long digit run, it’s usually not a company name.
_LONG_DIGITS_RE = re.compile(r"\d{6,}")
_COMPANY_SUFFIX_RE = re.compile(
    r"\b(SDN\.?\s*BHD\.?|BHD\.?|ENTERPRISE|PERNIAGAAN|AGENCY|RESOURCES|HOLDINGS|TRADING|SERVICES|TECHNOLOGY|VENTURES|INDUSTRIES|GLOBAL|GROUP|CORPORATION|PLT)\b",
    re.IGNORECASE,
)
_COMPANY_BAD_WORDS_RE = re.compile(
    r"\b(STATEMENT|ACCOUNT\s+STATEMENT|CURRENT\s+ACCOUNT|PAGE\b|BALANCE\b|SUMMARY\b|TRANSACTION|ENQUIRIES|BRANCH|PIDM|DATE\b|MUKA\b|HALAMAN\b|結單日期|结单日期)\b",
    re.IGNORECASE,
)


def _clean_candidate_name(s: str) -> str:
    s = (s or "").strip()
    # stop at common trailing fields
    s = re.split(
        r"\s{2,}|ACCOUNT\s+NO|A\/C\s+NO|NO\.\s*AKAUN|NO\s+AKAUN|STATEMENT|PENYATA|DATE|TARIKH|CURRENCY|BRANCH|PAGE|HALAMAN|結單日期|结单日期",
        s,
        flags=re.IGNORECASE,
    )[0].strip()
    # remove weird leading bullets/colons
    s = s.lstrip(":;-• ").strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _looks_like_account_number_line(s: str) -> bool:
    if not s:
        return True
    up = s.upper()
    if _EXCLUDE_LINE_REGEX.search(up):
        return True
    if _LONG_DIGITS_RE.search(s):
        # long digit run strongly suggests account number/reference, not company name
        return True
    # too short is suspicious
    if len(s.strip()) < 3:
        return True
    return False


def _looks_like_company_name(s: str) -> bool:
    if not s:
        return False

    cand = _clean_candidate_name(s)
    if not cand:
        return False
    if _looks_like_account_number_line(cand):
        return False
    if _COMPANY_BAD_WORDS_RE.search(cand):
        return False
    if re.search(r"https?://|www\.", cand, flags=re.IGNORECASE):
        return False
    if len(cand) < 6:
        return False
    if re.match(r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", cand):
        return False
    return bool(_COMPANY_SUFFIX_RE.search(cand))


def extract_company_name(pdf, max_pages: int = 2) -> Optional[str]:
    """
    Extract company/account holder name from statement.
    Strategy:
      1) Search explicit labels (Account Name / Customer Name / Nama...) on first N pages
      2) Fallback: choose first plausible line that is NOT account-number-ish
    """
    texts: List[str] = []
    try:
        for i in range(min(max_pages, len(pdf.pages))):
            texts.append((pdf.pages[i].extract_text() or "").strip())
    except Exception:
        pass

    texts = [t for t in texts if t]
    if not texts:
        return None

    full = "\n".join(texts)

    # 0) GX Bank greeting banner style
    # Example:
    #   "Hey Remy ..., here's a look at ELSANA TRADING & SERVICES's performance in September!"
    # Capture only the company segment between "look at" and "performance".
    m_gx = re.search(
        r"here['’]?s\s+a\s+look\s+at\s+(.+?)(?:['’]s)?\s+performance\b",
        full,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_gx:
        cand = _clean_candidate_name(m_gx.group(1))
        if cand and not _looks_like_account_number_line(cand):
            return cand

    # 0) UOB "Account Activities" export style
    # Example block:
    #   Company / Account Account Balance
    #   Company Available Balance
    #   UPELL CORPORATION SDN. BHD. MYR 55,744.04
    m_uob = re.search(
        r"Company\s*/\s*Account.*?\bCompany\b.*?\n\s*([A-Z0-9 &().,'\/-]{3,})",
        full,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m_uob:
        cand = _clean_candidate_name(m_uob.group(1))
        # strip appended currency/balance if present
        cand = re.split(r"\bMYR\b", cand, maxsplit=1, flags=re.IGNORECASE)[0].strip() or cand
        if cand and not _looks_like_account_number_line(cand):
            return cand

    # 0.5) Maybank bilingual header style (company line with statement-date markers)
    # Examples:
    #   LSR AGENCY 結單日期 : 31/03/25
    #   PERNIAGAAN SEPAKAT ABADI 結單日期 : 31/01/25
    maybank_lines = [ln.strip() for ln in full.splitlines() if ln.strip()]

    # Maybank often places the company around "TARIKH PENYATA", sometimes split:
    #   QUATTRO FRATELLI
    #   TARIKH PENYATA
    #   ENERGY SDN. BHD.
    for i, ln in enumerate(maybank_lines[:80]):
        if not re.search(r"^TARIKH\s+PENYATA$", ln, flags=re.IGNORECASE):
            continue

        prev_ln = _clean_candidate_name(maybank_lines[i - 1]) if i - 1 >= 0 else ""
        next_ln = _clean_candidate_name(maybank_lines[i + 1]) if i + 1 < len(maybank_lines) else ""

        if prev_ln and next_ln and not _looks_like_account_number_line(prev_ln):
            if re.search(r"(MUKA|PAGE|MAYBANK|IBS\s|BRANCH)", prev_ln, flags=re.IGNORECASE):
                prev_ln = ""

        if prev_ln and next_ln:
            merged = _clean_candidate_name(f"{prev_ln} {next_ln}")
            if merged and not _looks_like_account_number_line(merged):
                if _looks_like_company_name(merged) or re.search(
                    r"\b(SDN\.?\s*BHD\.?|PERNIAGAAN|AGENCY)\b",
                    merged,
                    flags=re.IGNORECASE,
                ):
                    return merged

        if next_ln and not _looks_like_account_number_line(next_ln):
            if _looks_like_company_name(next_ln):
                return next_ln

    for i, ln in enumerate(maybank_lines[:80]):
        m_maybank_line = re.match(
            r"^([A-Z][A-Z0-9 &().,\'\/-]{2,}?)\s+(?:結單日期|结单日期|STATEMENT\s+DATE)\s*:?\s*\d{2}/\d{2}/\d{2,4}\s*$",
            ln,
            flags=re.IGNORECASE,
        )
        if not m_maybank_line:
            continue

        cand = _clean_candidate_name(m_maybank_line.group(1))

        # Some Maybank statements split the name over 2 lines, e.g.:
        #   QUATTRO FRATELLI ENERGY
        #   SDN. BHD. 結單日期 : 31/07/2025
        if re.fullmatch(r"SDN\.?\s*BHD\.?", cand, flags=re.IGNORECASE):
            # In some files the line right above is "TARIKH PENYATA", so walk
            # backward to find the nearest plausible company prefix.
            for j in range(i - 1, max(-1, i - 4), -1):
                if j < 0:
                    break
                prefix = _clean_candidate_name(maybank_lines[j])
                if not prefix:
                    continue
                if re.search(r"^(TARIKH\s+PENYATA|STATEMENT\s+DATE|MUKA|PAGE)\b", prefix, flags=re.IGNORECASE):
                    continue
                merged = _clean_candidate_name(f"{prefix} {cand}")
                if merged and not _looks_like_account_number_line(merged):
                    return merged

        if cand and not _looks_like_account_number_line(cand):
            return cand

    # 1) label-based extraction
    for pat in _COMPANY_NAME_PATTERNS:
        m = re.search(pat, full, flags=re.IGNORECASE)
        if m:
            cand = _clean_candidate_name(m.group(1))
            if cand and not _looks_like_account_number_line(cand):
                return cand

    # 2) fallback: scan lines
    lines: List[str] = []
    for t in texts:
        lines.extend([ln.strip() for ln in t.splitlines() if ln.strip()])

    # 2) context-aware: line before account label often contains company name
    for i, ln in enumerate(lines[:80]):
        if re.search(r"A\/C|ACCOUNT\s*NO|ACCOUNT\s*NUMBER|NOMBOR\s+AKAUN|NO\.?\s*AKAUN", ln, flags=re.IGNORECASE):
            if i > 0:
                prev = _clean_candidate_name(lines[i - 1])
                if _looks_like_company_name(prev):
                    return prev

    # 3) suffix-aware scan (most reliable for Malaysian company names)
    for i, ln in enumerate(lines[:80]):
        cand = _clean_candidate_name(ln)
        if _looks_like_company_name(cand):
            return cand

        # handle split names e.g. "CLEAR WATER SERVICES" + "SDN. BHD."
        if i + 1 < len(lines):
            merged = _clean_candidate_name(f"{ln} {lines[i + 1]}")
            if _looks_like_company_name(merged) and len(merged) <= 120:
                return merged

    # 4) conservative fallback: only return if still company-like
    for i, ln in enumerate(lines[:80]):
        cand = _clean_candidate_name(ln)
        if _looks_like_company_name(cand):
            return cand
        if i + 1 < len(lines):
            merged = _clean_candidate_name(f"{ln} {lines[i + 1]}")
            if _looks_like_company_name(merged) and len(merged) <= 120:
                return merged

    return None


# -----------------------------
# Account number extraction (NEW)
# -----------------------------
_ACCOUNT_NO_PATTERNS = [
    r"(?:A\/C\s*NO|AC\s*NO|ACC(?:OUNT)?\s*NO\.?|ACCOUNT\s*NUMBER|NOMBOR\s+AKAUN|NO\.?\s*AKAUN|NO\s+AKAUN)\s*[:\-]?\s*([\d][\d\- ]{4,36}\d)",
    # UOB export: "Account Ledger Balance" then the account number on the next line
    r"Account\s+Ledger\s+Balance\s*\n\s*([\d][\d\- ]{4,36}\d)",
]

_ACCOUNT_LABEL_RE = re.compile(
    r"(A\/C\s*NO|AC\s*NO|ACC(?:OUNT)?\s*NO\.?|ACCOUNT\s*NUMBER|NOMBOR\s+AKAUN|NO\.?\s*AKAUN|NO\s+AKAUN)",
    re.IGNORECASE,
)

_ACCOUNT_NUM_RE = re.compile(r"\b\d(?:[\d\-]{4,28}\d)\b")


def _normalize_account_no(raw: str) -> Optional[str]:
    if not raw:
        return None
    cleaned = re.sub(r"\s+", "", str(raw).strip())
    digits_only = re.sub(r"\D", "", cleaned)
    if 6 <= len(digits_only) <= 16:
        return digits_only
    return None


def _candidate_account_numbers(text: str) -> List[str]:
    if not text:
        return []

    out: List[str] = []
    for m in _ACCOUNT_NUM_RE.finditer(text):
        num = _normalize_account_no(m.group(0) or "")
        if not num:
            continue
        # avoid date-like fragments accidentally captured from labels/windows
        if re.fullmatch(r"\d{8}", num):
            yyyy = int(num[:4])
            mm = int(num[4:6])
            dd = int(num[6:8])
            if 1900 <= yyyy <= 2100 and 1 <= mm <= 12 and 1 <= dd <= 31:
                continue
        out.append(num)
    return out


def extract_account_number(pdf, max_pages: int = 2) -> Optional[str]:
    texts: List[str] = []
    try:
        for i in range(min(max_pages, len(pdf.pages))):
            texts.append((pdf.pages[i].extract_text() or "").strip())
    except Exception:
        pass

    texts = [t for t in texts if t]
    if not texts:
        return None

    full = "\n".join(texts)
    lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
    full_upper = full.upper()

    # Bank-specific hardening: RHB Reflex headers usually print the account number directly
    # after "Reflex Cash Management ...", often on the next line.
    if ("REFLEX CASH MANAGEMENT" in full_upper) and ("DEPOSIT ACCOUNT SUMMARY" in full_upper):
        reflex_candidates: List[str] = []
        for m in re.finditer(r"REFLEX\s+CASH\s+MANAGEMENT[^\n\r]{0,120}[\n\r]+\s*([0-9][0-9\-\s]{9,20})\b", full, re.IGNORECASE):
            num = _normalize_account_no(m.group(1) or "")
            if num and len(num) >= 10:
                reflex_candidates.append(num)
        if reflex_candidates:
            # pick the most repeated, then the longest (stable across pages/months)
            uniq = sorted(set(reflex_candidates), key=lambda x: (-reflex_candidates.count(x), -len(x), x))
            return uniq[0]

    # Bank-specific hardening: RHB deposit-account summary pages often place the account number
    # in compact rows such as "ORDINARYCURRENTACCOUNT21406200114180".
    full_compact = re.sub(r"\s+", "", full_upper)
    if "DEPOSITACCOUNTSUMMARY" in full_compact or "RINGKASANAKAUNDEPOSIT" in full_compact:
        # Prefer summary rows: account number followed by balance columns.
        for ln in lines[:140]:
            m = re.search(
                r"(?:CURRENT\s*ACCOUNT(?:-I)?|ACCOUNT(?:-I)?)\s*([0-9]{10,16})\s+\d{1,3}(?:,\d{3})*\.\d{2}\s+\d{1,3}(?:,\d{3})*\.\d{2}",
                ln,
                re.IGNORECASE,
            )
            if m:
                num = _normalize_account_no(m.group(1) or "")
                if num:
                    return num

        # Fallback for compact rows like "...CURRENTACCOUNT21406200114180".
        for ln in lines[:140]:
            if len(ln) > 60:
                continue
            m = re.search(r"(?:CURRENT\s*ACCOUNT(?:-I)?|ACCOUNT(?:-I)?)\s*([0-9]{10,16})\b", ln, re.IGNORECASE)
            if m:
                num = _normalize_account_no(m.group(1) or "")
                if num:
                    return num

    scored: Dict[str, int] = {}

    def _add(num: Optional[str], points: int) -> None:
        if not num:
            return
        scored[num] = scored.get(num, 0) + points

    # 1) Strong patterns with account labels.
    for pat in _ACCOUNT_NO_PATTERNS:
        m = re.search(pat, full, flags=re.IGNORECASE | re.DOTALL)
        if m:
            num = _normalize_account_no(m.group(1) or "")
            if num:
                _add(num, 120)

    # Bonus for candidates that appear repeatedly in the document.
    for cand in {c for c in _candidate_account_numbers(full)}:
        repeats = len(re.findall(rf"\b{re.escape(cand)}\b", re.sub(r"\D", " ", full)))
        if repeats >= 2:
            _add(cand, repeats * 10)

    # 2) Label-aware scan on individual lines and short windows.
    for i, ln in enumerate(lines[:180]):
        if not _ACCOUNT_LABEL_RE.search(ln):
            continue

        for cand in _candidate_account_numbers(ln):
            _add(cand, 100)

        window = " ".join(lines[i : min(i + 3, len(lines))])
        for cand in _candidate_account_numbers(window):
            _add(cand, 60)

    if scored:
        return sorted(scored.items(), key=lambda kv: (-kv[1], -len(kv[0]), kv[0]))[0][0]

    # 4) Fallback: standalone account-number-like lines.
    for ln in lines[:120]:
        raw = (ln or "").strip()
        if re.fullmatch(r"\d{10,16}", raw):
            return raw

    return None

# -----------------------------
# Bank Islam: statement month for zero-transaction months
# -----------------------------
_BANK_ISLAM_STMT_DATE_RE = re.compile(
    r"(?:STATEMENT\s+DATE|TARIKH\s+PENYATA)\s*:?\s*(\d{1,2})/(\d{1,2})/(\d{2,4})",
    re.IGNORECASE,
)


def extract_bank_islam_statement_month(pdf) -> Optional[str]:
    try:
        t = (pdf.pages[0].extract_text() or "")
    except Exception:
        return None

    m = _BANK_ISLAM_STMT_DATE_RE.search(t)
    if not m:
        return None

    mm = int(m.group(2))
    yy_raw = m.group(3)
    yy = (2000 + int(yy_raw)) if len(yy_raw) == 2 else int(yy_raw)

    if 1 <= mm <= 12 and 2000 <= yy <= 2100:
        return f"{yy:04d}-{mm:02d}"
    return None


# -----------------------------
# CIMB totals extractor (existing)
# -----------------------------
_CIMB_STMT_DATE_RE = re.compile(
    r"(?:STATEMENT\s+DATE|TARIKH\s+PENYATA)\s*:?\s*(\d{1,2})/(\d{1,2})/(\d{2,4})",
    re.IGNORECASE,
)
_CIMB_CLOSING_RE = re.compile(
    r"CLOSING\s+BALANCE\s*/\s*BAKI\s+PENUTUP\s+(-?[\d,]+\.\d{2})",
    re.IGNORECASE,
)
_CIMB_MONTH_MAP = {
    "JAN": "01",
    "FEB": "02",
    "MAR": "03",
    "APR": "04",
    "MAY": "05",
    "JUN": "06",
    "JUL": "07",
    "AUG": "08",
    "SEP": "09",
    "OCT": "10",
    "NOV": "11",
    "DEC": "12",
}
_CIMB_TESSERACT_READY = None


def _prev_month(yyyy: int, mm: int) -> Tuple[int, int]:
    if mm == 1:
        return (yyyy - 1, 12)
    return (yyyy, mm - 1)


def _has_cimb_tesseract_binary() -> bool:
    global _CIMB_TESSERACT_READY
    if pytesseract is None:
        _CIMB_TESSERACT_READY = False
        return False
    if _CIMB_TESSERACT_READY is not None:
        return _CIMB_TESSERACT_READY
    try:
        pytesseract.get_tesseract_version()
        _CIMB_TESSERACT_READY = True
    except Exception:
        _CIMB_TESSERACT_READY = False
    return _CIMB_TESSERACT_READY


def _infer_cimb_statement_month_from_ocr(pdf) -> Optional[str]:
    if not getattr(pdf, "pages", None) or not _has_cimb_tesseract_binary():
        return None
    try:
        img = pdf.pages[0].to_image(resolution=350).original
        img = ImageOps.grayscale(img)
        img = ImageEnhance.Contrast(img).enhance(2.0)
        text = pytesseract.image_to_string(img, config="--psm 6") or ""
    except Exception:
        return None

    m = _CIMB_STMT_DATE_RE.search(text)
    if not m:
        return None
    mm = int(m.group(2))
    yy_raw = m.group(3)
    yy = (2000 + int(yy_raw)) if len(yy_raw) == 2 else int(yy_raw)
    if 1 <= mm <= 12 and 2000 <= yy <= 2100:
        py, pm = _prev_month(yy, mm)
        return f"{py:04d}-{pm:02d}"
    return None


def _infer_cimb_statement_month_from_filename(source_file: str) -> Optional[str]:
    name = (source_file or "").upper().strip()
    if not name:
        return None

    m = re.search(r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*[\s\-_]*(\d{2,4})\b", name)
    if m:
        mon = _CIMB_MONTH_MAP.get(m.group(1))
        yy_raw = m.group(2)
        yy = (2000 + int(yy_raw)) if len(yy_raw) == 2 else int(yy_raw)
        if mon and 2000 <= yy <= 2100:
            return f"{yy:04d}-{mon}"

    m = re.search(r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)(\d{2,4})\b", name)
    if m:
        mon = _CIMB_MONTH_MAP.get(m.group(1))
        yy_raw = m.group(2)
        yy = (2000 + int(yy_raw)) if len(yy_raw) == 2 else int(yy_raw)
        if mon and 2000 <= yy <= 2100:
            return f"{yy:04d}-{mon}"

    m = re.search(r"(20\d{2})[\s\-_](0[1-9]|1[0-2])", name)
    if m:
        return f"{int(m.group(1)):04d}-{m.group(2)}"

    return None


def extract_cimb_statement_totals(pdf, source_file: str) -> dict:
    full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    up = full_text.upper()

    page_opening_balance = None
    try:
        first_text = pdf.pages[0].extract_text() or ""
        mo = re.search(r"Opening\s+Balance\s+(-?[\d,]+\.\d{2})", first_text, re.IGNORECASE)
        if mo:
            page_opening_balance = float(mo.group(1).replace(",", ""))
    except Exception:
        page_opening_balance = None

    stmt_month = None
    m = _CIMB_STMT_DATE_RE.search(full_text)
    if m:
        mm = int(m.group(2))
        yy_raw = m.group(3)
        yy = (2000 + int(yy_raw)) if len(yy_raw) == 2 else int(yy_raw)
        if 1 <= mm <= 12 and 2000 <= yy <= 2100:
            py, pm = _prev_month(yy, mm)
            stmt_month = f"{py:04d}-{pm:02d}"
    if stmt_month is None:
        stmt_month = _infer_cimb_statement_month_from_ocr(pdf)
    if stmt_month is None:
        stmt_month = _infer_cimb_statement_month_from_filename(source_file)

    closing_balance = None
    m = _CIMB_CLOSING_RE.search(full_text)
    if m:
        closing_balance = float(m.group(1).replace(",", ""))

    total_debit = None
    total_credit = None
    if "TOTAL WITHDRAWAL" in up and "TOTAL DEPOSITS" in up:
        idx = up.rfind("TOTAL WITHDRAWAL")
        window = full_text[idx : idx + 900] if idx != -1 else full_text

        mm2 = re.search(r"\b\d{1,6}\s+\d{1,6}\s+(-?[\d,]+\.\d{2})\s+(-?[\d,]+\.\d{2})\b", window)
        if mm2:
            total_debit = float(mm2.group(1).replace(",", ""))
            total_credit = float(mm2.group(2).replace(",", ""))
        else:
            money = re.findall(r"-?[\d,]+\.\d{2}", window)
            if len(money) >= 2:
                total_debit = float(money[-2].replace(",", ""))
                total_credit = float(money[-1].replace(",", ""))

    return {
        "bank": "CIMB Bank",
        "source_file": source_file,
        "statement_month": stmt_month,
        "total_debit": total_debit,
        "total_credit": total_credit,
        "ending_balance": closing_balance,
        "page_opening_balance": page_opening_balance,
        "opening_balance": None,
    }



def extract_rhb_statement_totals(pdf, source_file: str) -> dict:
    full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    full_text_norm = re.sub(r"\s+", " ", full_text).strip()

    def _signed_money(token: str) -> Optional[float]:
        if not token:
            return None
        s = token.strip().replace(",", "")
        sign = 1.0
        if s.endswith("-"):
            sign = -1.0
            s = s[:-1]
        elif s.endswith("+"):
            s = s[:-1]
        try:
            return round(sign * float(s), 2)
        except Exception:
            return None

    period_match = re.search(
        r"Statement\s+Period.*?:\s*\d{1,2}\s+([A-Za-z]{3})\s+(\d{2,4})",
        full_text,
        re.IGNORECASE,
    )
    statement_month = None
    if period_match:
        month_map = {
            "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
            "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
        }
        mon = period_match.group(1).upper()
        yy = period_match.group(2)
        if mon in month_map:
            year = int(yy) if len(yy) == 4 else (2000 + int(yy))
            statement_month = f"{year:04d}-{month_map[mon]}"
    else:
        # Reflex-style: "Statement Period 01 August 2025 To 31 August 2025"
        period_match2 = re.search(
            r"Statement\s+Period\s+\d{1,2}\s+([A-Za-z]{3,9})\s+(\d{4})\s+To\s+\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}",
            full_text_norm,
            re.IGNORECASE,
        )
        if period_match2:
            mon = period_match2.group(1).upper()[:3]
            yy = int(period_match2.group(2))
            month_map = {
                "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
                "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
            }
            if mon in month_map:
                statement_month = f"{yy:04d}-{month_map[mon]}"

    opening_balance = None
    ending_balance = None
    total_debit = None
    total_credit = None

    bfm = re.search(r"\b\d{1,2}\s+[A-Za-z]{3}\s+B/F\s+BALANCE\s+(-?[\d,]+\.\d{2})", full_text, re.IGNORECASE)
    if bfm:
        opening_balance = float(bfm.group(1).replace(",", ""))

    cfm = re.search(r"\b\d{1,2}\s+[A-Za-z]{3}\s+C/F\s+BALANCE\s+(-?[\d,]+\.\d{2})", full_text, re.IGNORECASE)
    if cfm:
        ending_balance = float(cfm.group(1).replace(",", ""))

    tm = re.search(r"\(RM\)\s+(-?[\d,]+\.\d{2})\s+(-?[\d,]+\.\d{2})", full_text, re.IGNORECASE)
    if tm:
        total_debit = float(tm.group(1).replace(",", ""))
        total_credit = float(tm.group(2).replace(",", ""))

    # Reflex summary fallback
    if opening_balance is None:
        m = re.search(
            r"Beginning\s+Balance\s+as\s+of\s+\d{1,2}\s+[A-Za-z]{3,9}(?:\s+\d{2,4})?\s+([\d,]+\.\d{2}[+-]?)",
            full_text_norm,
            re.IGNORECASE,
        )
        opening_balance = _signed_money(m.group(1)) if m else None

    if ending_balance is None:
        m = re.search(
            r"Ending\s+Balance\s+as\s+of\s+\d{1,2}\s+[A-Za-z]{3,9}(?:\s+\d{2,4})?\s+([\d,]+\.\d{2}[+-]?)",
            full_text_norm,
            re.IGNORECASE,
        )
        ending_balance = _signed_money(m.group(1)) if m else None

    if total_credit is None:
        m = re.search(r"\b\d+\s+Deposits\s*\(Plus\)\s+([\d,]+\.\d{2})", full_text_norm, re.IGNORECASE)
        if m:
            total_credit = float(m.group(1).replace(",", ""))

    if total_debit is None:
        m = re.search(r"\b\d+\s+Withdraws\s*\(Minus\)\s+([\d,]+\.\d{2})", full_text_norm, re.IGNORECASE)
        if m:
            total_debit = float(m.group(1).replace(",", ""))

    return {
        "bank": "RHB Bank",
        "source_file": source_file,
        "statement_month": statement_month,
        "total_debit": total_debit,
        "total_credit": total_credit,
        "ending_balance": ending_balance,
        "opening_balance": opening_balance,
    }


def extract_gx_statement_totals(pdf, source_file: str) -> dict:
    full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)
    full_text_norm = re.sub(r"\s+", " ", full_text).strip()

    statement_month = None
    month_match = re.search(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(20\d{2})\b",
        full_text_norm,
        re.IGNORECASE,
    )
    if month_match:
        month_map = {
            "JANUARY": "01", "FEBRUARY": "02", "MARCH": "03", "APRIL": "04",
            "MAY": "05", "JUNE": "06", "JULY": "07", "AUGUST": "08",
            "SEPTEMBER": "09", "OCTOBER": "10", "NOVEMBER": "11", "DECEMBER": "12",
        }
        mon = month_map.get(month_match.group(1).upper())
        year = int(month_match.group(2))
        if mon:
            statement_month = f"{year:04d}-{mon}"

    opening_balance = None
    m_open = re.search(
        r"\b\d{1,2}\s+[A-Za-z]{3}\s+Opening\s+balance\s+(-?[\d,]+\.\d{2})\b",
        full_text,
        re.IGNORECASE,
    )
    if m_open:
        opening_balance = float(m_open.group(1).replace(",", ""))

    total_credit = None
    total_debit = None
    ending_balance = None
    m_total = re.search(
        r"Total:\s*([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})\s+([\d,]+\.\d{2})",
        full_text,
        re.IGNORECASE,
    )
    if m_total:
        money_in = float(m_total.group(1).replace(",", ""))
        money_out = float(m_total.group(2).replace(",", ""))
        interest = float(m_total.group(3).replace(",", ""))
        ending_balance = float(m_total.group(4).replace(",", ""))
        total_debit = money_out
        total_credit = round(money_in + interest, 2)

    return {
        "bank": "GX Bank",
        "source_file": source_file,
        "statement_month": statement_month,
        "total_debit": total_debit,
        "total_credit": total_credit,
        "ending_balance": ending_balance,
        "opening_balance": opening_balance,
    }

# -----------------------------
# Bank parsers
# -----------------------------
PARSERS: Dict[str, Callable[[bytes, str], List[dict]]] = {
    "Affin Bank": lambda b, f: _parse_with_pdfplumber(parse_affin_bank, b, f),
    "Agro Bank": lambda b, f: _parse_with_pdfplumber(parse_agro_bank, b, f),
    "Alliance Bank": lambda b, f: _parse_with_pdfplumber(parse_transactions_alliance, b, f),
    "Ambank": lambda b, f: _parse_with_pdfplumber(parse_ambank, b, f),
    "Bank Islam": lambda b, f: _parse_with_pdfplumber(parse_bank_islam, b, f),
    "Bank Muamalat": lambda b, f: _parse_with_pdfplumber(parse_transactions_bank_muamalat, b, f),
    "Bank Rakyat": lambda b, f: _parse_with_pdfplumber(parse_bank_rakyat, b, f),
    "CIMB Bank": lambda b, f: _parse_with_pdfplumber(parse_transactions_cimb, b, f),
    "Hong Leong": lambda b, f: _parse_with_pdfplumber(parse_hong_leong, b, f),
    "Maybank": lambda b, f: parse_transactions_maybank(b, f),
    "MBSB Bank": lambda b, f: _parse_with_pdfplumber(parse_transactions_mbsb, b, f),
    "Public Bank (PBB)": lambda b, f: _parse_with_pdfplumber(parse_transactions_pbb, b, f),
    "RHB Bank": lambda b, f: parse_transactions_rhb(b, f),
    "OCBC Bank": lambda b, f: parse_transactions_ocbc(b, f),
    "GX Bank": lambda b, f: _parse_with_pdfplumber(parse_transactions_gx_bank, b, f),
    "UOB Bank": lambda b, f: _parse_with_pdfplumber(parse_transactions_uob, b, f),
}


has_existing_results = bool(
    st.session_state.results
    or st.session_state.affin_statement_totals
    or st.session_state.ambank_statement_totals
    or st.session_state.cimb_statement_totals
    or st.session_state.rhb_statement_totals
    or st.session_state.gx_statement_totals
)

workspace_left, workspace_right = st.columns([0.9, 1.45], gap="large")

with workspace_right:

    render_tool_card_header("▣", "Select Bank", "Choose the issuing bank")
    if _supports_streamlit_kwarg(st.selectbox, "label_visibility"):
        bank_choice = st.selectbox("Select Bank Format", list(PARSERS.keys()), label_visibility="collapsed")
    else:
        bank_choice = st.selectbox("Select Bank Format", list(PARSERS.keys()))
    close_tool_card()

    render_tool_card_header("⤴", "Upload Statement", "PDF format, one or multiple files")
    if _supports_streamlit_kwarg(st.file_uploader, "label_visibility"):
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    else:
        uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded_files:
        uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
    close_tool_card()

    # Detect encrypted files
    encrypted_files: List[str] = []
    if uploaded_files:
        for uf in uploaded_files:
            try:
                if is_pdf_encrypted(uf.getvalue()):
                    encrypted_files.append(uf.name)
            except Exception:
                encrypted_files.append(uf.name)

    if uploaded_files:
        render_file_chips(uploaded_files, encrypted_files)

    if encrypted_files:
        render_tool_card_header("🔒", "Encrypted PDFs", "Enter the password once and it will be used for all encrypted files")
        st.warning(
            "Encrypted PDF(s) detected:\n\n" + "\n".join([f"- {n}" for n in encrypted_files])
        )
        st.text_input("PDF Password", type="password", key="pdf_password")
        close_tool_card()

    render_tool_card_header("✎", "Company Override", "Optional manual company name if you want to override extraction")
    st.text_input("Company Name (optional override)", key="company_name_override")
    close_tool_card()

    render_tool_card_header("▥", "Parser Actions", "Start processing, stop an active run, or reset the current workspace")
    col1, col2, col3 = st.columns(3)
    with col1:
        if button_compat("Start Parsing", primary=True, use_container_width=True):
            st.session_state.status = "running"
            st.session_state.affin_statement_totals = []
            st.session_state.affin_file_transactions = {}
            st.session_state.ambank_statement_totals = []
            st.session_state.ambank_file_transactions = {}
            st.session_state.cimb_statement_totals = []
            st.session_state.rhb_statement_totals = []
            st.session_state.gx_statement_totals = []
            st.session_state.cimb_file_transactions = {}
            st.session_state.rhb_file_transactions = {}
            st.session_state.gx_file_transactions = {}
            st.session_state.bank_islam_file_month = {}
            st.session_state.file_company_name = {}
            st.session_state.file_account_no = {}

    with col2:
        if button_compat("Stop", use_container_width=True):
            st.session_state.status = "stopped"

    with col3:
        if button_compat("Reset", use_container_width=True):
            st.session_state.status = "idle"
            st.session_state.results = []
            st.session_state.affin_statement_totals = []
            st.session_state.affin_file_transactions = {}
            st.session_state.ambank_statement_totals = []
            st.session_state.ambank_file_transactions = {}
            st.session_state.cimb_statement_totals = []
            st.session_state.rhb_statement_totals = []
            st.session_state.gx_statement_totals = []
            st.session_state.cimb_file_transactions = {}
            st.session_state.rhb_file_transactions = {}
            st.session_state.gx_file_transactions = {}
            st.session_state.bank_islam_file_month = {}
            st.session_state.file_company_name = {}
            st.session_state.file_account_no = {}
            st.session_state.pdf_password = ""
            st.session_state.company_name_override = ""
            st.rerun()

    render_status_card(st.session_state.status)
    close_tool_card()

with workspace_left:
    render_progress_panel(st.session_state.status, uploaded_files or [], has_existing_results)


all_tx: List[dict] = []

if uploaded_files and st.session_state.status == "running":
    bank_display_box = st.empty()
    progress_bar = st.progress(0)

    total_files = len(uploaded_files)
    parser = PARSERS[bank_choice]

    for file_idx, uploaded_file in enumerate(uploaded_files):
        if st.session_state.status == "stopped":
            st.warning("⏹️ Processing stopped by user.")
            break

        st.write(f"### 🗂️ Processing File: **{uploaded_file.name}**")
        bank_display_box.info(f"📄 Processing {bank_choice}: {uploaded_file.name}...")

        try:
            pdf_bytes = uploaded_file.getvalue()

            # decrypt if encrypted
            if is_pdf_encrypted(pdf_bytes):
                pdf_bytes = decrypt_pdf_bytes(pdf_bytes, st.session_state.pdf_password)

            # extract company name (FIXED)
            company_name = None
            try:
                with bytes_to_pdfplumber(pdf_bytes) as meta_pdf:
                    company_name = extract_company_name(meta_pdf, max_pages=2)
            except Exception:
                company_name = None

            # extract account number (NEW)
            account_no = None
            try:
                with bytes_to_pdfplumber(pdf_bytes) as meta_pdf:
                    account_no = extract_account_number(meta_pdf, max_pages=2)
            except Exception:
                account_no = None

            # manual override wins
            if (st.session_state.company_name_override or "").strip():
                company_name = st.session_state.company_name_override.strip()

            st.session_state.file_company_name[uploaded_file.name] = company_name
            st.session_state.file_account_no[uploaded_file.name] = account_no

            # Parse transactions (existing logic)
            if bank_choice == "Affin Bank":
                with bytes_to_pdfplumber(pdf_bytes) as pdf:
                    totals = extract_affin_statement_totals(pdf, uploaded_file.name)
                    st.session_state.affin_statement_totals.append(totals)
                    tx_raw = parse_affin_bank(pdf, uploaded_file.name) or []

            elif bank_choice == "Ambank":
                with bytes_to_pdfplumber(pdf_bytes) as pdf:
                    totals = extract_ambank_statement_totals(pdf, uploaded_file.name)
                    st.session_state.ambank_statement_totals.append(totals)
                    tx_raw = parse_ambank(pdf, uploaded_file.name) or []

            elif bank_choice == "CIMB Bank":
                with bytes_to_pdfplumber(pdf_bytes) as pdf:
                    totals = extract_cimb_statement_totals(pdf, uploaded_file.name)
                    st.session_state.cimb_statement_totals.append(totals)
                    tx_raw = parse_transactions_cimb(pdf, uploaded_file.name) or []

            elif bank_choice == "RHB Bank":
                with bytes_to_pdfplumber(pdf_bytes) as pdf:
                    totals = extract_rhb_statement_totals(pdf, uploaded_file.name)
                    st.session_state.rhb_statement_totals.append(totals)
                tx_raw = parser(pdf_bytes, uploaded_file.name) or []

            elif bank_choice == "GX Bank":
                with bytes_to_pdfplumber(pdf_bytes) as pdf:
                    totals = extract_gx_statement_totals(pdf, uploaded_file.name)
                    st.session_state.gx_statement_totals.append(totals)
                    tx_raw = parse_transactions_gx_bank(pdf, uploaded_file.name) or []

            elif bank_choice == "Bank Islam":
                with bytes_to_pdfplumber(pdf_bytes) as pdf:
                    tx_raw = parse_bank_islam(pdf, uploaded_file.name) or []
                    stmt_month = extract_bank_islam_statement_month(pdf)
                    if stmt_month:
                        st.session_state.bank_islam_file_month[uploaded_file.name] = stmt_month

            else:
                tx_raw = parser(pdf_bytes, uploaded_file.name) or []

            # Normalize then attach company_name
            tx_norm = normalize_transactions(
                tx_raw,
                default_bank=bank_choice,
                source_file=uploaded_file.name,
            )
            for t in tx_norm:
                if company_name:
                    t["company_name"] = company_name
                else:
                    t["company_name"] = t.get("company_name")
                if account_no:
                    t["account_no"] = account_no
                else:
                    t["account_no"] = t.get("account_no")

            if not company_name:
                for t in tx_norm:
                    cand = (t.get("company_name") or "").strip()
                    if cand:
                        company_name = cand
                        st.session_state.file_company_name[uploaded_file.name] = cand
                        break

            if bank_choice == "Affin Bank":
                st.session_state.affin_file_transactions[uploaded_file.name] = tx_norm
            if bank_choice == "Ambank":
                st.session_state.ambank_file_transactions[uploaded_file.name] = tx_norm
            if bank_choice == "CIMB Bank":
                st.session_state.cimb_file_transactions[uploaded_file.name] = tx_norm
            if bank_choice == "RHB Bank":
                st.session_state.rhb_file_transactions[uploaded_file.name] = tx_norm
            if bank_choice == "GX Bank":
                st.session_state.gx_file_transactions[uploaded_file.name] = tx_norm

            if tx_norm:
                st.success(f"✅ Extracted {len(tx_norm)} transactions from {uploaded_file.name}")
                all_tx.extend(tx_norm)
            else:
                st.warning(f"⚠️ No transactions found in {uploaded_file.name}")

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")
            st.exception(e)

        progress_bar.progress((file_idx + 1) / total_files)

    bank_display_box.success(f"🏦 Completed processing: **{bank_choice}**")

    all_tx = dedupe_transactions(all_tx)

    # Stable ordering
    for idx, t in enumerate(all_tx):
        if "__row_order" not in t:
            t["__row_order"] = idx

    def _sort_key(t: dict) -> Tuple:
        dt = parse_any_date_for_summary(t.get("date"))
        page = t.get("page")
        try:
            page_i = int(page) if page is not None else 10**9
        except Exception:
            page_i = 10**9

        seq = t.get("seq", None)
        try:
            seq_i = int(seq) if seq is not None else 10**9
        except Exception:
            seq_i = 10**9

        row_order = t.get("__row_order", 10**12)
        try:
            row_order_i = int(row_order)
        except Exception:
            row_order_i = 10**12

        return (
            dt if pd.notna(dt) else pd.Timestamp.max,
            page_i,
            seq_i,
            row_order_i,
        )

    all_tx = sorted(all_tx, key=_sort_key)
    st.session_state.results = all_tx


# =========================================================
# Monthly Summary Calculation (same logic, adds company_name)
# =========================================================
def calculate_monthly_summary(transactions: List[dict]) -> List[dict]:
    # Affin-only
    if bank_choice == "Affin Bank" and st.session_state.affin_statement_totals:
        rows: List[dict] = []
        for t in st.session_state.affin_statement_totals:
            month = t.get("statement_month") or "UNKNOWN"
            fname = t.get("source_file", "") or ""
            company_name = st.session_state.file_company_name.get(fname)
            account_no = st.session_state.file_account_no.get(fname)

            opening = t.get("opening_balance")
            ending = t.get("ending_balance")
            total_debit = t.get("total_debit")
            total_credit = t.get("total_credit")

            td = None if total_debit is None else round(float(safe_float(total_debit)), 2)
            tc = None if total_credit is None else round(float(safe_float(total_credit)), 2)

            opening_balance = round(float(safe_float(opening)), 2) if opening is not None else None
            ending_balance = round(float(safe_float(ending)), 2) if ending is not None else None

            txs = st.session_state.affin_file_transactions.get(fname, []) if fname else []
            tx_count = int(len(txs)) if txs else None

            balances: List[float] = []
            for x in txs:
                b = x.get("balance")
                if b is None:
                    continue
                try:
                    balances.append(float(safe_float(b)))
                except Exception:
                    pass

            if ending_balance is None and balances:
                ending_balance = round(float(balances[-1]), 2)

            lowest_balance = round(min(balances), 2) if balances else None
            highest_balance = round(max(balances), 2) if balances else None

            net_change = None
            if td is not None and tc is not None:
                net_change = round(float(tc - td), 2)

            if opening_balance is None and ending_balance is not None and td is not None and tc is not None:
                opening_balance = round(float(ending_balance - (tc - td)), 2)

            rows.append(
                {
                    "month": month,
                    "company_name": company_name,
                    "account_no": account_no,
                    "transaction_count": tx_count,
                    "opening_balance": opening_balance,
                    "total_debit": td,
                    "total_credit": tc,
                    "net_change": net_change,
                    "ending_balance": ending_balance,
                    "lowest_balance": lowest_balance,
                    "lowest_balance_raw": lowest_balance,
                    "highest_balance": highest_balance,
                    "od_flag": bool(lowest_balance is not None and float(lowest_balance) < 0),
                    "source_files": fname,
                }
            )
        return sorted(rows, key=lambda r: str(r.get("month", "9999-99")))

    # Ambank-only
    if bank_choice == "Ambank" and st.session_state.ambank_statement_totals:
        rows: List[dict] = []
        for t in st.session_state.ambank_statement_totals:
            month = t.get("statement_month") or "UNKNOWN"
            fname = t.get("source_file", "") or ""
            company_name = st.session_state.file_company_name.get(fname)
            account_no = st.session_state.file_account_no.get(fname)

            opening = t.get("opening_balance")
            ending = t.get("ending_balance")
            total_debit = t.get("total_debit")
            total_credit = t.get("total_credit")

            td = None if total_debit is None else round(float(safe_float(total_debit)), 2)
            tc = None if total_credit is None else round(float(safe_float(total_credit)), 2)

            opening_balance = round(float(safe_float(opening)), 2) if opening is not None else None
            ending_balance = round(float(safe_float(ending)), 2) if ending is not None else None

            txs = st.session_state.ambank_file_transactions.get(fname, []) if fname else []
            tx_count = int(len(txs)) if txs else None

            balances: List[float] = []
            for x in txs:
                b = x.get("balance")
                if b is None:
                    continue
                try:
                    balances.append(float(safe_float(b)))
                except Exception:
                    pass

            lowest_balance = round(min(balances), 2) if balances else None
            highest_balance = round(max(balances), 2) if balances else None

            net_change = None
            if td is not None and tc is not None:
                net_change = round(float(tc - td), 2)

            if opening_balance is None and ending_balance is not None and td is not None and tc is not None:
                opening_balance = round(float(ending_balance - (tc - td)), 2)

            rows.append(
                {
                    "month": month,
                    "company_name": company_name,
                    "account_no": account_no,
                    "transaction_count": tx_count,
                    "opening_balance": opening_balance,
                    "total_debit": td,
                    "total_credit": tc,
                    "net_change": net_change,
                    "ending_balance": ending_balance,
                    "lowest_balance": lowest_balance,
                    "lowest_balance_raw": lowest_balance,
                    "highest_balance": highest_balance,
                    "od_flag": bool(lowest_balance is not None and float(lowest_balance) < 0),
                    "source_files": fname,
                }
            )
        return sorted(rows, key=lambda r: str(r.get("month", "9999-99")))

    # CIMB-only
    if bank_choice == "CIMB Bank" and st.session_state.cimb_statement_totals:
        rows: List[dict] = []
        for t in st.session_state.cimb_statement_totals:
            month = t.get("statement_month") or "UNKNOWN"
            fname = t.get("source_file", "") or ""
            company_name = st.session_state.file_company_name.get(fname)
            account_no = st.session_state.file_account_no.get(fname)

            ending = t.get("ending_balance")
            total_debit = t.get("total_debit")
            total_credit = t.get("total_credit")

            td = None if total_debit is None else round(float(safe_float(total_debit)), 2)
            tc = None if total_credit is None else round(float(safe_float(total_credit)), 2)
            ending_balance = round(float(safe_float(ending)), 2) if ending is not None else None

            net_change = None
            opening_balance = None
            if td is not None and tc is not None:
                net_change = round(float(tc - td), 2)
                if ending_balance is not None:
                    opening_balance = round(float(ending_balance - (tc - td)), 2)

            txs = st.session_state.cimb_file_transactions.get(fname, []) if fname else []
            tx_count = int(len(txs)) if txs else None

            balances: List[float] = []
            for x in txs:
                desc = str(x.get("description") or "")
                if re.search(r"CLOSING\s+BALANCE\s*/\s*BAKI\s+PENUTUP", desc, flags=re.IGNORECASE):
                    continue
                b = x.get("balance")
                if b is None:
                    continue
                try:
                    balances.append(float(safe_float(b)))
                except Exception:
                    pass

            lowest_balance = round(min(balances), 2) if balances else None
            highest_balance = round(max(balances), 2) if balances else None

            rows.append(
                {
                    "month": month,
                    "company_name": company_name,
                    "account_no": account_no,
                    "transaction_count": tx_count,
                    "opening_balance": opening_balance,
                    "total_debit": td,
                    "total_credit": tc,
                    "net_change": net_change,
                    "ending_balance": ending_balance,
                    "lowest_balance": lowest_balance,
                    "lowest_balance_raw": lowest_balance,
                    "highest_balance": highest_balance,
                    "od_flag": bool(lowest_balance is not None and float(lowest_balance) < 0),
                    "source_files": fname,
                }
            )
        return sorted(rows, key=lambda r: str(r.get("month", "9999-99")))

    # RHB-only
    if bank_choice == "RHB Bank" and st.session_state.rhb_statement_totals:
        rows: List[dict] = []
        for t in st.session_state.rhb_statement_totals:
            month = t.get("statement_month") or "UNKNOWN"
            fname = t.get("source_file", "") or ""
            company_name = st.session_state.file_company_name.get(fname)
            account_no = st.session_state.file_account_no.get(fname)

            opening = t.get("opening_balance")
            ending = t.get("ending_balance")
            total_debit = t.get("total_debit")
            total_credit = t.get("total_credit")

            td = None if total_debit is None else round(float(safe_float(total_debit)), 2)
            tc = None if total_credit is None else round(float(safe_float(total_credit)), 2)
            opening_balance = round(float(safe_float(opening)), 2) if opening is not None else None
            ending_balance = round(float(safe_float(ending)), 2) if ending is not None else None

            txs = st.session_state.rhb_file_transactions.get(fname, []) if fname else []
            tx_count = int(len(txs)) if txs else None

            balances: List[float] = []
            for x in txs:
                b = x.get("balance")
                if b is None:
                    continue
                try:
                    balances.append(float(safe_float(b)))
                except Exception:
                    pass

            lowest_balance = round(min(balances), 2) if balances else None
            highest_balance = round(max(balances), 2) if balances else None

            net_change = None
            if td is not None and tc is not None:
                net_change = round(float(tc - td), 2)

            if opening_balance is None and ending_balance is not None and td is not None and tc is not None:
                opening_balance = round(float(ending_balance - (tc - td)), 2)

            rows.append(
                {
                    "month": month,
                    "company_name": company_name,
                    "account_no": account_no,
                    "transaction_count": tx_count,
                    "opening_balance": opening_balance,
                    "total_debit": td,
                    "total_credit": tc,
                    "net_change": net_change,
                    "ending_balance": ending_balance,
                    "lowest_balance": lowest_balance,
                    "lowest_balance_raw": lowest_balance,
                    "highest_balance": highest_balance,
                    "od_flag": bool(lowest_balance is not None and float(lowest_balance) < 0),
                    "source_files": fname,
                }
            )
        return sorted(rows, key=lambda r: str(r.get("month", "9999-99")))

    # GX-only
    if bank_choice == "GX Bank" and st.session_state.gx_statement_totals:
        rows: List[dict] = []
        for t in st.session_state.gx_statement_totals:
            month = t.get("statement_month") or "UNKNOWN"
            fname = t.get("source_file", "") or ""
            company_name = st.session_state.file_company_name.get(fname)
            account_no = st.session_state.file_account_no.get(fname)

            opening = t.get("opening_balance")
            ending = t.get("ending_balance")
            total_debit = t.get("total_debit")
            total_credit = t.get("total_credit")

            td = None if total_debit is None else round(float(safe_float(total_debit)), 2)
            tc = None if total_credit is None else round(float(safe_float(total_credit)), 2)
            opening_balance = round(float(safe_float(opening)), 2) if opening is not None else None
            ending_balance = round(float(safe_float(ending)), 2) if ending is not None else None

            txs = st.session_state.gx_file_transactions.get(fname, []) if fname else []
            tx_count = int(len(txs)) if txs else None

            balances: List[float] = []
            for x in txs:
                b = x.get("balance")
                if b is None:
                    continue
                try:
                    balances.append(float(safe_float(b)))
                except Exception:
                    pass

            lowest_balance = round(min(balances), 2) if balances else None
            highest_balance = round(max(balances), 2) if balances else None

            net_change = None
            if td is not None and tc is not None:
                net_change = round(float(tc - td), 2)

            if opening_balance is None and ending_balance is not None and td is not None and tc is not None:
                opening_balance = round(float(ending_balance - (tc - td)), 2)

            rows.append(
                {
                    "month": month,
                    "company_name": company_name,
                    "account_no": account_no,
                    "transaction_count": tx_count,
                    "opening_balance": opening_balance,
                    "total_debit": td,
                    "total_credit": tc,
                    "net_change": net_change,
                    "ending_balance": ending_balance,
                    "lowest_balance": lowest_balance,
                    "lowest_balance_raw": lowest_balance,
                    "highest_balance": highest_balance,
                    "od_flag": bool(lowest_balance is not None and float(lowest_balance) < 0),
                    "source_files": fname,
                }
            )
        return sorted(rows, key=lambda r: str(r.get("month", "9999-99")))

    # Default banks
    if not transactions:
        if bank_choice == "Bank Islam" and getattr(st.session_state, "bank_islam_file_month", {}):
            rows: List[dict] = []
            for fname, month in sorted(st.session_state.bank_islam_file_month.items(), key=lambda x: x[1]):
                company_name = st.session_state.file_company_name.get(fname)
                account_no = st.session_state.file_account_no.get(fname)
                rows.append(
                    {
                        "month": month,
                        "company_name": company_name,
                        "account_no": account_no,
                        "transaction_count": 0,
                        "opening_balance": None,
                        "total_debit": 0.0,
                        "total_credit": 0.0,
                        "net_change": 0.0,
                        "ending_balance": None,
                        "lowest_balance": None,
                        "lowest_balance_raw": None,
                        "highest_balance": None,
                        "od_flag": False,
                        "source_files": fname,
                    }
                )
            return rows
        return []

    df = pd.DataFrame(transactions)
    if df.empty:
        return []

    df = df.reset_index(drop=True)
    if "__row_order" not in df.columns:
        df["__row_order"] = range(len(df))

    df["date_parsed"] = df.get("date").apply(parse_any_date_for_summary)
    df = df.dropna(subset=["date_parsed"])
    if df.empty:
        st.warning("⚠️ No valid transaction dates found.")
        return []

    df["month_period"] = df["date_parsed"].dt.strftime("%Y-%m")
    df["debit"] = df.get("debit", 0).apply(safe_float)
    df["credit"] = df.get("credit", 0).apply(safe_float)
    df["balance"] = df.get("balance", None).apply(lambda x: safe_float(x) if x is not None else None)

    if "page" in df.columns:
        df["page"] = pd.to_numeric(df["page"], errors="coerce").fillna(0).astype(int)
    else:
        df["page"] = 0

    has_seq = "seq" in df.columns
    if has_seq:
        df["seq"] = pd.to_numeric(df["seq"], errors="coerce").fillna(0).astype(int)

    df["__row_order"] = pd.to_numeric(df["__row_order"], errors="coerce").fillna(0).astype(int)

    monthly_summary: List[dict] = []
    for period, group in df.groupby("month_period", sort=True):
        sort_cols = ["date_parsed", "page"]
        if has_seq:
            sort_cols.append("seq")
        sort_cols.append("__row_order")

        group_sorted = group.sort_values(sort_cols, na_position="last")

        balances = group_sorted["balance"].dropna()
        ending_balance = round(float(balances.iloc[-1]), 2) if not balances.empty else None
        highest_balance = round(float(balances.max()), 2) if not balances.empty else None
        lowest_balance_raw = round(float(balances.min()), 2) if not balances.empty else None
        lowest_balance = lowest_balance_raw
        od_flag = bool(lowest_balance is not None and float(lowest_balance) < 0)

        company_vals = [
            x for x in group_sorted.get("company_name", pd.Series([], dtype=object)).dropna().astype(str).unique().tolist()
            if x.strip()
        ]
        company_name = company_vals[0] if company_vals else None

        acct_vals = [
            x for x in group_sorted.get("account_no", pd.Series([], dtype=object)).dropna().astype(str).unique().tolist() if x.strip()
        ]
        account_no = acct_vals[0] if len(acct_vals) == 1 else (", ".join(acct_vals) if acct_vals else None)

        monthly_summary.append(
            {
                "month": period,
                "company_name": company_name,
                "account_no": account_no,
                "transaction_count": int(len(group_sorted)),
                "opening_balance": None,
                "total_debit": round(float(group_sorted["debit"].sum()), 2),
                "total_credit": round(float(group_sorted["credit"].sum()), 2),
                "net_change": round(float(group_sorted["credit"].sum() - group_sorted["debit"].sum()), 2),
                "ending_balance": ending_balance,
                "lowest_balance": lowest_balance,
                "lowest_balance_raw": lowest_balance_raw,
                "highest_balance": highest_balance,
                "od_flag": od_flag,
                "source_files": ", ".join(sorted(set(group_sorted.get("source_file", []))))
                if "source_file" in group_sorted.columns
                else "",
            }
        )

    # Bank Islam ensure statement months with zero tx still appear
    if bank_choice == "Bank Islam" and getattr(st.session_state, "bank_islam_file_month", {}):
        existing_months = {r.get("month") for r in monthly_summary}
        for fname, month in st.session_state.bank_islam_file_month.items():
            if month in existing_months:
                continue
            company_name = st.session_state.file_company_name.get(fname)
            account_no = st.session_state.file_account_no.get(fname)
            monthly_summary.append(
                {
                    "month": month,
                    "company_name": company_name,
                    "account_no": account_no,
                    "transaction_count": 0,
                    "opening_balance": None,
                    "total_debit": 0.0,
                    "total_credit": 0.0,
                    "net_change": 0.0,
                    "ending_balance": None,
                    "lowest_balance": None,
                    "lowest_balance_raw": None,
                    "highest_balance": None,
                    "od_flag": False,
                    "source_files": fname,
                }
            )

    # Fill opening_balance for default banks using prior month's ending_balance when possible.
    monthly_summary_sorted = sorted(monthly_summary, key=lambda x: x["month"])
    prev_end = None
    for r in monthly_summary_sorted:
        if r.get("opening_balance") is None:
            if prev_end is not None:
                r["opening_balance"] = round(float(prev_end), 2)
            else:
                # best-effort fallback: opening = ending - net_change
                eb = r.get("ending_balance")
                nc = r.get("net_change")
                if eb is not None and nc is not None:
                    try:
                        r["opening_balance"] = round(float(safe_float(eb) - safe_float(nc)), 2)
                    except Exception:
                        r["opening_balance"] = None

        # update prev_end for next month
        if r.get("ending_balance") is not None:
            prev_end = safe_float(r.get("ending_balance"))

    return monthly_summary_sorted


# =========================================================
# Presentation-only Monthly Summary Standardization
# =========================================================
def present_monthly_summary_standard(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows or []:
        highest = r.get("highest_balance")
        lowest = r.get("lowest_balance")

        swing = None
        try:
            if highest is not None and lowest is not None:
                swing = round(float(safe_float(highest) - safe_float(lowest)), 2)
        except Exception:
            swing = None

        out.append(
            {
                "month": r.get("month"),
                "company_name": r.get("company_name"),
                "account_no": r.get("account_no"),
                "opening_balance": r.get("opening_balance"),
                "total_debit": r.get("total_debit"),
                "total_credit": r.get("total_credit"),
                "highest_balance": highest,
                "lowest_balance": lowest,
                "swing": swing,
                "ending_balance": r.get("ending_balance"),
                "source_files": r.get("source_files"),
            }
        )
    return out


# ---------------------------------------------------
# DISPLAY
# ---------------------------------------------------
if st.session_state.results or (bank_choice == "Affin Bank" and st.session_state.affin_statement_totals) or (
    bank_choice == "Ambank" and st.session_state.ambank_statement_totals
) or (bank_choice == "CIMB Bank" and st.session_state.cimb_statement_totals) or (
    bank_choice == "RHB Bank" and st.session_state.rhb_statement_totals
):
    df = pd.DataFrame(st.session_state.results) if st.session_state.results else pd.DataFrame()

    monthly_summary_raw = calculate_monthly_summary(st.session_state.results)
    monthly_summary = present_monthly_summary_standard(monthly_summary_raw)

    date_min = df["date"].min() if "date" in df.columns and not df.empty else None
    date_max = df["date"].max() if "date" in df.columns and not df.empty else None

    total_files_processed = None
    if "source_file" in df.columns and not df.empty:
        total_files_processed = int(df["source_file"].nunique())
    else:
        if bank_choice == "Affin Bank":
            total_files_processed = len(st.session_state.affin_statement_totals)
        elif bank_choice == "Ambank":
            total_files_processed = len(st.session_state.ambank_statement_totals)
        elif bank_choice == "CIMB Bank":
            total_files_processed = len(st.session_state.cimb_statement_totals)
        elif bank_choice == "RHB Bank":
            total_files_processed = len(st.session_state.rhb_statement_totals)

    summary_range = f"{date_min} to {date_max}" if date_min and date_max else "Not available"
    render_metric_cards(
        [
            ("Bank Format", bank_choice),
            ("Files Processed", str(total_files_processed or 0)),
            ("Transactions", str(len(df))),
            ("Date Range", summary_range),
        ]
    )

    render_section_header(
        "Results",
        "Extracted transactions",
        "Review normalized line items before exporting or moving to the monthly summary.",
    )

    if not df.empty:
        display_cols = [
            "date",
            "description",
            "debit",
            "credit",
            "balance",
            "company_name",
            "account_no",
            "page",
            "seq",
            "bank",
            "source_file",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)
    else:
        st.info("No line-item transactions extracted.")

    if monthly_summary:
        render_section_header(
            "Summary",
            "Monthly summary (standardized)",
            "Opening balances, total flows, and ending balances are preserved from the existing summary logic.",
        )
        summary_df = pd.DataFrame(monthly_summary)
        desired_cols = [
            "month",
            "company_name",
            "account_no",
            "opening_balance",
            "total_debit",
            "total_credit",
            "highest_balance",
            "lowest_balance",
            "swing",
            "ending_balance",
            "source_files",
        ]
        summary_df = summary_df[[c for c in desired_cols if c in summary_df.columns]]
        st.dataframe(summary_df, use_container_width=True)
    
    render_section_header(
        "Exports",
        "Download options",
        "Export transactions only, or generate full JSON and XLSX reports using the same underlying data.",
    )
    col1, col2, col3 = st.columns(3)

    df_download = df.copy() if not df.empty else pd.DataFrame([])

    with col1:
        download_button_compat(
            "📄 Download Transactions (JSON)",
            json.dumps(df_download.to_dict(orient="records"), indent=4),
            "transactions.json",
            "application/json",
            use_container_width=True,
        )

    with col2:
        company_names = sorted(
            {x for x in df_download.get("company_name", pd.Series([], dtype=object)).dropna().astype(str).tolist() if x.strip()}
        )

        account_nos = sorted(
            {x for x in df_download.get("account_no", pd.Series([], dtype=object)).dropna().astype(str).tolist() if x.strip()}
        )

        full_report = {
            "summary": {
                "total_transactions": int(len(df_download)),
                "date_range": f"{date_min} to {date_max}" if date_min and date_max else None,
                "total_files_processed": total_files_processed,
                "company_names": company_names,
                "account_nos": account_nos,
                "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "monthly_summary": monthly_summary,
            "transactions": df_download.to_dict(orient="records"),
        }

        download_button_compat(
            "📊 Download Full Report (JSON)",
            json.dumps(full_report, indent=4),
            "full_report.json",
            "application/json",
            use_container_width=True,
        )

    with col3:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_download.to_excel(writer, sheet_name="Transactions", index=False)
            if monthly_summary:
                pd.DataFrame(monthly_summary).to_excel(writer, sheet_name="Monthly Summary", index=False)

        download_button_compat(
            "📊 Download Full Report (XLSX)",
            output.getvalue(),
            "full_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

else:
    if uploaded_files:
        st.warning("⚠️ No transactions found — click **Start Processing**.")
