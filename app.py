import json
import re
from datetime import datetime
from io import BytesIO
from typing import Callable, Dict, List, Tuple

import pandas as pd
import streamlit as st

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
from ambank import parse_ambank
from bank_muamalat import parse_transactions_bank_muamalat
from affin_bank import parse_affin_bank, extract_affin_statement_totals
from agro_bank import parse_agro_bank
from ocbc import parse_transactions_ocbc


st.set_page_config(page_title="Bank Statement Parser", layout="wide")
st.title("📄 Bank Statement Parser (Multi-File Support)")
st.write("Upload one or more bank statement PDFs to extract transactions.")


if "status" not in st.session_state:
    st.session_state.status = "idle"

if "results" not in st.session_state:
    st.session_state.results = []

if "affin_statement_totals" not in st.session_state:
    st.session_state.affin_statement_totals = []

if "affin_file_transactions" not in st.session_state:
    st.session_state.affin_file_transactions = {}


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


PARSERS: Dict[str, Callable[[bytes, str], List[dict]]] = {
    "Affin Bank": lambda b, f: _parse_with_pdfplumber(parse_affin_bank, b, f),
    "Agro Bank": lambda b, f: _parse_with_pdfplumber(parse_agro_bank, b, f),
    "Ambank": lambda b, f: _parse_with_pdfplumber(parse_ambank, b, f),
    "Bank Islam": lambda b, f: _parse_with_pdfplumber(parse_bank_islam, b, f),
    "Bank Muamalat": lambda b, f: _parse_with_pdfplumber(parse_transactions_bank_muamalat, b, f),
    "Bank Rakyat": lambda b, f: _parse_with_pdfplumber(parse_bank_rakyat, b, f),
    "CIMB Bank": lambda b, f: _parse_with_pdfplumber(parse_transactions_cimb, b, f),
    "Hong Leong": lambda b, f: _parse_with_pdfplumber(parse_hong_leong, b, f),
    "Maybank": lambda b, f: parse_transactions_maybank(b, f),
    "Public Bank (PBB)": lambda b, f: _parse_with_pdfplumber(parse_transactions_pbb, b, f),
    "RHB Bank": lambda b, f: parse_transactions_rhb(b, f),
    "OCBC Bank": lambda b, f: parse_transactions_ocbc(b, f),
}


bank_choice = st.selectbox("Select Bank Format", list(PARSERS.keys()))

uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    uploaded_files = sorted(uploaded_files, key=lambda x: x.name)


col1, col2, col3 = st.columns(3)
with col1:
    if st.button("▶️ Start Processing"):
        st.session_state.status = "running"
        st.session_state.affin_statement_totals = []
        st.session_state.affin_file_transactions = {}

with col2:
    if st.button("⏹️ Stop"):
        st.session_state.status = "stopped"

with col3:
    if st.button("🔄 Reset"):
        st.session_state.status = "idle"
        st.session_state.results = []
        st.session_state.affin_statement_totals = []
        st.session_state.affin_file_transactions = {}
        st.rerun()

st.write(f"### ⚙️ Status: **{st.session_state.status.upper()}**")


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

            if bank_choice == "Affin Bank":
                with bytes_to_pdfplumber(pdf_bytes) as pdf:
                    totals = extract_affin_statement_totals(pdf, uploaded_file.name)
                    st.session_state.affin_statement_totals.append(totals)

                    tx_raw = parse_affin_bank(pdf, uploaded_file.name) or []
            else:
                tx_raw = parser(pdf_bytes, uploaded_file.name) or []

            tx_norm = normalize_transactions(
                tx_raw,
                default_bank=bank_choice,
                source_file=uploaded_file.name,
            )

            if bank_choice == "Affin Bank":
                st.session_state.affin_file_transactions[uploaded_file.name] = tx_norm

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

    # Dedupe first
    all_tx = dedupe_transactions(all_tx)

    # ------------------------------------------------------------------
    # FIX 1: Stable global ordering (prevents “ending balance wrong” ties)
    # Many banks (esp. AmBank) have multiple rows with same date+page.
    # If you sort only by (date, page), “last row” becomes unstable.
    # We add a deterministic tiebreaker:
    #   - prefer tx["seq"] when available (parser-provided order)
    #   - else fall back to preserved original list order
    # ------------------------------------------------------------------
    for idx, t in enumerate(all_tx):
        # internal stable order fallback; harmless if already exists
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

        # IMPORTANT: do NOT use description as a tiebreaker; it can reshuffle.
        return (
            dt if pd.notna(dt) else pd.Timestamp.max,
            page_i,
            seq_i,
            row_order_i,
        )

    all_tx = sorted(all_tx, key=_sort_key)
    st.session_state.results = all_tx


def calculate_monthly_summary(transactions: List[dict]) -> List[dict]:
    # -------------------------
    # Affin-only: statement totals (with derived opening)
    # -------------------------
    if bank_choice == "Affin Bank" and st.session_state.affin_statement_totals:
        rows: List[dict] = []

        for t in st.session_state.affin_statement_totals:
            month = t.get("statement_month") or "UNKNOWN"
            fname = t.get("source_file", "") or ""

            opening = t.get("opening_balance")
            ending = t.get("ending_balance")
            total_debit = t.get("total_debit")
            total_credit = t.get("total_credit")

            td = None if total_debit is None else round(float(safe_float(total_debit)), 2)
            tc = None if total_credit is None else round(float(safe_float(total_credit)), 2)

            opening_balance = round(float(safe_float(opening)), 2) if opening is not None else None
            ending_balance = round(float(safe_float(ending)), 2) if ending is not None else None

            # Pull extracted transactions ONLY for fallback ending and min/max
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

            # If opening is missing but we have ending and totals, derive opening from accounting identity.
            # opening = ending - (credit - debit)
            if opening_balance is None and ending_balance is not None and td is not None and tc is not None:
                opening_balance = round(float(ending_balance - (tc - td)), 2)

            rows.append(
                {
                    "month": month,
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

        rows = sorted(rows, key=lambda r: str(r.get("month", "9999-99")))

        # Reconcile opening chain for stability across files (does not touch totals)
        tol = 0.01
        for i in range(1, len(rows)):
            prev_close = rows[i - 1].get("ending_balance")
            cur_open = rows[i].get("opening_balance")
            if prev_close is None:
                continue
            try:
                prev_close_f = float(prev_close)
            except Exception:
                continue

            mismatch = False
            if cur_open is None:
                mismatch = True
            else:
                try:
                    if abs(float(cur_open) - prev_close_f) > tol:
                        mismatch = True
                except Exception:
                    mismatch = True

            if mismatch:
                rows[i]["opening_balance_raw"] = cur_open
                rows[i]["opening_balance"] = round(prev_close_f, 2)

        missing = [r for r in rows if r.get("total_debit") is None or r.get("total_credit") is None]
        if missing:
            st.warning(
                "Affin: Some files are missing printed TOTAL DEBIT/TOTAL CREDIT from statement extraction. "
                "They are left as None to avoid OCR-inflated numbers."
            )

        return rows

    # -------------------------
    # Default for other banks (FIXED: stable ending balance)
    # -------------------------
    if not transactions:
        return []
    df = pd.DataFrame(transactions)
    if df.empty:
        return []

    # FIX 2: deterministic row order to break ties when seq is missing
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

    # Make sure page/seq are sortable
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
        # CRITICAL FIX:
        # Sort by date + page + seq (if present) else __row_order
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

        monthly_summary.append(
            {
                "month": period,
                "transaction_count": int(len(group_sorted)),
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

    return sorted(monthly_summary, key=lambda x: x["month"])


# ---------------------------------------------------
# DISPLAY
# ---------------------------------------------------
if st.session_state.results or (bank_choice == "Affin Bank" and st.session_state.affin_statement_totals):
    st.subheader("📊 Extracted Transactions")
    df = pd.DataFrame(st.session_state.results) if st.session_state.results else pd.DataFrame()

    if not df.empty:
        display_cols = ["date", "description", "debit", "credit", "balance", "page", "seq", "bank", "source_file"]
        display_cols = [c for c in display_cols if c in df.columns]
        df_display = df[display_cols] if display_cols else df
        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No line-item transactions extracted. For Affin, monthly totals are computed from statement totals when available.")

    monthly_summary = calculate_monthly_summary(st.session_state.results)
    if monthly_summary:
        st.subheader("📅 Monthly Summary")
        summary_df = pd.DataFrame(monthly_summary)
        st.dataframe(summary_df, use_container_width=True)

    st.subheader("⬇️ Download Options")
    col1, col2, col3 = st.columns(3)

    df_display = df.copy() if not df.empty else pd.DataFrame([])

    with col1:
        st.download_button(
            "📄 Download Transactions (JSON)",
            json.dumps(df_display.to_dict(orient="records"), indent=4),
            "transactions.json",
            "application/json",
        )

    with col2:
        date_min = df_display["date"].min() if "date" in df_display.columns and not df_display.empty else None
        date_max = df_display["date"].max() if "date" in df_display.columns and not df_display.empty else None

        full_report = {
            "summary": {
                "total_transactions": int(len(df_display)),
                "date_range": f"{date_min} to {date_max}" if date_min and date_max else None,
                "total_files_processed": int(df_display["source_file"].nunique())
                if "source_file" in df_display.columns and not df_display.empty
                else (len(st.session_state.affin_statement_totals) if bank_choice == "Affin Bank" else None),
                "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
            "monthly_summary": monthly_summary,
            "transactions": df_display.to_dict(orient="records"),
        }

        st.download_button(
            "📊 Download Full Report (JSON)",
            json.dumps(full_report, indent=4),
            "full_report.json",
            "application/json",
        )

    with col3:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_display.to_excel(writer, sheet_name="Transactions", index=False)
            if monthly_summary:
                pd.DataFrame(monthly_summary).to_excel(writer, sheet_name="Monthly Summary", index=False)

        st.download_button(
            "📊 Download Full Report (XLSX)",
            output.getvalue(),
            "full_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

else:
    if uploaded_files:
        st.warning("⚠️ No transactions found — click **Start Processing**.")
