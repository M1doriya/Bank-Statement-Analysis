# app.py
# Streamlit Multi-Bank Statement Parser (Multi-File Support)
#
# FIX INCLUDED:
#   Monthly summary month-bucketing was wrong for ISO dates (YYYY-MM-DD) because
#   pd.to_datetime(..., dayfirst=True) can flip month/day on ISO strings where both <= 12.
#   This version parses ISO strictly with format="%Y-%m-%d" and only uses dayfirst=True
#   for non-ISO formats. Same fix applied to sorting.
#
# NEW FIX (Bank Islam only):
#   Bank Islam statements may show temporary negative ledger balances which are reversed
#   on the same date (e.g., "REVERSE POSTED DEBIT"). These should NOT be treated as OD.
#   We keep balances as-is for audit, but compute lowest_balance_for_od by ignoring
#   transient reversed negative dips (Bank Islam only). Other banks unchanged.
#
# NEW FIX (Affin only):
#   Affin scanned statements produce unreliable line-item OCR sums.
#   Monthly totals should come from statement printed totals (opening/total debit/total credit/ending).
#   We still extract transactions (OCR) for listing and for min/max/count if available.

import json
import re
from datetime import datetime
from io import BytesIO
from typing import Callable, Dict, List, Tuple, Optional

import pandas as pd
import streamlit as st

from core_utils import (
    bytes_to_pdfplumber,
    dedupe_transactions,
    normalize_transactions,
    safe_float,
)

# ---------------------------------------------------
# Import bank parsers
# ---------------------------------------------------
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


# ---------------------------------------------------
# Streamlit Setup
# ---------------------------------------------------
st.set_page_config(page_title="Bank Statement Parser", layout="wide")
st.title("📄 Bank Statement Parser (Multi-File Support)")
st.write("Upload one or more bank statement PDFs to extract transactions.")


# ---------------------------------------------------
# Session State
# ---------------------------------------------------
if "status" not in st.session_state:
    st.session_state.status = "idle"  # idle, running, stopped

if "results" not in st.session_state:
    st.session_state.results = []

# Affin-only: store statement totals per file (ground truth)
if "affin_statement_totals" not in st.session_state:
    st.session_state.affin_statement_totals = []

# Affin-only: store normalized transactions per file (for count/min/max/ending fallback)
if "affin_file_transactions" not in st.session_state:
    st.session_state.affin_file_transactions = {}  # filename -> List[dict]


# ---------------------------------------------------
# Date parsing helper (FIX)
# ---------------------------------------------------
_ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def parse_any_date_for_summary(x) -> pd.Timestamp:
    """
    Parse dates safely:
    - ISO YYYY-MM-DD: parse with explicit format (never dayfirst)
    - Otherwise: fall back to dayfirst=True for DD/MM/YYYY etc.
    """
    if x is None:
        return pd.NaT
    s = str(x).strip()
    if not s:
        return pd.NaT

    if _ISO_RE.match(s):
        return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")

    return pd.to_datetime(s, errors="coerce", dayfirst=True)


# ---------------------------------------------------
# Parser Registry Helper
# ---------------------------------------------------
def _parse_with_pdfplumber(parser_func: Callable, pdf_bytes: bytes, filename: str) -> List[dict]:
    """Open pdfplumber once for parsers expecting a pdfplumber.PDF."""
    with bytes_to_pdfplumber(pdf_bytes) as pdf:
        return parser_func(pdf, filename)


# ---------------------------------------------------
# Parser Registry
# All parsers are called as: parser(pdf_bytes, filename) -> List[dict]
# ---------------------------------------------------
PARSERS: Dict[str, Callable[[bytes, str], List[dict]]] = {
    "Affin Bank": lambda b, f: _parse_with_pdfplumber(parse_affin_bank, b, f),
    "Agro Bank": lambda b, f: _parse_with_pdfplumber(parse_agro_bank, b, f),
    "Ambank": lambda b, f: _parse_with_pdfplumber(parse_ambank, b, f),
    "Bank Islam": lambda b, f: _parse_with_pdfplumber(parse_bank_islam, b, f),
    "Bank Muamalat": lambda b, f: _parse_with_pdfplumber(parse_transactions_bank_muamalat, b, f),
    "Bank Rakyat": lambda b, f: _parse_with_pdfplumber(parse_bank_rakyat, b, f),
    "CIMB Bank": lambda b, f: _parse_with_pdfplumber(parse_transactions_cimb, b, f),
    "Hong Leong": lambda b, f: _parse_with_pdfplumber(parse_hong_leong, b, f),

    # Maybank parser should accept bytes (fitz open via stream)
    "Maybank": lambda b, f: parse_transactions_maybank(b, f),

    "Public Bank (PBB)": lambda b, f: _parse_with_pdfplumber(parse_transactions_pbb, b, f),

    # RHB parser should accept bytes
    "RHB Bank": lambda b, f: parse_transactions_rhb(b, f),

    # OCBC parser accepts bytes
    "OCBC Bank": lambda b, f: parse_transactions_ocbc(b, f),
}


# ---------------------------------------------------
# Bank Selection
# ---------------------------------------------------
bank_choice = st.selectbox("Select Bank Format", list(PARSERS.keys()))


# ---------------------------------------------------
# File Upload
# ---------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if uploaded_files:
    uploaded_files = sorted(uploaded_files, key=lambda x: x.name)


# ---------------------------------------------------
# Start / Stop / Reset Controls
# ---------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("▶️ Start Processing"):
        st.session_state.status = "running"
        # reset Affin-only caches each run
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


# ---------------------------------------------------
# MAIN PROCESSING
# ---------------------------------------------------
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

            # 1) Parse (bank-specific)
            if bank_choice == "Affin Bank":
                # Affin: also extract statement totals (ground truth for monthly summary)
                with bytes_to_pdfplumber(pdf_bytes) as pdf:
                    totals = extract_affin_statement_totals(pdf, uploaded_file.name)
                    st.session_state.affin_statement_totals.append(totals)

                    tx_raw = parse_affin_bank(pdf, uploaded_file.name) or []
            else:
                tx_raw = parser(pdf_bytes, uploaded_file.name) or []

            # 2) Normalize schema/types
            tx_norm = normalize_transactions(
                tx_raw,
                default_bank=bank_choice,
                source_file=uploaded_file.name,
            )

            # Affin: store per-file normalized tx for count/min/max/ending fallback
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

    # 3) De-duplicate across files
    all_tx = dedupe_transactions(all_tx)

    # 4) Sort deterministically by date -> page -> description
    def _sort_key(t: dict) -> Tuple:
        dt = parse_any_date_for_summary(t.get("date"))
        return (
            dt if pd.notna(dt) else pd.Timestamp.max,
            t.get("page") if t.get("page") is not None else 10**9,
            t.get("description", ""),
        )

    all_tx = sorted(all_tx, key=_sort_key)
    st.session_state.results = all_tx


# ---------------------------------------------------
# CALCULATE MONTHLY SUMMARY (FIXED + Bank Islam OD logic + Affin totals logic)
# ---------------------------------------------------
def calculate_monthly_summary(transactions: List[dict]) -> List[dict]:
    # -------------------------
    # Affin-only: monthly summary from statement totals, plus optional tx-derived stats
    # -------------------------
    if bank_choice == "Affin Bank" and st.session_state.affin_statement_totals:
        rows: List[dict] = []

        for t in st.session_state.affin_statement_totals:
            month = t.get("statement_month") or "UNKNOWN"
            fname = t.get("source_file", "") or ""

            opening = t.get("opening_balance")
            total_debit = t.get("total_debit")
            total_credit = t.get("total_credit")
            ending = t.get("ending_balance")

            td = round(float(safe_float(total_debit)), 2)
            tc = round(float(safe_float(total_credit)), 2)

            opening_balance = round(float(safe_float(opening)), 2) if opening is not None else None

            # Pull per-file transactions (if any) to compute count/min/max and ending fallback
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

            # Ending balance:
            # 1) prefer statement ending_balance
            # 2) else fallback to last extracted transaction balance (if available)
            ending_balance = None
            if ending is not None:
                ending_balance = round(float(safe_float(ending)), 2)
            elif balances:
                ending_balance = round(float(balances[-1]), 2)

            # Lowest/highest from extracted balances (if available)
            lowest_balance = round(min(balances), 2) if balances else None
            highest_balance = round(max(balances), 2) if balances else None

            od_flag = bool(lowest_balance is not None and float(lowest_balance) < 0)

            rows.append(
                {
                    "month": month,
                    "transaction_count": tx_count,
                    "opening_balance": opening_balance,
                    "total_debit": td,
                    "total_credit": tc,
                    "net_change": round(tc - td, 2),
                    "ending_balance": ending_balance,
                    "lowest_balance": lowest_balance,
                    "lowest_balance_raw": lowest_balance,
                    "highest_balance": highest_balance,
                    "od_flag": od_flag,
                    "source_files": fname,
                }
            )

        return sorted(rows, key=lambda r: str(r.get("month", "9999-99")))

    # -------------------------
    # Default path for other banks (your existing logic)
    # -------------------------
    if not transactions:
        return []

    df = pd.DataFrame(transactions)
    if df.empty:
        return []

    # FIX: Use safe parser that treats ISO unambiguously
    df["date_parsed"] = df.get("date").apply(parse_any_date_for_summary)
    df = df.dropna(subset=["date_parsed"])
    if df.empty:
        st.warning("⚠️ No valid transaction dates found.")
        return []

    df["month_period"] = df["date_parsed"].dt.strftime("%Y-%m")

    # Defensive numeric normalization
    df["debit"] = df.get("debit", 0).apply(safe_float)
    df["credit"] = df.get("credit", 0).apply(safe_float)
    df["balance"] = df.get("balance", None).apply(lambda x: safe_float(x) if x is not None else None)

    def _transient_negative_mask_bank_islam(group_sorted: pd.DataFrame) -> pd.Series:
        """
        Bank Islam statements may show temporary negative ledger balances which are reversed
        on the same date (e.g., "REVERSE POSTED DEBIT"). These should not be treated as OD.

        We mark negative balances as 'transient' if:
          - balance < 0 at row i
          - within the next 1..3 rows on the same date, there is a row containing REVERSE/REVERSAL
            and its balance returns to the last known non-negative balance (within 0.01).
        """
        if group_sorted.empty:
            return pd.Series([], dtype=bool)

        g = group_sorted.reset_index(drop=True).copy()
        g["_desc_up"] = g.get("description", "").astype(str).str.upper()

        is_transient = [False] * len(g)
        last_nonneg: Optional[float] = None

        for i in range(len(g)):
            bal = g.at[i, "balance"]
            dt = g.at[i, "date_parsed"]

            try:
                bal_f = float(bal) if bal is not None else None
            except Exception:
                bal_f = None

            # update anchor
            if bal_f is not None and bal_f >= 0:
                last_nonneg = bal_f
                continue

            # consider negative only
            if bal_f is None or bal_f >= 0:
                continue

            # cannot classify without a prior non-negative anchor
            if last_nonneg is None:
                continue

            # look ahead up to 3 rows on same date
            for j in range(i + 1, min(i + 4, len(g))):
                dt_j = g.at[j, "date_parsed"]
                if pd.isna(dt) or pd.isna(dt_j) or dt_j.date() != dt.date():
                    break

                desc_j = g.at[j, "_desc_up"]
                bal_j = g.at[j, "balance"]
                try:
                    bal_jf = float(bal_j) if bal_j is not None else None
                except Exception:
                    bal_jf = None

                if bal_jf is None:
                    continue

                if ("REVERSE" in desc_j or "REVERSAL" in desc_j) and abs(bal_jf - last_nonneg) <= 0.01:
                    # mark negative rows from i..j-1 as transient
                    for k in range(i, j):
                        try:
                            bk = float(g.at[k, "balance"])
                        except Exception:
                            bk = None
                        if bk is not None and bk < 0:
                            is_transient[k] = True
                    break

        return pd.Series(is_transient, index=group_sorted.index)

    monthly_summary: List[dict] = []
    for period, group in df.groupby("month_period", sort=True):
        group_sorted = group.sort_values(["date_parsed", "page"], na_position="last")

        balances = group_sorted["balance"].dropna()
        ending_balance = round(float(balances.iloc[-1]), 2) if not balances.empty else None
        highest_balance = round(float(balances.max()), 2) if not balances.empty else None

        # Raw lowest (audit truth)
        lowest_balance_raw = round(float(balances.min()), 2) if not balances.empty else None

        # Bank Islam only: adjust lowest balance used for OD decisions
        bank_names = set(str(x) for x in group_sorted.get("bank", []).dropna().unique())
        if "Bank Islam" in bank_names and not balances.empty:
            transient_mask = _transient_negative_mask_bank_islam(group_sorted)
            balances_for_od = group_sorted.loc[~transient_mask, "balance"].dropna()
            lowest_balance = (
                round(float(balances_for_od.min()), 2) if not balances_for_od.empty else lowest_balance_raw
            )
        else:
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
# DISPLAY RESULTS
# ---------------------------------------------------
if st.session_state.results or (bank_choice == "Affin Bank" and st.session_state.affin_statement_totals):
    st.subheader("📊 Extracted Transactions")

    df = pd.DataFrame(st.session_state.results) if st.session_state.results else pd.DataFrame()

    if not df.empty:
        display_cols = ["date", "description", "debit", "credit", "balance", "page", "bank", "source_file"]
        display_cols = [c for c in display_cols if c in df.columns]
        df_display = df[display_cols] if display_cols else df
        st.dataframe(df_display, use_container_width=True)
    else:
        st.info("No line-item transactions extracted. For Affin, monthly totals can still be computed from statement totals.")

    monthly_summary = calculate_monthly_summary(st.session_state.results)
    if monthly_summary:
        st.subheader("📅 Monthly Summary")
        summary_df = pd.DataFrame(monthly_summary)
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if "transaction_count" in summary_df.columns and summary_df["transaction_count"].notna().any():
                st.metric("Total Transactions", int(summary_df["transaction_count"].dropna().sum()))
            else:
                st.metric("Total Transactions", "—")
        with col2:
            st.metric("Total Debits", f"RM {summary_df['total_debit'].sum():,.2f}")
        with col3:
            st.metric("Total Credits", f"RM {summary_df['total_credit'].sum():,.2f}")
        with col4:
            net_total = float(summary_df["net_change"].sum())
            st.metric("Net Change", f"RM {net_total:,.2f}")

    # ---------------------------------------------------
    # DOWNLOAD OPTIONS
    # ---------------------------------------------------
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
