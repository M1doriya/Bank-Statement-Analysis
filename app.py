import json
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

# ---------------------------------------------------
# Import standalone parsers
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
from affin_bank import parse_affin_bank
from agro_bank import parse_agro_bank


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


# ---------------------------------------------------
# Parser Registry
# ---------------------------------------------------
def _parse_with_pdfplumber(parser_func: Callable, pdf_bytes: bytes, filename: str) -> List[dict]:
    """Open pdfplumber once for parsers expecting a pdfplumber.PDF."""
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
    # Maybank parser supports bytes directly
    "Maybank": lambda b, f: parse_transactions_maybank(b, f),
    "Public Bank (PBB)": lambda b, f: _parse_with_pdfplumber(parse_transactions_pbb, b, f),
    # RHB parser supports bytes; pass bytes for consistency
    "RHB Bank": lambda b, f: parse_transactions_rhb(b, f),
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

with col2:
    if st.button("⏹️ Stop"):
        st.session_state.status = "stopped"

with col3:
    if st.button("🔄 Reset"):
        st.session_state.status = "idle"
        st.session_state.results = []
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
            tx_raw = parser(pdf_bytes, uploaded_file.name) or []

            # Normalize schema/types and de-risk downstream (summary, exports, fraud)
            tx_norm = normalize_transactions(
                tx_raw,
                default_bank=bank_choice,
                source_file=uploaded_file.name,
            )

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

    # De-duplicate across multi-files (common when users upload overlapping statements)
    all_tx = dedupe_transactions(all_tx)

    # Sort by date (best-effort), then page
    def _sort_key(t: dict) -> Tuple:
        d = t.get("date")
        try:
            dt = pd.to_datetime(d, errors="coerce", dayfirst=True)
        except Exception:
            dt = pd.NaT
        return (
            dt if pd.notna(dt) else pd.Timestamp.max,
            t.get("page") if t.get("page") is not None else 10**9,
            t.get("description", ""),
        )

    all_tx = sorted(all_tx, key=_sort_key)
    st.session_state.results = all_tx


# ---------------------------------------------------
# CALCULATE MONTHLY SUMMARY
# ---------------------------------------------------
def calculate_monthly_summary(transactions: List[dict]) -> List[dict]:
    if not transactions:
        return []

    df = pd.DataFrame(transactions)
    if df.empty:
        return []

    # Robust, day-first
    df["date_parsed"] = pd.to_datetime(df.get("date"), errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date_parsed"])
    if df.empty:
        st.warning("⚠️ No valid transaction dates found.")
        return []

    df["month_period"] = df["date_parsed"].dt.strftime("%Y-%m")

    # Normalize numeric columns defensively
    df["debit"] = df.get("debit", 0).apply(safe_float)
    df["credit"] = df.get("credit", 0).apply(safe_float)
    df["balance"] = df.get("balance", None).apply(lambda x: safe_float(x) if x is not None else None)

    monthly_summary: List[dict] = []
    for period, group in df.groupby("month_period", sort=True):
        group_sorted = group.sort_values("date_parsed")

        balances = group_sorted["balance"].dropna()
        ending_balance = round(float(balances.iloc[-1]), 2) if not balances.empty else None
        lowest_balance = round(float(balances.min()), 2) if not balances.empty else None
        highest_balance = round(float(balances.max()), 2) if not balances.empty else None

        monthly_summary.append(
            {
                "month": period,
                "transaction_count": int(len(group_sorted)),
                "total_debit": round(float(group_sorted["debit"].sum()), 2),
                "total_credit": round(float(group_sorted["credit"].sum()), 2),
                "net_change": round(float(group_sorted["credit"].sum() - group_sorted["debit"].sum()), 2),
                "ending_balance": ending_balance,
                "lowest_balance": lowest_balance,
                "highest_balance": highest_balance,
                "source_files": ", ".join(sorted(set(group_sorted.get("source_file", []))))
                if "source_file" in group_sorted.columns
                else "",
            }
        )

    return sorted(monthly_summary, key=lambda x: x["month"])


# ---------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------
if st.session_state.results:
    st.subheader("📊 Extracted Transactions")

    df = pd.DataFrame(st.session_state.results)

    display_cols = ["date", "description", "debit", "credit", "balance", "page", "bank", "source_file"]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols] if display_cols else df

    st.dataframe(df_display, use_container_width=True)

    monthly_summary = calculate_monthly_summary(st.session_state.results)
    if monthly_summary:
        st.subheader("📅 Monthly Summary")
        summary_df = pd.DataFrame(monthly_summary)
        st.dataframe(summary_df, use_container_width=True)

        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Transactions", int(summary_df["transaction_count"].sum()))
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

    with col1:
        st.download_button(
            "📄 Download Transactions (JSON)",
            json.dumps(df_display.to_dict(orient="records"), indent=4),
            "transactions.json",
            "application/json",
        )

    with col2:
        # Robust date range
        date_min = df_display["date"].min() if "date" in df_display.columns and not df_display.empty else None
        date_max = df_display["date"].max() if "date" in df_display.columns and not df_display.empty else None

        full_report = {
            "summary": {
                "total_transactions": int(len(df_display)),
                "date_range": f"{date_min} to {date_max}" if date_min and date_max else None,
                "total_files_processed": int(df_display["source_file"].nunique())
                if "source_file" in df_display.columns
                else None,
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
