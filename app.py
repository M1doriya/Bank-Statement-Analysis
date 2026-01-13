import streamlit as st
import pdfplumber
import json
import pandas as pd
import re
from datetime import datetime
from io import BytesIO

# ---------------------------------------------------
# Import standalone parsers (EXISTING)
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
# DATE PARSING / NORMALIZATION (FIX)
# ---------------------------------------------------
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
DMY_SLASH_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{2,4}$")
DMY_DASH_RE = re.compile(r"^\d{1,2}-\d{1,2}-\d{2,4}$")

def parse_any_date(value) -> pd.Timestamp:
    """
    Robust date parser:
    - ISO: YYYY-MM-DD -> parse with explicit format (prevents day/month flips)
    - DMY: DD/MM/YYYY or DD-MM-YYYY -> parse dayfirst=True
    - Fallback: pandas best-effort
    """
    if value is None:
        return pd.NaT

    s = str(value).strip()
    if not s:
        return pd.NaT

    if ISO_RE.match(s):
        return pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")

    if DMY_SLASH_RE.match(s) or DMY_DASH_RE.match(s):
        # Day-first for common Malaysian statement formats
        return pd.to_datetime(s, dayfirst=True, errors="coerce")

    # Fallback: try pandas inference (do NOT force dayfirst here)
    return pd.to_datetime(s, errors="coerce")

def normalize_date_to_iso(value) -> str | None:
    ts = parse_any_date(value)
    if pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%d")

def normalize_transactions_dates(tx_list):
    """
    Normalizes tx['date'] into YYYY-MM-DD when possible.
    Leaves as-is if missing/unparseable (summary will drop NaT dates).
    """
    if not tx_list:
        return tx_list

    for tx in tx_list:
        if isinstance(tx, dict) and "date" in tx:
            iso = normalize_date_to_iso(tx.get("date"))
            if iso:
                tx["date"] = iso
    return tx_list

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
    st.session_state.status = "idle"    # idle, running, stopped

if "results" not in st.session_state:
    st.session_state.results = []

# ---------------------------------------------------
# Bank Selection
# ---------------------------------------------------
bank_choice = st.selectbox(
    "Select Bank Format",
    [
        "Affin Bank",
        "Agro Bank",
        "Ambank",
        "Bank Islam",
        "Bank Muamalat",
        "Bank Rakyat",
        "CIMB Bank",
        "Hong Leong",
        "Maybank",
        "Public Bank (PBB)",
        "RHB Bank"
    ]
)

# ---------------------------------------------------
# File Upload
# ---------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# Sort uploaded files by name
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
all_tx = []

if uploaded_files and st.session_state.status == "running":

    bank_display_box = st.empty()
    progress_bar = st.progress(0)

    total_files = len(uploaded_files)

    for file_idx, uploaded_file in enumerate(uploaded_files):

        if st.session_state.status == "stopped":
            st.warning("⏹️ Processing stopped by user.")
            break

        st.write(f"### 🗂️ Processing File: **{uploaded_file.name}**")
        bank_display_box.info(f"📄 Processing {bank_choice}: {uploaded_file.name}...")

        try:
            with pdfplumber.open(uploaded_file) as pdf:

                tx = []

                if bank_choice == "Maybank":
                    tx = parse_transactions_maybank(pdf, uploaded_file.name)

                elif bank_choice == "Public Bank (PBB)":
                    tx = parse_transactions_pbb(pdf, uploaded_file.name)

                elif bank_choice == "RHB Bank":
                    tx = parse_transactions_rhb(uploaded_file, uploaded_file.name)

                elif bank_choice == "CIMB Bank":
                    tx = parse_transactions_cimb(pdf, uploaded_file.name)

                elif bank_choice == "Ambank":
                    tx = parse_ambank(pdf, uploaded_file.name)

                elif bank_choice == "Bank Islam":
                    tx = parse_bank_islam(pdf, uploaded_file.name)

                elif bank_choice == "Bank Rakyat":
                    tx = parse_bank_rakyat(pdf, uploaded_file.name)

                elif bank_choice == "Bank Muamalat":
                    tx = parse_transactions_bank_muamalat(pdf, uploaded_file.name)

                elif bank_choice == "Agro Bank":
                    tx = parse_agro_bank(pdf, uploaded_file.name)

                elif bank_choice == "Hong Leong":
                    tx = parse_hong_leong(pdf, uploaded_file.name)

                elif bank_choice == "Affin Bank":
                    tx = parse_affin_bank(pdf, uploaded_file.name)

                # ---- FIX: normalize dates BEFORE adding to all_tx ----
                tx = normalize_transactions_dates(tx)

                if tx:
                    st.success(f"✅ Extracted {len(tx)} transactions from {uploaded_file.name}")
                    all_tx.extend(tx)
                else:
                    st.warning(f"⚠️ No transactions found in {uploaded_file.name}")

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {e}")

        progress = (file_idx + 1) / total_files
        progress_bar.progress(progress)

    bank_display_box.success(f"🏦 Completed processing: **{bank_choice}**")
    st.session_state.results = all_tx

# ---------------------------------------------------
# CALCULATE MONTHLY SUMMARY (FIX)
# ---------------------------------------------------
def calculate_monthly_summary(transactions):
    if not transactions:
        return []

    df = pd.DataFrame(transactions)

    # Robust parsing for mixed formats
    df["date_parsed"] = df["date"].apply(parse_any_date) if "date" in df.columns else pd.NaT
    df = df.dropna(subset=["date_parsed"])

    if df.empty:
        st.warning("⚠️ No valid transaction dates found.")
        return []

    df["month_period"] = df["date_parsed"].dt.strftime("%Y-%m")

    df["debit"] = pd.to_numeric(df.get("debit", 0), errors="coerce").fillna(0)
    df["credit"] = pd.to_numeric(df.get("credit", 0), errors="coerce").fillna(0)
    df["balance"] = pd.to_numeric(df.get("balance", None), errors="coerce")

    monthly_summary = []

    for period, group in df.groupby("month_period", sort=True):
        # Sort within month so ending_balance is deterministic
        group_sorted = group.sort_values(["date_parsed", "page"], na_position="last")

        ending_balance = None
        balances = group_sorted["balance"].dropna()
        if not balances.empty:
            ending_balance = round(balances.iloc[-1], 2)

        monthly_summary.append({
            "month": period,
            "transaction_count": int(len(group_sorted)),
            "total_debit": round(float(group_sorted["debit"].sum()), 2),
            "total_credit": round(float(group_sorted["credit"].sum()), 2),
            "net_change": round(float(group_sorted["credit"].sum() - group_sorted["debit"].sum()), 2),
            "ending_balance": ending_balance,
            "lowest_balance": round(float(group_sorted["balance"].min()), 2) if not group_sorted["balance"].isna().all() else None,
            "highest_balance": round(float(group_sorted["balance"].max()), 2) if not group_sorted["balance"].isna().all() else None,
            "source_files": ", ".join(sorted(group_sorted["source_file"].dropna().unique()))
                           if "source_file" in group_sorted.columns else ""
        })

    return sorted(monthly_summary, key=lambda x: x["month"])

# ---------------------------------------------------
# DISPLAY RESULTS
# ---------------------------------------------------
if st.session_state.results:

    st.subheader("📊 Extracted Transactions")

    df = pd.DataFrame(st.session_state.results)

    display_cols = [
        "date", "description", "debit", "credit",
        "balance", "page", "bank", "source_file"
    ]
    display_cols = [c for c in display_cols if c in df.columns]

    df_display = df[display_cols].copy()

    # Ensure dates are normalized for export/display consistency
    if "date" in df_display.columns:
        df_display["date"] = df_display["date"].apply(lambda x: normalize_date_to_iso(x) or x)

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
            net_total = summary_df["net_change"].sum()
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
            "application/json"
        )

    with col2:
        # Better date_range using parsed dates
        date_parsed = df_display["date"].apply(parse_any_date) if "date" in df_display.columns else pd.Series([], dtype="datetime64[ns]")
        date_parsed = date_parsed.dropna()
        if not date_parsed.empty:
            date_range = f"{date_parsed.min().date()} to {date_parsed.max().date()}"
        else:
            date_range = f"{df_display['date'].min()} to {df_display['date'].max()}" if "date" in df_display.columns else ""

        full_report = {
            "summary": {
                "total_transactions": int(len(df_display)),
                "date_range": date_range,
                "total_files_processed": int(df_display["source_file"].nunique()) if "source_file" in df_display.columns else 0
            },
            "monthly_summary": monthly_summary,
            "transactions": df_display.to_dict(orient="records")
        }
        st.download_button(
            "📊 Download Full Report (JSON)",
            json.dumps(full_report, indent=4),
            "full_report.json",
            "application/json"
        )

    with col3:
        output = BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df_display.to_excel(writer, sheet_name="Transactions", index=False)
            if monthly_summary:
                pd.DataFrame(monthly_summary).to_excel(
                    writer, sheet_name="Monthly Summary", index=False
                )

        st.download_button(
            "📊 Download Full Report (XLSX)",
            output.getvalue(),
            "full_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

else:
    if uploaded_files:
        st.warning("⚠️ No transactions found — click **Start Processing**.")
