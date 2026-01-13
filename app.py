import streamlit as st
import pdfplumber
import json
import pandas as pd
from datetime import datetime
from io import BytesIO
import streamlit.components.v1 as components

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
# NEW: Deterministic analysis + HTML report
# (Make sure these files exist in your repo)
# ---------------------------------------------------
try:
    from analysis_engine import build_analysis, AnalysisInputs
    from report_renderer import render_report_html
    ANALYSIS_ENABLED = True
except Exception:
    ANALYSIS_ENABLED = False


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
# Sidebar: Optional Analysis Inputs
# ---------------------------------------------------
with st.sidebar:
    st.header("Analysis Inputs (Optional)")

    od_limit_raw = st.text_input("Approved OD Limit (RM)", value="")
    declared_sales_raw = st.text_input("Declared Sales (RM)", value="")

    def _parse_optional_float(s: str):
        s = (s or "").strip()
        if not s:
            return None
        try:
            return float(s.replace(",", ""))
        except Exception:
            return None

    od_limit = _parse_optional_float(od_limit_raw)
    declared_sales = _parse_optional_float(declared_sales_raw)

    if od_limit_raw.strip() and od_limit is None:
        st.warning("OD Limit is not a valid number. Leave empty if not applicable.")
    if declared_sales_raw.strip() and declared_sales is None:
        st.warning("Declared Sales is not a valid number. Leave empty if not applicable.")

    if not ANALYSIS_ENABLED:
        st.info("Credit Analysis Report is disabled. Add analysis_engine.py, report_renderer.py, and templates/BankAnalysis_template.html.")


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
        "RHB Bank",
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
# Helper: parse one PDF by bank (FIXES YOUR ERROR)
# ---------------------------------------------------
def parse_by_bank(bank: str, uploaded_file, pdf_bytes: bytes, filename: str):
    """
    Routes PDF inputs correctly:
      - Maybank: pass bytes to fitz-based parser
      - RHB: pass bytes (rhb.py reads bytes safely)
      - Others: open pdfplumber with BytesIO and pass pdfplumber.PDF
    """
    if bank == "Maybank":
        # ✅ maybank.py uses fitz.open(stream=..., filetype="pdf")
        return parse_transactions_maybank(pdf_bytes, filename)

    if bank == "RHB Bank":
        # ✅ rhb.py reads bytes internally and uses pdfplumber on BytesIO
        return parse_transactions_rhb(pdf_bytes, filename)

    # ✅ all other banks remain pdfplumber-based
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        if bank == "Public Bank (PBB)":
            return parse_transactions_pbb(pdf, filename)

        if bank == "CIMB Bank":
            return parse_transactions_cimb(pdf, filename)

        if bank == "Ambank":
            return parse_ambank(pdf, filename)

        if bank == "Bank Islam":
            return parse_bank_islam(pdf, filename)

        if bank == "Bank Rakyat":
            return parse_bank_rakyat(pdf, filename)

        if bank == "Bank Muamalat":
            return parse_transactions_bank_muamalat(pdf, filename)

        if bank == "Agro Bank":
            return parse_agro_bank(pdf, filename)

        if bank == "Hong Leong":
            return parse_hong_leong(pdf, filename)

        if bank == "Affin Bank":
            return parse_affin_bank(pdf, filename)

    return []


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
            pdf_bytes = uploaded_file.getvalue()
            tx = parse_by_bank(bank_choice, uploaded_file, pdf_bytes, uploaded_file.name)

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
# CALCULATE MONTHLY SUMMARY (Calendar months - your existing logic)
# ---------------------------------------------------
def calculate_monthly_summary(transactions):
    if not transactions:
        return []

    df = pd.DataFrame(transactions)

    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_parsed"])

    if df.empty:
        st.warning("⚠️ No valid transaction dates found.")
        return []

    df["month_period"] = df["date_parsed"].dt.strftime("%Y-%m")

    df["debit"] = pd.to_numeric(df["debit"], errors="coerce").fillna(0)
    df["credit"] = pd.to_numeric(df["credit"], errors="coerce").fillna(0)
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")

    monthly_summary = []

    for period, group in df.groupby("month_period", sort=True):

        ending_balance = None
        if not group["balance"].isna().all():
            group_sorted = group.sort_values("date_parsed")
            balances = group_sorted["balance"].dropna()
            if not balances.empty:
                ending_balance = round(float(balances.iloc[-1]), 2)

        monthly_summary.append({
            "month": period,
            "transaction_count": int(len(group)),
            "total_debit": round(float(group["debit"].sum()), 2),
            "total_credit": round(float(group["credit"].sum()), 2),
            "net_change": round(float(group["credit"].sum() - group["debit"].sum()), 2),
            "ending_balance": ending_balance,
            "lowest_balance": round(float(group["balance"].min()), 2) if not group["balance"].isna().all() else None,
            "highest_balance": round(float(group["balance"].max()), 2) if not group["balance"].isna().all() else None,
            "source_files": ", ".join(sorted(group["source_file"].unique())) if "source_file" in group.columns else ""
        })

    return sorted(monthly_summary, key=lambda x: x["month"])


# ---------------------------------------------------
# DISPLAY RESULTS + TABS
# ---------------------------------------------------
if st.session_state.results:

    df = pd.DataFrame(st.session_state.results)

    display_cols = [
        "date", "description", "debit", "credit",
        "balance", "page", "bank", "source_file"
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    df_display = df[display_cols].copy()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Transactions",
        "Calendar Monthly Summary",
        "Credit Analysis Report",
        "Downloads"
    ])

    # ------------------- TAB 1: TRANSACTIONS -------------------
    with tab1:
        st.subheader("📊 Extracted Transactions")
        st.dataframe(df_display, use_container_width=True)

    # ------------------- TAB 2: CALENDAR MONTHLY SUMMARY -------------------
    with tab2:
        monthly_summary = calculate_monthly_summary(st.session_state.results)
        if monthly_summary:
            st.subheader("📅 Monthly Summary (Calendar Month)")
            summary_df = pd.DataFrame(monthly_summary)
            st.dataframe(summary_df, use_container_width=True)

            st.markdown("---")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Total Transactions", int(summary_df["transaction_count"].sum()))
            with c2:
                st.metric("Total Debits", f"RM {summary_df['total_debit'].sum():,.2f}")
            with c3:
                st.metric("Total Credits", f"RM {summary_df['total_credit'].sum():,.2f}")
            with c4:
                st.metric("Net Change", f"RM {summary_df['net_change'].sum():,.2f}")
        else:
            st.info("No calendar-month summary available.")

    # ------------------- TAB 3: CREDIT ANALYSIS REPORT -------------------
    with tab3:
        st.subheader("🧾 Credit Analysis Outputs (Deterministic)")

        if not ANALYSIS_ENABLED:
            st.error("analysis_engine.py / report_renderer.py / templates/BankAnalysis_template.html not found. Add them to enable this tab.")
        else:
            inputs = AnalysisInputs(od_limit=od_limit, declared_sales=declared_sales)

            try:
                analysis = build_analysis(st.session_state.results, inputs)

                st.write("### OUTPUT 1: GEM_5_BANK_DATA (JSON)")
                gem_json_str = json.dumps(analysis.gem_5_bank_data, indent=2)
                st.code(gem_json_str, language="json")

                st.write("### OUTPUT 2: HTML Report")
                html = render_report_html(
                    analysis=analysis,
                    bank_name=bank_choice,
                    inputs=inputs,
                    template_path="templates/BankAnalysis_template.html",
                )

                components.html(html, height=800, scrolling=True)

                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(
                        "⬇️ Download GEM_5_BANK_DATA.json",
                        gem_json_str,
                        "GEM_5_BANK_DATA.json",
                        "application/json",
                    )
                with c2:
                    st.download_button(
                        "⬇️ Download BankAnalysis_Report.html",
                        html,
                        "BankAnalysis_Report.html",
                        "text/html",
                    )

                if analysis.warnings:
                    st.warning("Data QA Warnings:")
                    for w in analysis.warnings:
                        st.write(f"- {w}")

            except Exception as e:
                st.error(f"Analysis/Report generation failed: {e}")

    # ------------------- TAB 4: DOWNLOADS (Your original exports) -------------------
    with tab4:
        st.subheader("⬇️ Download Options")

        monthly_summary = calculate_monthly_summary(st.session_state.results)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                "📄 Download Transactions (JSON)",
                json.dumps(df_display.to_dict(orient="records"), indent=4),
                "transactions.json",
                "application/json"
            )

        with col2:
            full_report = {
                "summary": {
                    "total_transactions": int(len(df_display)),
                    "date_range": f"{df_display['date'].min()} to {df_display['date'].max()}" if "date" in df_display.columns else None,
                    "total_files_processed": int(df_display["source_file"].nunique()) if "source_file" in df_display.columns else None,
                    "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "bank_selected": bank_choice,
                    "inputs": {
                        "od_limit": od_limit,
                        "declared_sales": declared_sales
                    }
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
                    pd.DataFrame(monthly_summary).to_excel(writer, sheet_name="Monthly Summary", index=False)

            st.download_button(
                "📊 Download Full Report (XLSX)",
                output.getvalue(),
                "full_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

else:
    if uploaded_files:
        st.warning("⚠️ No transactions found — click **Start Processing**.")
