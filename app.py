import streamlit as st
import pdfplumber
import json
import pandas as pd
from datetime import datetime
from io import BytesIO
import streamlit.components.v1 as components

# Existing parsers
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

# v4 engine + renderer
from feature_engine_v4 import build_v4_context
from report_renderer_v4 import render_v4_html


st.set_page_config(page_title="Bank Statement Parser + v4 Report", layout="wide")
st.title("📄 Bank Statement Parser + v4 HTML Credit Report")

# -----------------------------
# Session state
# -----------------------------
if "status" not in st.session_state:
    st.session_state.status = "idle"
if "results" not in st.session_state:
    st.session_state.results = []

# -----------------------------
# Sidebar inputs
# -----------------------------
with st.sidebar:
    st.header("Report Inputs")

    company_name = st.text_input("Company Name (for report)", value="Company")

    od_limit_raw = st.text_input("Approved OD Limit (RM) - optional", value="")
    declared_sales_raw = st.text_input("Declared Sales (RM) - optional", value="")
    related_raw = st.text_area("Related Party Keywords (comma-separated)", value="")

    def parse_optional_float(s: str):
        s = (s or "").strip()
        if not s:
            return None
        try:
            return float(s.replace(",", ""))
        except Exception:
            return None

    od_limit = parse_optional_float(od_limit_raw)
    declared_sales = parse_optional_float(declared_sales_raw)

    related_party_keywords = [x.strip() for x in related_raw.split(",") if x.strip()]

    if od_limit_raw.strip() and od_limit is None:
        st.warning("OD Limit is not a valid number. Leave empty if not applicable.")
    if declared_sales_raw.strip() and declared_sales is None:
        st.warning("Declared Sales is not a valid number. Leave empty if not applicable.")

# -----------------------------
# Bank selection
# -----------------------------
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
    ],
)

# -----------------------------
# File upload
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)
if uploaded_files:
    uploaded_files = sorted(uploaded_files, key=lambda x: x.name)

# -----------------------------
# Controls
# -----------------------------
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

# -----------------------------
# Routing: FIX Maybank error
# -----------------------------
def parse_by_bank(bank: str, uploaded_file, pdf_bytes: bytes, filename: str):
    """
    - Maybank uses fitz: MUST pass bytes (or UploadedFile), not pdfplumber.PDF
    - RHB in your repo accepts bytes (it uses BytesIO internally)
    - Others use pdfplumber: pass pdfplumber.PDF object
    """
    if bank == "Maybank":
        return parse_transactions_maybank(pdf_bytes, filename)

    if bank == "RHB Bank":
        return parse_transactions_rhb(pdf_bytes, filename)

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


# -----------------------------
# Main processing
# -----------------------------
all_tx = []

if uploaded_files and st.session_state.status == "running":
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)

    for idx, f in enumerate(uploaded_files):
        if st.session_state.status == "stopped":
            st.warning("⏹️ Processing stopped by user.")
            break

        st.write(f"### 🗂️ Processing File: **{f.name}**")
        try:
            pdf_bytes = f.getvalue()
            tx = parse_by_bank(bank_choice, f, pdf_bytes, f.name)

            if tx:
                st.success(f"✅ Extracted {len(tx)} transactions from {f.name}")
                all_tx.extend(tx)
            else:
                st.warning(f"⚠️ No transactions found in {f.name}")

        except Exception as e:
            st.error(f"❌ Error processing {f.name}: {e}")

        progress_bar.progress((idx + 1) / total_files)

    st.session_state.results = all_tx


# -----------------------------
# Monthly summary (calendar month) – your original
# -----------------------------
def calculate_monthly_summary(transactions):
    if not transactions:
        return []
    df = pd.DataFrame(transactions)
    df["date_parsed"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date_parsed"])
    if df.empty:
        return []
    df["month_period"] = df["date_parsed"].dt.strftime("%Y-%m")
    df["debit"] = pd.to_numeric(df["debit"], errors="coerce").fillna(0)
    df["credit"] = pd.to_numeric(df["credit"], errors="coerce").fillna(0)
    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")

    out = []
    for period, g in df.groupby("month_period", sort=True):
        ending_balance = None
        if not g["balance"].isna().all():
            g2 = g.sort_values("date_parsed")
            b = g2["balance"].dropna()
            if not b.empty:
                ending_balance = round(float(b.iloc[-1]), 2)

        out.append({
            "month": period,
            "transaction_count": int(len(g)),
            "total_debit": round(float(g["debit"].sum()), 2),
            "total_credit": round(float(g["credit"].sum()), 2),
            "net_change": round(float(g["credit"].sum() - g["debit"].sum()), 2),
            "ending_balance": ending_balance,
            "lowest_balance": round(float(g["balance"].min()), 2) if not g["balance"].isna().all() else None,
            "highest_balance": round(float(g["balance"].max()), 2) if not g["balance"].isna().all() else None,
            "source_files": ", ".join(sorted(g["source_file"].unique())) if "source_file" in g.columns else ""
        })

    return sorted(out, key=lambda x: x["month"])


# -----------------------------
# Display + downloads
# -----------------------------
if st.session_state.results:
    df = pd.DataFrame(st.session_state.results)

    cols = ["date", "description", "debit", "credit", "balance", "page", "bank", "source_file"]
    cols = [c for c in cols if c in df.columns]
    df_display = df[cols].copy()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Transactions",
        "Calendar Monthly Summary",
        "v4 HTML Report",
        "Downloads"
    ])

    with tab1:
        st.subheader("📊 Extracted Transactions")
        st.dataframe(df_display, use_container_width=True)

    with tab2:
        st.subheader("📅 Monthly Summary (Calendar Month)")
        ms = calculate_monthly_summary(st.session_state.results)
        if ms:
            st.dataframe(pd.DataFrame(ms), use_container_width=True)
        else:
            st.info("No valid dates to compute calendar monthly summary.")

    with tab3:
        st.subheader("🧾 v4 Interactive HTML Report")
        try:
            ctx = build_v4_context(
                transactions=st.session_state.results,
                company_name=company_name,
                od_limit=od_limit,
                declared_sales=declared_sales,
                related_party_keywords=related_party_keywords,
            )
            html = render_v4_html(ctx, template_path="templates/BankAnalysis_v4.html")

            components.html(html, height=900, scrolling=True)

            st.download_button(
                "⬇️ Download v4 HTML Report",
                html,
                file_name=f"{company_name}_BankAnalysis_v4.html",
                mime="text/html",
            )

            st.download_button(
                "⬇️ Download GEM_5_BANK_DATA.json",
                json.dumps(ctx.get("gem_5_bank_data", {}), indent=2),
                file_name="GEM_5_BANK_DATA.json",
                mime="application/json",
            )
        except Exception as e:
            st.error(f"v4 report generation failed: {e}")
            st.info("Confirm you added: feature_engine_v4.py, report_renderer_v4.py, templates/BankAnalysis_v4.html, and jinja2 in requirements.")

    with tab4:
        st.subheader("⬇️ Download Options")
        ms = calculate_monthly_summary(st.session_state.results)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "📄 Download Transactions (JSON)",
                json.dumps(df_display.to_dict(orient="records"), indent=2),
                "transactions.json",
                "application/json"
            )

        with c2:
            full_report = {
                "summary": {
                    "bank_selected": bank_choice,
                    "total_transactions": int(len(df_display)),
                    "date_range": f"{df_display['date'].min()} to {df_display['date'].max()}" if "date" in df_display.columns else None,
                    "total_files_processed": int(df_display["source_file"].nunique()) if "source_file" in df_display.columns else None,
                    "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
                "monthly_summary": ms,
                "transactions": df_display.to_dict(orient="records"),
            }
            st.download_button(
                "📊 Download Full Report (JSON)",
                json.dumps(full_report, indent=2),
                "full_report.json",
                "application/json"
            )

        with c3:
            output = BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                df_display.to_excel(writer, sheet_name="Transactions", index=False)
                if ms:
                    pd.DataFrame(ms).to_excel(writer, sheet_name="Monthly Summary", index=False)

            st.download_button(
                "📊 Download Full Report (XLSX)",
                output.getvalue(),
                "full_report.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
else:
    if uploaded_files:
        st.warning("⚠️ No transactions found — click **Start Processing**.")
