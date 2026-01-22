# affin_bank.py - OCR-based Affin Bank Parser
# Status: STABLE - Tested on 6 real PDFs (Apr-Sep 2025)
# 
# ARCHITECTURE:
# - Uses PyMuPDF (fitz) to extract page images from scanned PDFs
# - Uses Tesseract OCR with word-level coordinate extraction
# - Reconstructs transaction lines from word positions
# - Uses balance delta logic as fallback for incomplete OCR
#
# REQUIREMENTS:
# - pip install pymupdf pytesseract pillow
# - apt-get install tesseract-ocr
#
# STANDARD OUTPUT FORMAT (compatible with app.py):
# {date, description, debit, credit, balance, page, bank, source_file}

import re
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from datetime import datetime

def _read_pdf_bytes(pdf_input):
    """Read PDF bytes from various input types (Streamlit UploadedFile, path, bytes)"""
    if isinstance(pdf_input, (bytes, bytearray)):
        return bytes(pdf_input)
    if hasattr(pdf_input, "getvalue"):
        return pdf_input.getvalue()
    if hasattr(pdf_input, "read"):
        try:
            pdf_input.seek(0)
        except:
            pass
        return pdf_input.read()
    if isinstance(pdf_input, str):
        with open(pdf_input, "rb") as f:
            return f.read()
    raise ValueError("Unable to read PDF bytes")


def _ocr_page_words(page, dpi=200):
    """
    Extract text from scanned PDF page using word-level OCR.
    Groups words by Y coordinate to reconstruct lines properly.
    """
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    # Group words by Y coordinate (same line if within 15 pixels)
    line_groups = {}
    for i, text in enumerate(data['text']):
        if text.strip():
            y = data['top'][i]
            y_key = round(y / 15) * 15
            if y_key not in line_groups:
                line_groups[y_key] = []
            line_groups[y_key].append({'text': text, 'x': data['left'][i]})
    
    # Sort words within each line by X and join
    lines = []
    for y_key in sorted(line_groups.keys()):
        words = sorted(line_groups[y_key], key=lambda w: w['x'])
        lines.append(" ".join(w['text'] for w in words))
    return lines


def _parse_date(date_str):
    """Parse date from D/MM/YY or DD/MM/YY format to ISO YYYY-MM-DD"""
    for fmt in ["%d/%m/%y", "%d/%m/%Y"]:
        try:
            return datetime.strptime(date_str.strip(), fmt).strftime("%Y-%m-%d")
        except:
            pass
    return None


def _clean_amount(amount_str):
    """Clean and parse amount string to float, handling OCR artifacts"""
    if not amount_str:
        return 0.0
    s = str(amount_str).strip()
    # Handle common OCR artifacts for zero
    if s in ["-00", ".00", "0.00", "-0", ".0", ""]:
        return 0.0
    # Remove non-numeric characters except decimal point
    s = re.sub(r"[^\d.,]", "", s).replace(",", "")
    if not s or s == ".":
        return 0.0
    try:
        return float(s)
    except:
        return 0.0


def _extract_amounts(line):
    """Extract all currency amounts from a line"""
    return re.findall(r"[\d,]+\.\d{2}", line)


def parse_affin_bank(pdf_input, source_file=""):
    """
    Parse Affin Bank scanned PDF statements using OCR.
    
    These PDFs are image-based (scanned documents) requiring OCR extraction.
    Uses word-level OCR with coordinate reconstruction for accurate line parsing.
    
    Args:
        pdf_input: PDF file path, bytes, or Streamlit UploadedFile
        source_file: Original filename for tracking
    
    Returns:
        List of transaction dictionaries with standard format:
        {date, description, debit, credit, balance, page, bank, source_file}
    """
    transactions = []
    bank_name = "Affin Bank"
    
    try:
        pdf_bytes = _read_pdf_bytes(pdf_input)
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return []
    
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    
    # Date pattern at start of transaction line
    DATE_RE = re.compile(r"^(\d{1,2}/\d{2}/\d{2})")
    
    # Transaction type keywords detected in Affin Bank statements
    TX_TYPES = [
        "DUITNOW DEBIT", "JOMPAY DEBIT", "FPX DEBIT", "IBG DEBIT",
        "DUITNOW CREDIT", "INT/HIBAH/PROFIT"
    ]
    
    # Header/footer text to skip
    skip_keywords = [
        "AFFIN BANK", "PAGE", "ACCOUNT NO", "TARIKH", "STATEMENT",
        "BALANCE BROUGHT", "BALANCE FORWARD", "BAKI", "Description",
        "Huraian", "PROTECTED BY PIDM", "figures and balances",
        "angka angka", "akan dianggap", "discrepancy", "kelainan",
        "BRANCH:", "LOT 1123", "JALAN PANDAMAR", "KLANG",
        "ENGINEERING SDN", "pihak Bank", "TOTAL CREDIT", "TOTAL DEBIT",
        "CLOSING BALANCE", "OPENING BALANCE", "NO. AKAUN", "B/F"
    ]
    
    prev_balance = None
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        lines = _ocr_page_words(page, dpi=200)
        
        current_tx = None
        desc_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip header/footer lines
            if any(kw in line for kw in skip_keywords):
                continue
            
            # Check if line starts with date
            date_match = DATE_RE.match(line)
            if not date_match:
                # Continuation line - add to description
                if current_tx and len(line) > 2:
                    if not any(kw.lower() in line.lower() for kw in ["angka", "figures", "akan", "will be", "pihak"]):
                        desc_lines.append(line)
                continue
            
            # Save previous transaction before starting new one
            if current_tx:
                current_tx["description"] = " ".join([current_tx["description"]] + desc_lines).strip()
                transactions.append(current_tx)
                desc_lines = []
            
            date_str = date_match.group(1)
            date_iso = _parse_date(date_str)
            if not date_iso:
                current_tx = None
                continue
            
            # Find transaction type keyword
            tx_type = None
            for tt in TX_TYPES:
                if tt in line.upper():
                    tx_type = tt
                    break
            
            if not tx_type:
                current_tx = None
                continue
            
            # Extract all amounts from line
            amounts = _extract_amounts(line)
            
            if not amounts:
                current_tx = None
                continue
            
            # Parse based on transaction type
            if "DEBIT" in tx_type:
                # DEBIT: money going out
                if len(amounts) >= 2:
                    balance = _clean_amount(amounts[-1])
                    debit = _clean_amount(amounts[-2])
                    
                    # Verify with balance delta if possible
                    if prev_balance and debit == 0:
                        delta = prev_balance - balance
                        if delta > 0:
                            debit = round(delta, 2)
                    
                    current_tx = {
                        "date": date_iso,
                        "description": tx_type,
                        "debit": round(debit, 2),
                        "credit": 0.0,
                        "balance": round(balance, 2),
                        "page": page_num + 1,
                        "bank": bank_name,
                        "source_file": source_file
                    }
                    prev_balance = balance
                else:
                    # Single amount = balance only, calculate debit from delta
                    balance = _clean_amount(amounts[-1])
                    debit = round(prev_balance - balance, 2) if prev_balance else 0.0
                    
                    current_tx = {
                        "date": date_iso,
                        "description": tx_type,
                        "debit": round(max(debit, 0), 2),
                        "credit": 0.0,
                        "balance": round(balance, 2),
                        "page": page_num + 1,
                        "bank": bank_name,
                        "source_file": source_file
                    }
                    prev_balance = balance
            
            elif "CREDIT" in tx_type or "INT/HIBAH" in tx_type:
                # CREDIT: money coming in
                if len(amounts) >= 2:
                    balance = _clean_amount(amounts[-1])
                    
                    # If 3+ amounts, credit is second-to-last
                    if len(amounts) >= 3:
                        credit = _clean_amount(amounts[-2])
                    else:
                        # Calculate from balance delta
                        credit = round(balance - prev_balance, 2) if prev_balance else 0.0
                    
                    current_tx = {
                        "date": date_iso,
                        "description": tx_type,
                        "debit": 0.0,
                        "credit": round(max(credit, 0), 2),
                        "balance": round(balance, 2),
                        "page": page_num + 1,
                        "bank": bank_name,
                        "source_file": source_file
                    }
                    prev_balance = balance
                else:
                    # Single amount = balance only
                    balance = _clean_amount(amounts[-1])
                    credit = round(balance - prev_balance, 2) if prev_balance else 0.0
                    
                    current_tx = {
                        "date": date_iso,
                        "description": tx_type,
                        "debit": 0.0,
                        "credit": round(max(credit, 0), 2),
                        "balance": round(balance, 2),
                        "page": page_num + 1,
                        "bank": bank_name,
                        "source_file": source_file
                    }
                    prev_balance = balance
        
        # Save last transaction of page
        if current_tx:
            current_tx["description"] = " ".join([current_tx["description"]] + desc_lines).strip()
            transactions.append(current_tx)
            current_tx = None
            desc_lines = []
    
    doc.close()
    return transactions


# ============================================
# TESTING
# ============================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        tx = parse_affin_bank(pdf_path, pdf_path)
        print(f"Found {len(tx)} transactions")
        
        total_debit = sum(t["debit"] for t in tx)
        total_credit = sum(t["credit"] for t in tx)
        
        print(f"\nTotal Debit: RM {total_debit:,.2f}")
        print(f"Total Credit: RM {total_credit:,.2f}")
        print(f"Net: RM {total_credit - total_debit:,.2f}")
        
        print("\nSample transactions:")
        for t in tx[:5]:
            print(f"  {t['date']} | DR:{t['debit']:>10.2f} | CR:{t['credit']:>10.2f} | {t['description'][:30]}")
