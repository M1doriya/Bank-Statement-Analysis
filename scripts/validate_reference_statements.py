from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd
import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from affin_bank import parse_affin_bank
from agro_bank import parse_agro_bank
from alliance import parse_transactions_alliance
from ambank import parse_ambank
from bank_islam import parse_bank_islam
from bank_muamalat import parse_transactions_bank_muamalat
from bank_rakyat import parse_bank_rakyat
from cimb import parse_transactions_cimb
from hong_leong import parse_hong_leong
from maybank import parse_transactions_maybank
from mbsb import parse_transactions_mbsb
from ocbc import parse_transactions_ocbc
from public_bank import parse_transactions_pbb
from rhb import parse_transactions_rhb
from uob import parse_transactions_uob

ROOT = Path("Bank-Statement")
REQUIRED_KEYS = {"date", "description", "debit", "credit", "balance", "bank", "source_file"}


def _with_pdfplumber(parser: Callable, path: Path, name: str) -> List[dict]:
    with pdfplumber.open(str(path)) as pdf:
        return parser(pdf, name)


PARSERS: Dict[str, Callable[[Path, str], List[dict]]] = {
    "AffinBank": lambda p, n: _with_pdfplumber(parse_affin_bank, p, n),
    "AgroBank": lambda p, n: _with_pdfplumber(parse_agro_bank, p, n),
    "Alliance": lambda p, n: _with_pdfplumber(parse_transactions_alliance, p, n),
    "Ambank": lambda p, n: _with_pdfplumber(parse_ambank, p, n),
    "BankIslam": lambda p, n: _with_pdfplumber(parse_bank_islam, p, n),
    "BankMuamalat": lambda p, n: _with_pdfplumber(parse_transactions_bank_muamalat, p, n),
    "BankRakyat": lambda p, n: _with_pdfplumber(parse_bank_rakyat, p, n),
    "CIMB": lambda p, n: _with_pdfplumber(parse_transactions_cimb, p, n),
    "HongLeong": lambda p, n: _with_pdfplumber(parse_hong_leong, p, n),
    "Maybank": lambda p, n: parse_transactions_maybank(p.read_bytes(), n),
    "MBSB": lambda p, n: _with_pdfplumber(parse_transactions_mbsb, p, n),
    "OCBC": lambda p, n: parse_transactions_ocbc(p.read_bytes(), n),
    "PublicBank": lambda p, n: _with_pdfplumber(parse_transactions_pbb, p, n),
    "RHB": lambda p, n: parse_transactions_rhb(p.read_bytes(), n),
    "UOB": lambda p, n: _with_pdfplumber(parse_transactions_uob, p, n),
}


def _run_bank(bank: str, parser: Callable[[Path, str], List[dict]], verbose: bool) -> str:
    folder = ROOT / bank
    if not folder.exists():
        return ""

    files = sorted(folder.rglob("*.pdf"))
    zero_files = 0
    total_tx = 0
    invalid_dates = 0
    parse_errors = 0
    missing_required_keys = 0
    both_debit_credit_positive = 0

    for idx, pdf in enumerate(files, start=1):
        start = time.time()
        try:
            tx = parser(pdf, pdf.name) or []
        except Exception as exc:  # noqa: BLE001
            parse_errors += 1
            tx = []
            print(f"ERROR,{bank},{pdf.relative_to(ROOT)},{type(exc).__name__}: {exc}", file=sys.stderr)

        elapsed = time.time() - start
        if verbose:
            print(f"INFO,{bank},{idx}/{len(files)},{pdf.name},tx={len(tx)},sec={elapsed:.2f}", file=sys.stderr)

        if not tx:
            zero_files += 1
        total_tx += len(tx)
        for row in tx:
            if not REQUIRED_KEYS.issubset(row.keys()):
                missing_required_keys += 1
            if pd.isna(pd.to_datetime(row.get("date"), errors="coerce")):
                invalid_dates += 1

            debit = float(row.get("debit") or 0.0)
            credit = float(row.get("credit") or 0.0)
            if debit > 0.0 and credit > 0.0:
                both_debit_credit_positive += 1

    return (
        f"{bank},{len(files)},{zero_files},{total_tx},{invalid_dates},{parse_errors},"
        f"{missing_required_keys},{both_debit_credit_positive}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all statement parsers against Bank-Statement reference PDFs.")
    parser.add_argument("--bank", choices=sorted(PARSERS.keys()), help="Run validation for a single bank only.")
    parser.add_argument("--verbose", action="store_true", help="Print per-file timing/status to stderr.")
    args = parser.parse_args()

    print(
        "bank,files,files_with_zero_tx,total_tx,invalid_dates,parse_errors,"
        "missing_required_keys,both_debit_credit_positive"
    )

    if args.bank:
        row = _run_bank(args.bank, PARSERS[args.bank], args.verbose)
        if row:
            print(row)
        return

    for bank, bank_parser in PARSERS.items():
        row = _run_bank(bank, bank_parser, args.verbose)
        if row:
            print(row)


if __name__ == "__main__":
    main()
