import html
import inspect
from typing import List, Tuple

import streamlit as st

def inject_global_styles(theme_mode: str = "Dark") -> None:
    is_light = str(theme_mode or "Dark").strip().lower() == "light"
    if is_light:
        theme_vars = """
            --page-bg: #f4f7fb;
            --page-bg-soft: #eef3f8;
            --page-spotlight: rgba(31, 122, 140, 0.10);
            --surface: rgba(255, 255, 255, 0.92);
            --surface-soft: rgba(255, 255, 255, 0.72);
            --surface-elevated: #ffffff;
            --panel: rgba(255, 255, 255, 0.88);
            --panel-soft: rgba(247, 250, 252, 0.86);
            --text: #233243;
            --text-strong: #0f172a;
            --muted: #66758a;
            --line: rgba(15, 23, 42, 0.08);
            --line-strong: rgba(15, 23, 42, 0.14);
            --accent: #0f8f8b;
            --accent-strong: #0a6f71;
            --accent-soft: rgba(15, 143, 139, 0.10);
            --hero-bg: linear-gradient(135deg, rgba(255,255,255,0.94) 0%, rgba(246,250,252,0.92) 100%);
            --hero-line: rgba(15, 23, 42, 0.08);
            --hero-text: #0f172a;
            --hero-muted: #5f6f84;
            --topbar-bg: rgba(255, 255, 255, 0.82);
            --topbar-border: rgba(15, 23, 42, 0.08);
            --topbar-text: #0f172a;
            --topbar-muted: #607086;
            --topbar-active: #0f172a;
            --progress-bg: rgba(255, 255, 255, 0.84);
            --progress-border: rgba(15, 23, 42, 0.08);
            --progress-title: #0f172a;
            --progress-copy: #617185;
            --progress-subtle: #142032;
            --progress-pill-bg: rgba(15, 143, 139, 0.08);
            --progress-pill-text: #0a6f71;
            --tool-bg: rgba(255, 255, 255, 0.84);
            --tool-border: rgba(15, 23, 42, 0.08);
            --tool-card-bg: rgba(255,255,255,0.72);
            --tool-card-border: rgba(15, 23, 42, 0.08);
            --tool-title: #0f172a;
            --tool-copy: #617185;
            --tool-icon-bg: rgba(15, 143, 139, 0.10);
            --tool-icon-border: rgba(15, 143, 139, 0.16);
            --tool-icon-text: #0a6f71;
            --tool-input-bg: rgba(255,255,255,0.96);
            --tool-input-border: rgba(15, 23, 42, 0.10);
            --tool-input-text: #0f172a;
            --tool-placeholder: #8a97a8;
            --tool-button-bg: rgba(255,255,255,0.98);
            --tool-button-text: #132031;
            --tool-button-border: rgba(15, 23, 42, 0.10);
            --tool-button-hover-bg: #f6f9fc;
            --tool-primary-bg: linear-gradient(135deg, #0f8f8b 0%, #0b6f78 100%);
            --tool-primary-text: #ffffff;
            --tool-uploader-shell-bg: rgba(255,255,255,0.92);
            --tool-uploader-shell-border: rgba(15, 23, 42, 0.08);
            --tool-uploader-copy: #617185;
            --shadow: 0 24px 60px rgba(15, 23, 42, 0.10);
            --shadow-soft: 0 12px 28px rgba(15, 23, 42, 0.06);
            --badge-bg: rgba(15, 143, 139, 0.08);
            --badge-border: rgba(15, 143, 139, 0.14);
            --badge-text: #0a6f71;
            --display-heading: #0f172a;
            --display-copy: #617185;
            --auth-bg: rgba(255, 255, 255, 0.86);
            --auth-heading: #0f172a;
            --auth-copy: #617185;
            --input-bg: rgba(255,255,255,0.95);
            --input-border: rgba(15, 23, 42, 0.10);
            --input-text: #0f172a;
            --placeholder: #8a97a8;
            --form-label: #172438;
            --status-idle-bg: rgba(148, 163, 184, 0.12);
            --status-idle-text: #5f6c80;
            --status-running-bg: rgba(15, 143, 139, 0.12);
            --status-running-text: #0a6f71;
            --status-stopped-bg: rgba(202, 138, 4, 0.12);
            --status-stopped-text: #9a6700;
            --table-bg: rgba(255,255,255,0.94);
            --table-head: #f7fafc;
            --table-text: #132031;
            --select-menu-bg: rgba(255,255,255,0.98);
            --select-menu-surface: #ffffff;
            --select-menu-row-bg: #ffffff;
            --select-menu-text: #132031;
            --select-menu-border: rgba(15, 23, 42, 0.10);
            --select-menu-hover-bg: rgba(15, 143, 139, 0.10);
            --select-menu-hover-text: #0a6f71;
            --select-menu-shadow: 0 22px 40px rgba(15, 23, 42, 0.10);
        """
    else:
        theme_vars = """
            --page-bg: #081118;
            --page-bg-soft: #0d1620;
            --page-spotlight: rgba(78, 168, 192, 0.16);
            --surface: rgba(15, 23, 33, 0.84);
            --surface-soft: rgba(17, 27, 39, 0.66);
            --surface-elevated: #131d2a;
            --panel: rgba(14, 22, 31, 0.86);
            --panel-soft: rgba(18, 28, 39, 0.78);
            --text: #d3deea;
            --text-strong: #f7fbff;
            --muted: #93a4b6;
            --line: rgba(148, 163, 184, 0.14);
            --line-strong: rgba(148, 163, 184, 0.20);
            --accent: #5bb8c9;
            --accent-strong: #8ad8e3;
            --accent-soft: rgba(91, 184, 201, 0.14);
            --hero-bg: linear-gradient(135deg, rgba(15,24,34,0.92) 0%, rgba(9,16,23,0.88) 100%);
            --hero-line: rgba(148, 163, 184, 0.14);
            --hero-text: #f7fbff;
            --hero-muted: #a2b2c3;
            --topbar-bg: rgba(12, 19, 28, 0.80);
            --topbar-border: rgba(148, 163, 184, 0.14);
            --topbar-text: #f5f9ff;
            --topbar-muted: #9aabbe;
            --topbar-active: #ffffff;
            --progress-bg: rgba(13, 21, 30, 0.82);
            --progress-border: rgba(148, 163, 184, 0.14);
            --progress-title: #f7fbff;
            --progress-copy: #97a8ba;
            --progress-subtle: #eaf2fb;
            --progress-pill-bg: rgba(91, 184, 201, 0.12);
            --progress-pill-text: #a8ebf5;
            --tool-bg: rgba(13, 21, 30, 0.82);
            --tool-border: rgba(148, 163, 184, 0.14);
            --tool-card-bg: rgba(255,255,255,0.03);
            --tool-card-border: rgba(148, 163, 184, 0.14);
            --tool-title: #f6fbff;
            --tool-copy: #97a8ba;
            --tool-icon-bg: rgba(91, 184, 201, 0.12);
            --tool-icon-border: rgba(91, 184, 201, 0.20);
            --tool-icon-text: #9de3ee;
            --tool-input-bg: rgba(255,255,255,0.04);
            --tool-input-border: rgba(148, 163, 184, 0.14);
            --tool-input-text: #f7fbff;
            --tool-placeholder: #8093a7;
            --tool-button-bg: rgba(255,255,255,0.04);
            --tool-button-text: #e8f0f8;
            --tool-button-border: rgba(148, 163, 184, 0.14);
            --tool-button-hover-bg: rgba(255,255,255,0.07);
            --tool-primary-bg: linear-gradient(135deg, #5bb8c9 0%, #2b8fa5 100%);
            --tool-primary-text: #051017;
            --tool-uploader-shell-bg: rgba(255,255,255,0.02);
            --tool-uploader-shell-border: rgba(148, 163, 184, 0.12);
            --tool-uploader-copy: #93a4b6;
            --shadow: 0 28px 72px rgba(0, 0, 0, 0.38);
            --shadow-soft: 0 14px 32px rgba(0, 0, 0, 0.24);
            --badge-bg: rgba(91, 184, 201, 0.12);
            --badge-border: rgba(91, 184, 201, 0.16);
            --badge-text: #a8ebf5;
            --display-heading: #f7fbff;
            --display-copy: #97a8ba;
            --auth-bg: rgba(13, 21, 30, 0.84);
            --auth-heading: #f7fbff;
            --auth-copy: #98aabc;
            --input-bg: rgba(255,255,255,0.05);
            --input-border: rgba(148, 163, 184, 0.14);
            --input-text: #f7fbff;
            --placeholder: #8093a7;
            --form-label: #edf5ff;
            --status-idle-bg: rgba(148, 163, 184, 0.12);
            --status-idle-text: #c5d1df;
            --status-running-bg: rgba(91, 184, 201, 0.14);
            --status-running-text: #b7f0f7;
            --status-stopped-bg: rgba(245, 158, 11, 0.14);
            --status-stopped-text: #ffd89a;
            --table-bg: rgba(12,19,28,0.94);
            --table-head: rgba(255,255,255,0.04);
            --table-text: #eef5fd;
            --select-menu-bg: #0f1822;
            --select-menu-surface: #121d28;
            --select-menu-row-bg: #121d28;
            --select-menu-text: #eef5fd;
            --select-menu-border: rgba(148, 163, 184, 0.16);
            --select-menu-hover-bg: rgba(91, 184, 201, 0.14);
            --select-menu-hover-text: #b7f0f7;
            --select-menu-shadow: 0 24px 48px rgba(0, 0, 0, 0.40);
        """

    css = """
    <style>
        :root {
""" + theme_vars + """
            --radius-xl: 28px;
            --radius-lg: 20px;
            --radius-md: 14px;
            --glass-blur: blur(20px);
        }

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
            background:
                radial-gradient(circle at top left, var(--page-spotlight), transparent 28%),
                radial-gradient(circle at top right, rgba(255, 255, 255, 0.05), transparent 18%),
                linear-gradient(180deg, var(--page-bg) 0%, var(--page-bg-soft) 100%);
            color: var(--text);
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        [data-testid="stHeader"] { background: transparent; }
        #MainMenu, footer { visibility: hidden; }

        .block-container {
            max-width: 1180px;
            padding-top: 1.15rem;
            padding-bottom: 3.5rem;
        }

        .topbar-row-anchor,
        .theme-topbar-anchor {
            display: none;
        }

        .topbar-shell,
        .hero-shell,
        .steps-shell,
        .progress-shell,
        .tool-shell,
        .results-shell,
        .download-shell,
        .auth-shell,
        .section-head,
        .tool-card,
        .metric-card,
        .status-card,
        div[data-testid="stForm"] {
            backdrop-filter: var(--glass-blur);
            -webkit-backdrop-filter: var(--glass-blur);
        }

        .topbar-shell,
        .hero-shell,
        .steps-shell,
        .progress-shell,
        .tool-shell,
        .results-shell,
        .download-shell,
        .auth-shell {
            border: 1px solid var(--line);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow);
        }

        .topbar-shell {
            background: var(--topbar-bg);
            padding: 18px 22px;
            min-height: 88px;
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }

        .hero-shell,
        .steps-shell {
            background: var(--hero-bg);
            border-color: var(--hero-line);
        }

        .progress-shell { background: var(--progress-bg); }
        .tool-shell { background: var(--tool-bg); }
        .results-shell,
        .download-shell,
        .auth-shell { background: var(--auth-bg); }

        .brand-lockup {
            display: flex;
            align-items: center;
            gap: 14px;
            min-width: 0;
        }

        .brand-mark {
            width: 42px;
            height: 42px;
            border-radius: 14px;
            display: grid;
            place-items: center;
            background: linear-gradient(135deg, var(--accent-soft) 0%, rgba(255,255,255,0.08) 100%);
            border: 1px solid var(--line);
            color: var(--accent-strong);
            font-size: 1rem;
            font-weight: 800;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.14);
        }

        .brand-title {
            margin: 0;
            color: var(--topbar-text);
            font-size: 1rem;
            font-weight: 700;
            letter-spacing: -0.02em;
            line-height: 1.1;
        }

        .brand-subtitle {
            margin: 4px 0 0;
            color: var(--topbar-muted);
            font-size: 0.82rem;
            line-height: 1.3;
        }

        .nav-links {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 22px;
            flex-wrap: wrap;
            width: 100%;
            color: var(--topbar-muted);
            font-size: 0.86rem;
            font-weight: 600;
            letter-spacing: 0.01em;
        }

        .nav-links span {
            white-space: nowrap;
            position: relative;
        }

        .nav-links .is-active {
            color: var(--topbar-active);
        }

        .nav-links .is-active::after {
            content: "";
            position: absolute;
            left: 0;
            right: 0;
            bottom: -12px;
            margin: auto;
            width: 32px;
            height: 2px;
            border-radius: 999px;
            background: var(--accent);
        }

        .appearance-shell {
            justify-content: space-between;
            gap: 16px;
        }

        .appearance-shell__kicker {
            color: var(--topbar-muted);
            font-size: 0.74rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            margin: 0 0 6px;
        }

        .appearance-shell__title {
            color: var(--topbar-text);
            font-size: 1.02rem;
            font-weight: 700;
            line-height: 1.2;
            margin: 0;
            letter-spacing: -0.02em;
        }

        .appearance-shell__hint {
            color: var(--topbar-muted);
            font-size: 0.83rem;
            line-height: 1.4;
            margin: 6px 0 0;
        }

        .hero-shell {
            position: relative;
            overflow: hidden;
            padding: 40px 34px;
            margin-bottom: 1rem;
            text-align: center;
        }

        .hero-shell::before {
            content: "";
            position: absolute;
            inset: 0;
            background:
                radial-gradient(circle at 20% 15%, var(--accent-soft), transparent 24%),
                linear-gradient(180deg, rgba(255,255,255,0.03), transparent 45%);
            pointer-events: none;
        }

        .hero-shell > * {
            position: relative;
            z-index: 1;
        }

        .hero-badge,
        .section-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 8px 14px;
            border-radius: 999px;
            background: var(--badge-bg);
            border: 1px solid var(--badge-border);
            color: var(--badge-text);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .hero-shell h1,
        .steps-head h2,
        .parser-heading h2,
        .auth-shell h1 {
            margin: 18px 0 0;
            color: var(--hero-text);
            font-weight: 750;
            line-height: 1.04;
            letter-spacing: -0.045em;
        }

        .hero-shell h1 {
            font-size: clamp(2.5rem, 5vw, 4.4rem);
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        }

        .hero-shell h1 .accent {
            color: var(--accent-strong);
        }

        .hero-copy,
        .parser-copy,
        .auth-copy,
        .section-copy {
            max-width: 760px;
            margin: 16px auto 0;
            color: var(--hero-muted);
            line-height: 1.75;
            font-size: 0.98rem;
        }

        .hero-actions {
            margin-top: 26px;
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 12px;
        }

        .hero-btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 48px;
            padding: 0 22px;
            border-radius: 14px;
            border: 1px solid var(--line);
            font-size: 0.92rem;
            font-weight: 650;
            box-shadow: var(--shadow-soft);
        }

        .hero-btn.primary {
            background: var(--tool-primary-bg);
            border-color: transparent;
            color: var(--tool-primary-text);
        }

        .hero-btn.ghost {
            background: var(--surface-soft);
            color: var(--hero-text);
        }

        .hero-benefits {
            margin: 28px auto 4px;
            max-width: 940px;
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
        }

        .hero-benefit,
        .step-card,
        .metric-card,
        .tool-card,
        .status-card,
        .section-head {
            background: var(--surface);
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
        }

        .hero-benefit {
            border-radius: 18px;
            padding: 18px 18px 16px;
            text-align: left;
            color: var(--hero-muted);
            font-size: 0.84rem;
            line-height: 1.55;
        }

        .hero-benefit strong {
            display: block;
            color: var(--hero-text);
            font-size: 0.96rem;
            font-weight: 700;
            margin-bottom: 6px;
            letter-spacing: -0.02em;
        }

        .steps-shell {
            padding: 30px;
            margin-bottom: 1.2rem;
        }

        .steps-head {
            text-align: center;
            margin-bottom: 22px;
        }

        .steps-head h2 {
            font-size: clamp(2rem, 4vw, 3.1rem);
        }

        .steps-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 16px;
        }

        .step-card {
            position: relative;
            border-radius: 20px;
            padding: 20px 18px 18px;
            min-height: 162px;
            overflow: hidden;
        }

        .step-card::after {
            content: "";
            position: absolute;
            top: -40px;
            right: -30px;
            width: 100px;
            height: 100px;
            border-radius: 999px;
            background: var(--accent-soft);
            opacity: 0.8;
        }

        .step-icon {
            width: 36px;
            height: 36px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            background: var(--tool-icon-bg);
            border: 1px solid var(--tool-icon-border);
            color: var(--tool-icon-text);
            font-size: 0.92rem;
            margin-bottom: 14px;
            position: relative;
            z-index: 1;
        }

        .step-kicker,
        .metric-card__label {
            color: var(--badge-text);
            font-size: 0.70rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .step-title,
        .tool-card__title,
        .status-card__title,
        .section-title,
        .progress-title,
        .progress-step-title {
            color: var(--text-strong);
            font-weight: 700;
            letter-spacing: -0.02em;
        }

        .step-title {
            font-size: 1rem;
            margin-bottom: 8px;
            line-height: 1.3;
            position: relative;
            z-index: 1;
        }

        .step-copy,
        .tool-card__copy,
        .status-card__copy,
        .progress-step-copy,
        .metric-card__value,
        .file-chip,
        .auth-footer-note {
            color: var(--muted);
            line-height: 1.55;
        }

        .parser-intro {
            text-align: center;
            padding: 1rem 0 1.25rem;
        }

        .parser-heading h2 {
            color: var(--display-heading);
            font-size: clamp(1.9rem, 4vw, 3rem);
        }

        .workspace-grid {
            display: grid;
            grid-template-columns: minmax(260px, 0.88fr) minmax(0, 1.42fr);
            gap: 18px;
            align-items: start;
            margin-bottom: 1.2rem;
        }

        .progress-shell,
        .tool-shell,
        .results-shell,
        .download-shell {
            padding: 20px;
        }

        .progress-title {
            font-size: 1rem;
            margin-bottom: 18px;
        }

        .progress-steps {
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        .progress-step {
            display: grid;
            grid-template-columns: 34px 1fr;
            gap: 12px;
            align-items: start;
        }

        .progress-index {
            width: 34px;
            height: 34px;
            border-radius: 999px;
            display: grid;
            place-items: center;
            font-size: 0.84rem;
            font-weight: 700;
            border: 1px solid var(--progress-border);
            color: var(--progress-copy);
            background: var(--surface-soft);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }

        .progress-step.is-active .progress-index {
            background: var(--tool-primary-bg);
            color: var(--tool-primary-text);
            border-color: transparent;
            box-shadow: 0 0 0 6px var(--accent-soft);
        }

        .progress-step-copy {
            font-size: 0.80rem;
            margin-top: 4px;
        }

        .progress-divider {
            height: 1px;
            background: var(--progress-border);
            margin: 18px 0 14px;
        }

        .progress-footer {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            color: var(--progress-copy);
            font-size: 0.82rem;
            font-weight: 650;
        }

        .mini-pill,
        .status-pill,
        .file-chip {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            border-radius: 999px;
            font-weight: 700;
        }

        .mini-pill {
            padding: 7px 12px;
            background: var(--progress-pill-bg);
            color: var(--progress-pill-text);
            font-size: 0.76rem;
            min-width: 74px;
        }

        .tool-card {
            border-radius: 18px;
            padding: 16px;
            margin-bottom: 0.7rem;
        }

        .tool-card__head {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .tool-card__icon {
            width: 32px;
            height: 32px;
            border-radius: 11px;
            display: grid;
            place-items: center;
            background: var(--tool-icon-bg);
            border: 1px solid var(--tool-icon-border);
            color: var(--tool-icon-text);
            font-size: 0.94rem;
            flex: none;
        }

        .tool-card__title { font-size: 0.96rem; }
        .tool-card__copy { font-size: 0.80rem; margin-top: 3px; }

        .file-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 0.6rem 0 0.95rem;
        }

        .file-chip {
            padding: 9px 13px;
            background: var(--surface-elevated);
            border: 1px solid var(--line);
            color: var(--text);
            font-size: 0.8rem;
            box-shadow: var(--shadow-soft);
        }

        .file-chip.is-encrypted {
            background: var(--status-stopped-bg);
            color: var(--status-stopped-text);
            border-color: transparent;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-bottom: 1rem;
        }

        .metric-card,
        .status-card,
        .section-head {
            border-radius: 20px;
        }

        .metric-card {
            padding: 16px;
        }

        .metric-card__value {
            color: var(--text-strong);
            font-size: 1.04rem;
            font-weight: 700;
        }

        .status-card {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 18px;
            margin-bottom: 1rem;
        }

        .status-card__group {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .status-card__dot {
            width: 12px;
            height: 12px;
            border-radius: 999px;
            background: var(--muted);
            box-shadow: 0 0 0 8px rgba(148, 163, 184, 0.08);
            flex: none;
        }

        .status-card.is-running .status-card__dot {
            background: var(--accent);
            box-shadow: 0 0 0 8px var(--accent-soft);
        }

        .status-card.is-stopped .status-card__dot {
            background: #f59e0b;
            box-shadow: 0 0 0 8px rgba(245, 158, 11, 0.10);
        }

        .status-pill {
            min-width: 86px;
            padding: 8px 12px;
            font-size: 0.76rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .status-card.is-idle .status-pill {
            background: var(--status-idle-bg);
            color: var(--status-idle-text);
        }

        .status-card.is-running .status-pill {
            background: var(--status-running-bg);
            color: var(--status-running-text);
        }

        .status-card.is-stopped .status-pill {
            background: var(--status-stopped-bg);
            color: var(--status-stopped-text);
        }

        .section-head {
            display: flex;
            flex-direction: column;
            gap: 8px;
            padding: 18px;
            margin: 0 0 14px;
        }

        .section-title {
            margin: 0;
            font-size: 1.06rem;
        }

        .section-copy {
            margin: 0;
            font-size: 0.92rem;
            max-width: none;
        }

        .auth-shell {
            max-width: 760px;
            margin: 7vh auto 0;
            padding: 28px 26px 22px;
        }

        .auth-shell h1 {
            font-size: clamp(1.9rem, 4vw, 2.7rem);
            color: var(--auth-heading);
        }

        .auth-copy {
            color: var(--auth-copy);
            margin-left: 0;
            margin-right: 0;
        }

        .auth-footer-note {
            margin-top: 12px;
            text-align: center;
            font-size: 0.88rem;
        }

        div[data-testid="stForm"] {
            background: var(--surface);
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
            padding: 22px 20px 20px;
            margin: 1rem auto 0;
            max-width: 760px;
            border-radius: 24px;
        }

        div[data-testid="stWidgetLabel"] p,
        div[data-testid="stTextInput"] label p,
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stFileUploader"] label p,
        div[data-testid="stTextArea"] label p {
            color: var(--form-label) !important;
            font-size: 0.9rem;
            font-weight: 650 !important;
            opacity: 1 !important;
            letter-spacing: -0.01em;
        }

        .tool-shell div[data-testid="stWidgetLabel"] p,
        .tool-shell div[data-testid="stTextInput"] label p,
        .tool-shell div[data-testid="stSelectbox"] label p,
        .tool-shell div[data-testid="stFileUploader"] label p,
        .tool-shell div[data-testid="stTextArea"] label p {
            color: var(--tool-title) !important;
        }

        div[data-testid="stSelectbox"],
        div[data-testid="stTextInput"],
        div[data-testid="stTextArea"],
        div[data-testid="stFileUploader"] {
            margin-bottom: 1rem;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="textarea"] > div {
            min-height: 52px;
            border-radius: var(--radius-md) !important;
            border: 1px solid var(--input-border) !important;
            background: var(--input-bg) !important;
            box-shadow: none !important;
            transition: border-color 160ms ease, box-shadow 160ms ease, transform 160ms ease;
        }

        .tool-shell div[data-baseweb="input"] > div,
        .tool-shell div[data-baseweb="select"] > div,
        .tool-shell div[data-baseweb="textarea"] > div {
            background: var(--tool-input-bg) !important;
            border-color: var(--tool-input-border) !important;
        }

        div[data-baseweb="input"] > div:hover,
        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="textarea"] > div:hover {
            border-color: var(--line-strong) !important;
        }

        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within,
        div[data-baseweb="textarea"] > div:focus-within {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 4px var(--accent-soft) !important;
        }

        div[data-baseweb="input"] input,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="textarea"] textarea,
        div[data-baseweb="select"] svg,
        div[data-baseweb="select"] path {
            color: var(--input-text) !important;
            -webkit-text-fill-color: var(--input-text) !important;
            fill: var(--input-text) !important;
            stroke: var(--input-text) !important;
            opacity: 1 !important;
        }

        .tool-shell div[data-baseweb="input"] input,
        .tool-shell div[data-baseweb="select"] input,
        .tool-shell div[data-baseweb="select"] span,
        .tool-shell div[data-baseweb="textarea"] textarea,
        .tool-shell div[data-baseweb="select"] svg,
        .tool-shell div[data-baseweb="select"] path {
            color: var(--tool-input-text) !important;
            -webkit-text-fill-color: var(--tool-input-text) !important;
            fill: var(--tool-input-text) !important;
            stroke: var(--tool-input-text) !important;
        }

        div[data-baseweb="input"] input::placeholder,
        div[data-baseweb="textarea"] textarea::placeholder {
            color: var(--placeholder) !important;
            opacity: 1 !important;
        }

        .tool-shell div[data-baseweb="input"] input::placeholder,
        .tool-shell div[data-baseweb="textarea"] textarea::placeholder {
            color: var(--tool-placeholder) !important;
        }

        body div[data-baseweb="popover"] {
            background: transparent !important;
        }

        body div[data-baseweb="popover"] > div,
        body div[data-baseweb="popover"] > div > div,
        body div[data-baseweb="menu"],
        body ul[role="listbox"],
        body div[role="listbox"] {
            background: var(--select-menu-bg) !important;
            background-color: var(--select-menu-bg) !important;
            border: 1px solid var(--select-menu-border) !important;
            border-radius: 18px !important;
            box-shadow: var(--select-menu-shadow) !important;
        }

        body ul[role="listbox"],
        body div[role="listbox"],
        body div[data-baseweb="menu"] {
            background: var(--select-menu-surface) !important;
            background-color: var(--select-menu-surface) !important;
            padding: 8px !important;
            overflow: hidden !important;
        }

        body div[role="option"],
        body li[role="option"] {
            background: var(--select-menu-row-bg) !important;
            border-radius: 12px !important;
            margin: 2px 0 !important;
        }

        body div[role="option"],
        body li[role="option"],
        body div[role="option"] *,
        body li[role="option"] * {
            color: var(--select-menu-text) !important;
            -webkit-text-fill-color: var(--select-menu-text) !important;
            opacity: 1 !important;
        }

        body div[role="option"]:hover,
        body li[role="option"]:hover,
        body div[role="option"][aria-selected="true"],
        body li[role="option"][aria-selected="true"] {
            background: var(--select-menu-hover-bg) !important;
            color: var(--select-menu-hover-text) !important;
        }

        body div[role="option"]:hover *,
        body li[role="option"]:hover *,
        body div[role="option"][aria-selected="true"] *,
        body li[role="option"][aria-selected="true"] * {
            color: var(--select-menu-hover-text) !important;
            -webkit-text-fill-color: var(--select-menu-hover-text) !important;
        }

        div[data-testid="stFileUploader"] > section {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: var(--shadow-soft);
        }

        .tool-shell div[data-testid="stFileUploader"] > section {
            background: var(--tool-uploader-shell-bg);
            border-color: var(--tool-uploader-shell-border);
            box-shadow: none;
        }

        div[data-testid="stFileUploaderDropzone"] {
            border: 1.5px dashed var(--line-strong);
            border-radius: 16px;
            background: var(--input-bg);
            padding: 20px;
        }

        .tool-shell div[data-testid="stFileUploaderDropzone"] {
            background: var(--tool-input-bg);
            border-color: var(--tool-input-border);
        }

        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] p {
            color: var(--muted);
        }

        .tool-shell div[data-testid="stFileUploader"] small,
        .tool-shell div[data-testid="stFileUploader"] span,
        .tool-shell div[data-testid="stFileUploader"] p {
            color: var(--tool-uploader-copy);
        }

        div.stButton > button,
        div.stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] > button,
        div[data-testid="stFileUploader"] section button {
            min-height: 46px;
            border-radius: 14px;
            padding: 0 16px;
            font-weight: 650;
            border: 1px solid var(--tool-button-border);
            background: var(--tool-button-bg);
            color: var(--tool-button-text);
            box-shadow: var(--shadow-soft);
            transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease, background 160ms ease;
        }

        div.stButton > button:hover,
        div.stDownloadButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover,
        div[data-testid="stFileUploader"] section button:hover {
            transform: translateY(-1px);
            border-color: var(--accent);
            background: var(--tool-button-hover-bg);
        }

        div.stButton > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
            background: var(--tool-primary-bg);
            color: var(--tool-primary-text);
            border-color: transparent;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 18px;
            overflow: hidden;
            border: 1px solid var(--line);
            background: var(--table-bg);
            box-shadow: var(--shadow-soft);
        }

        div[data-testid="stDataFrame"] [role="grid"] {
            background: var(--table-bg);
            color: var(--table-text);
        }

        div[data-testid="stAlert"] {
            border-radius: 14px;
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
        }

        div[data-testid="stProgressBar"] > div > div {
            background-color: var(--accent-soft);
        }

        div[data-testid="stProgressBar"] div[role="progressbar"] {
            background: var(--tool-primary-bg);
        }

        @media (max-width: 980px) {
            .hero-benefits,
            .steps-grid,
            .workspace-grid {
                grid-template-columns: 1fr;
            }

            .nav-links {
                justify-content: flex-start;
            }
        }

        @media (max-width: 760px) {
            .block-container {
                padding-top: 0.8rem;
            }

            .topbar-shell,
            .hero-shell,
            .steps-shell,
            .progress-shell,
            .tool-shell,
            .results-shell,
            .download-shell,
            .auth-shell,
            div[data-testid="stForm"] {
                padding: 16px;
            }

            .hero-shell h1,
            .steps-head h2,
            .parser-heading h2,
            .auth-shell h1 {
                font-size: 2.15rem;
            }

            .status-card {
                flex-direction: column;
                align-items: flex-start;
            }
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

def render_top_bar() -> None:
    st.markdown('<div class="topbar-row-anchor"></div>', unsafe_allow_html=True)
    left, middle, right = columns_compat([1.32, 1.38, 1.30], gap="medium", vertical_alignment="center")
    left.markdown(
        """
        <div class="topbar-shell">
            <div class="brand-lockup">
                <div class="brand-mark">✶</div>
                <div class="brand-copy">
                    <div class="brand-title">KreditLab</div>
                    <div class="brand-subtitle">Statement intelligence workspace</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    middle.markdown(
        """
        <div class="topbar-shell">
            <div class="nav-links">
                <span class="is-active">Overview</span>
                <span>Workflow</span>
                <span>Security</span>
                <span>Support</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    is_light = st.session_state.get("ui_theme_light", False)
    theme_state = "Light mode" if is_light else "Dark mode"
    right.markdown(
        f"""
        <div class="topbar-shell appearance-shell">
            <div class="appearance-shell__copy">
                <p class="appearance-shell__kicker">Appearance · {html.escape(theme_state)}</p>
                <p class="appearance-shell__title">Ready to parse</p>
                <p class="appearance-shell__hint">Upload a statement and review structured output</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mode_button_label = "☀️ Light mode" if is_light else "🌙 Dark mode"
    mode_changed = st.button(
        mode_button_label,
        key="theme_mode_button",
        help="Switch between light and dark interface modes",
    )
    if mode_changed:
        st.session_state.ui_theme_light = not is_light
        st.session_state.ui_theme_mode = "Light" if st.session_state.ui_theme_light else "Dark"
        st.rerun()


def render_auth_shell() -> None:
    st.markdown(
        """
        <section class="auth-shell">
            <div class="auth-shell__logo">
                <span class="section-badge">Secure access</span>
            </div>
            <h1>Access the statement workspace</h1>
            <p class="auth-copy">Sign in to continue. The interface has been refined for a cleaner, more premium experience while keeping the same authentication and parser behaviour.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_app_hero() -> None:
    st.markdown(
        """
        <section class="hero-shell">
            <span class="hero-badge">KreditLab · Bank statement parser</span>
            <h1>Professional statement parsing with a <span class="accent">clear, modern workspace</span></h1>
            <p class="hero-copy">Upload a bank statement PDF and review structured transactions, summaries, and export-ready results in a more polished interface designed for speed and clarity.</p>
            <div class="hero-actions">
                <span class="hero-btn primary">⇪&nbsp; Upload statement</span>
                <span class="hero-btn ghost">Review workflow</span>
            </div>
            <div class="hero-benefits">
                <div class="hero-benefit"><strong>Cleaner hierarchy</strong>Sharper spacing and calmer surfaces make high-value actions easier to scan.</div>
                <div class="hero-benefit"><strong>Premium feel</strong>Refined contrast, depth, and typography elevate the interface without adding clutter.</div>
                <div class="hero-benefit"><strong>Same workflow</strong>All parser controls, upload steps, and outputs remain unchanged.</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_steps_showcase() -> None:
    st.markdown(
        """
        <section class="steps-shell">
            <div class="steps-head">
                <span class="section-badge">Workflow</span>
                <h2>Four familiar steps, refined visually</h2>
            </div>
            <div class="steps-grid">
                <div class="step-card">
                    <div class="step-icon">▣</div>
                    <div class="step-kicker">Step 1</div>
                    <div class="step-title">Select your bank</div>
                    <div class="step-copy">Choose the statement format from the supported bank list.</div>
                </div>
                <div class="step-card">
                    <div class="step-icon">⤴</div>
                    <div class="step-kicker">Step 2</div>
                    <div class="step-title">Upload statement</div>
                    <div class="step-copy">Add the PDF file exactly as before using the existing uploader flow.</div>
                </div>
                <div class="step-card">
                    <div class="step-icon">∿</div>
                    <div class="step-kicker">Step 3</div>
                    <div class="step-title">Process and analyse</div>
                    <div class="step-copy">The parser continues extracting and structuring the statement automatically.</div>
                </div>
                <div class="step-card">
                    <div class="step-icon">▥</div>
                    <div class="step-kicker">Step 4</div>
                    <div class="step-title">Review results</div>
                    <div class="step-copy">Inspect transactions, summaries, and downloads in the refreshed workspace.</div>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_parser_intro() -> None:
    st.markdown(
        """
        <div class="parser-intro">
            <div class="parser-heading">
                <span class="section-badge">Parser engine</span>
                <h2>Upload and parse with the same trusted flow</h2>
                <p class="parser-copy">Select the bank, upload the statement PDF, and let the parser generate structured financial data in a cleaner, more professional interface.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tool_card_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="tool-card">
            <div class="tool-card__head">
                <div class="tool-card__icon">{html.escape(icon)}</div>
                <div>
                    <div class="tool-card__title">{html.escape(title)}</div>
                    <div class="tool-card__copy">{html.escape(subtitle)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def close_tool_card() -> None:
    return None


def _current_progress_step(uploaded_files: List, status: str, has_results: bool) -> int:
    if has_results:
        return 4
    if str(status or "").lower() == "running":
        return 3
    if uploaded_files:
        return 2
    return 1


def render_progress_panel(status: str, uploaded_files: List, has_results: bool) -> None:
    current_step = _current_progress_step(uploaded_files, status, has_results)
    status_key = str(status or "idle").strip().lower()
    status_label = {"idle": "Idle", "running": "Running", "stopped": "Stopped"}.get(status_key, status.title())
    steps = [
        ("Select Your Bank", "Current step" if current_step == 1 else "Choose the parser format"),
        ("Upload Statement", "Ready when PDF files are added"),
        ("Process & Analyse", "Start the parser to extract data"),
        ("View Results", "Review tables and download reports"),
    ]

    steps_html = "".join(
        f'<div class="progress-step{" is-active" if idx == current_step else ""}"><div class="progress-index">{idx}</div><div><div class="progress-step-title">{html.escape(title)}</div><div class="progress-step-copy">{html.escape(copy)}</div></div></div>'
        for idx, (title, copy) in enumerate(steps, start=1)
    )
    st.markdown(
        f"""
        <section class="progress-shell">
            <div class="progress-title">Progress</div>
            <div class="progress-steps">{steps_html}</div>
            <div class="progress-divider"></div>
            <div class="progress-footer"><span>Status</span><span class="mini-pill">{html.escape(status_label)}</span></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(label: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-head">
            <span class="section-badge">{html.escape(label)}</span>
            <h2 class="section-title">{html.escape(title)}</h2>
            <p class="section-copy">{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_card(status: str) -> None:
    status_key = (status or "idle").strip().lower()
    status_copy = {
        "idle": "Ready to accept uploads and begin parsing.",
        "running": "Processing uploaded statements and generating outputs.",
        "stopped": "Run paused. You can resume or reset the workspace.",
    }
    status_label = {
        "idle": "Idle",
        "running": "Running",
        "stopped": "Stopped",
    }.get(status_key, status.upper())
    st.markdown(
        f"""
        <div class="status-card is-{html.escape(status_key)}">
            <div class="status-card__group">
                <span class="status-card__dot"></span>
                <div>
                    <div class="status-card__title">Processing status</div>
                    <div class="status-card__copy">{html.escape(status_copy.get(status_key, "Workspace updated."))}</div>
                </div>
            </div>
            <span class="status-pill">{html.escape(status_label)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_file_chips(uploaded_files: List, encrypted_files: List[str]) -> None:
    if not uploaded_files:
        return

    chips = []
    encrypted_set = set(encrypted_files or [])
    for uploaded_file in uploaded_files:
        name = getattr(uploaded_file, "name", str(uploaded_file))
        extra_class = " is-encrypted" if name in encrypted_set else ""
        icon = "🔒" if name in encrypted_set else "📎"
        chips.append(f'<span class="file-chip{extra_class}">{icon} {html.escape(name)}</span>')

    st.markdown(f'<div class="file-chip-row">{"".join(chips)}</div>', unsafe_allow_html=True)


def _supports_streamlit_kwarg(func, name: str) -> bool:
    try:
        return name in inspect.signature(func).parameters
    except Exception:
        return False


def columns_compat(spec, **kwargs):
    call_kwargs = dict(kwargs)
    if "vertical_alignment" in call_kwargs and not _supports_streamlit_kwarg(st.columns, "vertical_alignment"):
        call_kwargs.pop("vertical_alignment", None)
    return st.columns(spec, **call_kwargs)


def button_compat(label: str, primary: bool = False, **kwargs):
    call_kwargs = dict(kwargs)
    if primary and _supports_streamlit_kwarg(st.button, "type"):
        call_kwargs["type"] = "primary"
    if "use_container_width" in call_kwargs and not _supports_streamlit_kwarg(st.button, "use_container_width"):
        call_kwargs.pop("use_container_width", None)
    return st.button(label, **call_kwargs)


def form_submit_button_compat(label: str, primary: bool = False, **kwargs):
    call_kwargs = dict(kwargs)
    if primary and _supports_streamlit_kwarg(st.form_submit_button, "type"):
        call_kwargs["type"] = "primary"
    if "use_container_width" in call_kwargs and not _supports_streamlit_kwarg(st.form_submit_button, "use_container_width"):
        call_kwargs.pop("use_container_width", None)
    return st.form_submit_button(label, **call_kwargs)


def toggle_compat(label: str, **kwargs):
    call_kwargs = dict(kwargs)
    if hasattr(st, "toggle"):
        if "label_visibility" in call_kwargs and not _supports_streamlit_kwarg(st.toggle, "label_visibility"):
            call_kwargs.pop("label_visibility", None)
        return st.toggle(label, **call_kwargs)
    call_kwargs.pop("label_visibility", None)
    return st.checkbox(label, **call_kwargs)


def download_button_compat(label: str, *args, **kwargs):
    call_kwargs = dict(kwargs)
    if "use_container_width" in call_kwargs and not _supports_streamlit_kwarg(st.download_button, "use_container_width"):
        call_kwargs.pop("use_container_width", None)
    return st.download_button(label, *args, **call_kwargs)


def render_metric_cards(items: List[Tuple[str, str]]) -> None:
    if not items:
        return
    cards_html = "".join(
        f'<div class="metric-card"><div class="metric-card__label">{html.escape(label)}</div><div class="metric-card__value">{html.escape(value)}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="metric-grid">{cards_html}</div>', unsafe_allow_html=True)
