import html
import inspect
from typing import List, Tuple

import streamlit as st

def inject_global_styles(theme_mode: str = "Dark") -> None:
    is_light = str(theme_mode or "Dark").strip().lower() == "light"
    if is_light:
        theme_vars = """

            --page-bg: #ffffff;
            --page-bg-soft: #ffffff;
            --page-spotlight: rgba(18, 184, 171, 0.00);
            --surface: #ffffff;
            --surface-soft: #ffffff;
            --surface-elevated: #ffffff;
            --panel: #ffffff;
            --panel-soft: #ffffff;
            --text: #111827;
            --text-strong: #0b1220;
            --muted: #475569;
            --line: rgba(15, 23, 42, 0.10);
            --line-strong: rgba(15, 23, 42, 0.16);
            --accent: #12b8ab;
            --accent-strong: #0d8f85;
            --accent-soft: rgba(18, 184, 171, 0.10);
            --navy: #0f172a;
            --navy-soft: #1e293b;
            --hero-bg: #ffffff;
            --hero-surface: #ffffff;
            --hero-line: rgba(15, 23, 42, 0.10);
            --hero-text: #0b1220;
            --hero-muted: #475569;
            --hero-subtle: #0b1220;
            --hero-card-bg: #ffffff;
            --hero-card-overlay: rgba(18, 184, 171, 0.05);
            --hero-ghost: #f8fafc;
            --topbar-bg: #ffffff;
            --topbar-border: rgba(15, 23, 42, 0.10);
            --topbar-text: #0b1220;
            --topbar-muted: #475569;
            --topbar-active: #0b1220;
            --theme-card-bg: #ffffff;
            --theme-card-border: rgba(15, 23, 42, 0.10);
            --theme-icon-bg: #f0fdfa;
            --theme-icon-border: rgba(18, 184, 171, 0.24);
            --theme-icon-text: #0d8f85;
            --mode-toggle-bg: #0f172a;
            --mode-toggle-text: #f8fafc;
            --mode-toggle-border: rgba(15, 23, 42, 0.22);
            --mode-toggle-hover-bg: #111827;
            --mode-toggle-hover-text: #ffffff;
            --mode-toggle-hover-border: rgba(15, 23, 42, 0.32);
            --progress-bg: #ffffff;
            --progress-border: rgba(15, 23, 42, 0.10);
            --progress-title: #0b1220;
            --progress-copy: #475569;
            --progress-subtle: #0b1220;
            --progress-pill-bg: #f8fafc;
            --progress-pill-text: #475569;
            --tool-bg: #ffffff;
            --tool-border: rgba(15, 23, 42, 0.10);
            --tool-card-bg: #ffffff;
            --tool-card-border: rgba(15, 23, 42, 0.10);
            --tool-title: #0b1220;
            --tool-copy: #475569;
            --tool-icon-bg: rgba(18, 184, 171, 0.10);
            --tool-icon-border: rgba(18, 184, 171, 0.20);
            --tool-icon-text: #0d8f85;
            --tool-input-bg: #ffffff;
            --tool-input-border: rgba(15, 23, 42, 0.16);
            --tool-input-text: #0b1220;
            --tool-placeholder: #64748b;
            --tool-button-bg: #ffffff;
            --tool-button-text: #0b1220;
            --tool-button-border: rgba(15, 23, 42, 0.12);
            --tool-button-hover-bg: #f8fafc;
            --tool-primary-bg: #14b8a6;
            --tool-primary-text: #ffffff;
            --tool-uploader-shell-bg: #ffffff;
            --tool-uploader-shell-border: rgba(15, 23, 42, 0.10);
            --tool-uploader-copy: #475569;
            --shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
            --shadow-soft: 0 6px 16px rgba(15, 23, 42, 0.04);
            --badge-bg: rgba(18, 184, 171, 0.10);
            --badge-border: rgba(18, 184, 171, 0.20);
            --badge-text: #0d8f85;
            --display-heading: #0b1220;
            --display-copy: #475569;
            --auth-bg: #ffffff;
            --auth-heading: #0b1220;
            --auth-copy: #475569;
            --input-bg: #ffffff;
            --input-border: rgba(15, 23, 42, 0.18);
            --input-text: #0b1220;
            --placeholder: #64748b;
            --form-label: #0b1220;
            --status-idle-bg: #f8fafc;
            --status-idle-text: #475569;
            --status-running-bg: rgba(18, 184, 171, 0.12);
            --status-running-text: #0d8f85;
            --status-stopped-bg: #fff1e7;
            --status-stopped-text: #a85a10;
            --table-bg: #ffffff;
            --table-head: #f8fafc;
            --table-text: #0b1220;
            --select-menu-bg: #f8fafc;
            --select-menu-surface: #ffffff;
            --select-menu-row-bg: #ffffff;
            --select-menu-text: #0f172a;
            --select-menu-border: rgba(15, 23, 42, 0.14);
            --select-menu-hover-bg: rgba(18, 184, 171, 0.12);
            --select-menu-hover-text: #0d8f85;
            --select-menu-shadow: 0 18px 34px rgba(15, 23, 42, 0.10);

        """
    else:

        theme_vars = """
            --page-bg: #06131a;
            --page-bg-soft: #0a1820;
            --page-spotlight: rgba(17, 213, 196, 0.07);
            --surface: #0c1a22;
            --surface-soft: #10222b;
            --surface-elevated: #122733;
            --panel: #0f202a;
            --panel-soft: #142a35;
            --text: #d4eef0;
            --text-strong: #f6ffff;
            --muted: #84afb1;
            --line: rgba(17, 213, 196, 0.14);
            --line-strong: rgba(17, 213, 196, 0.32);
            --accent: #11d5c4;
            --accent-strong: #0fb7a8;
            --accent-soft: rgba(17, 213, 196, 0.12);
            --navy: #08141b;
            --navy-soft: #0d1f29;
            --hero-bg: linear-gradient(180deg, #0d1d27 0%, #09161d 100%);
            --hero-surface: rgba(255, 255, 255, 0.02);
            --hero-line: rgba(17, 213, 196, 0.18);
            --hero-text: #f6ffff;
            --hero-muted: #8cc5c4;
            --hero-subtle: #d8f7f4;
            --hero-card-bg: rgba(255, 255, 255, 0.02);
            --hero-card-overlay: rgba(17, 213, 196, 0.05);
            --hero-ghost: rgba(255, 255, 255, 0.03);
            --topbar-bg: linear-gradient(180deg, #0d1d27 0%, #09161d 100%);
            --topbar-border: rgba(17, 213, 196, 0.18);
            --topbar-text: #f6ffff;
            --topbar-muted: #8cc5c4;
            --topbar-active: #f6ffff;
            --theme-card-bg: linear-gradient(180deg, #0d1d27 0%, #09161d 100%);
            --theme-card-border: rgba(17, 213, 196, 0.18);
            --theme-icon-bg: rgba(17, 213, 196, 0.10);
            --theme-icon-border: rgba(17, 213, 196, 0.18);
            --theme-icon-text: #11d5c4;
            --mode-toggle-bg: rgba(255, 255, 255, 0.06);
            --mode-toggle-text: #eaf9fa;
            --mode-toggle-border: rgba(17, 213, 196, 0.26);
            --mode-toggle-hover-bg: rgba(17, 213, 196, 0.16);
            --mode-toggle-hover-text: #f6ffff;
            --mode-toggle-hover-border: rgba(17, 213, 196, 0.40);
            --progress-bg: linear-gradient(180deg, #0d1d27 0%, #09161d 100%);
            --progress-border: rgba(17, 213, 196, 0.18);
            --progress-title: #f6ffff;
            --progress-copy: #8cc5c4;
            --progress-subtle: #d8f7f4;
            --progress-pill-bg: rgba(255, 255, 255, 0.94);
            --progress-pill-text: #39505b;
            --tool-bg: #0d1d27;
            --tool-border: rgba(17, 213, 196, 0.18);
            --tool-card-bg: rgba(255, 255, 255, 0.02);
            --tool-card-border: rgba(17, 213, 196, 0.18);
            --tool-title: #f6ffff;
            --tool-copy: #8cc5c4;
            --tool-icon-bg: rgba(17, 213, 196, 0.12);
            --tool-icon-border: rgba(17, 213, 196, 0.16);
            --tool-icon-text: #11d5c4;
            --tool-input-bg: rgba(255,255,255,0.03);
            --tool-input-border: rgba(17, 213, 196, 0.18);
            --tool-input-text: #f6ffff;
            --tool-placeholder: #8cc5c4;
            --tool-button-bg: rgba(255,255,255,0.04);
            --tool-button-text: #d8f7f4;
            --tool-button-border: rgba(17, 213, 196, 0.18);
            --tool-button-hover-bg: rgba(255,255,255,0.07);
            --tool-primary-bg: #11d5c4;
            --tool-primary-text: #082126;
            --tool-uploader-shell-bg: #101922;
            --tool-uploader-shell-border: rgba(17, 213, 196, 0.12);
            --tool-uploader-copy: #9db8bb;
            --shadow: 0 18px 44px rgba(0, 0, 0, 0.28);
            --shadow-soft: 0 10px 26px rgba(0, 0, 0, 0.18);
            --badge-bg: rgba(17, 213, 196, 0.10);
            --badge-border: rgba(17, 213, 196, 0.20);
            --badge-text: #7ef1e6;
            --display-heading: #eaf9fa;
            --display-copy: #8db4b6;
            --auth-bg: #0d1d27;
            --auth-heading: #f6ffff;
            --auth-copy: #94c4c6;
            --input-bg: #0f202a;
            --input-border: rgba(17, 213, 196, 0.18);
            --input-text: #eaf8f8;
            --placeholder: #7ea6a8;
            --form-label: #d9f0f1;
            --status-idle-bg: rgba(235, 241, 245, 0.10);
            --status-idle-text: #dfeef0;
            --status-running-bg: rgba(17, 213, 196, 0.15);
            --status-running-text: #8ff6ec;
            --status-stopped-bg: rgba(255, 167, 38, 0.12);
            --status-stopped-text: #ffd39c;
            --table-bg: #0d1b23;
            --table-head: #122733;
            --table-text: #eaf8f8;
            --select-menu-bg: #08141b;
            --select-menu-surface: #0d1d27;
            --select-menu-row-bg: #0d1d27;
            --select-menu-text: #eaf9fa;
            --select-menu-border: rgba(17, 213, 196, 0.16);
            --select-menu-hover-bg: rgba(17, 213, 196, 0.12);
            --select-menu-hover-text: #8ff6ec;
            --select-menu-shadow: 0 22px 40px rgba(0, 0, 0, 0.34);
        """

    css = f"""
    <style>
        :root {{
{theme_vars}
            --radius-xl: 24px;
            --radius-lg: 18px;
            --radius-md: 14px;
        }}

        html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
            background:
                repeating-linear-gradient(0deg, rgba(17, 213, 196, 0.06) 0, rgba(17, 213, 196, 0.06) 1px, transparent 1px, transparent 42px),
                repeating-linear-gradient(90deg, rgba(17, 213, 196, 0.05) 0, rgba(17, 213, 196, 0.05) 1px, transparent 1px, transparent 42px),
                radial-gradient(circle at top center, var(--page-spotlight), transparent 24%),
                linear-gradient(180deg, var(--page-bg) 0%, var(--page-bg-soft) 100%);
            color: var(--text);
        }}

        [data-testid="stHeader"] {{ background: transparent; }}
        #MainMenu, footer {{ visibility: hidden; }}

        .block-container {{
            max-width: 1180px;
            padding-top: 1rem;
            padding-bottom: 3rem;
        }}

        .topbar-shell {{
            background: var(--topbar-bg);
            border: 1px solid var(--topbar-border);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow-soft);
        }}

        .hero-shell,
        .steps-shell {{
            background: var(--hero-bg);
            border: 1px solid var(--hero-line);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow-soft);
        }}

        .progress-shell {{
            background: var(--progress-bg);
            border: 1px solid var(--progress-border);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow-soft);
        }}

        .tool-shell {{
            background: var(--tool-bg);
            border: 1px solid var(--tool-border);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow);
        }}

        .topbar-shell {{
            padding: 18px 22px;
            margin-bottom: 1rem;
            min-height: 86px;
            display: flex;
            align-items: center;
        }}

        .topbar-shell--theme {{
            display: block;
        }}

        .topbar-row-anchor,
        .theme-topbar-anchor {{
            display: none;
        }}

        .theme-mode-toggle-anchor {{
            display: none;
        }}

        div[data-testid="column"]:has(.theme-topbar-anchor) {{
            margin-bottom: 1rem;
        }}

        div[data-testid="column"]:has(.theme-topbar-anchor) > div,
        div[data-testid="column"]:has(.theme-topbar-anchor) > div > div,
        div[data-testid="column"]:has(.theme-topbar-anchor) > div > div > div {{
            background: var(--topbar-bg);
            border: 1px solid var(--topbar-border);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow-soft);
            padding: 18px 22px;
            min-height: 86px;
            box-sizing: border-box;
            display: flex;
            align-items: center;
        }}

        div[data-testid="column"]:has(.theme-topbar-anchor) [data-testid="stVerticalBlock"] {{
            width: 100%;
        }}

        div[data-testid="column"]:has(.theme-topbar-anchor) div[data-testid="element-container"] {{
            margin-bottom: 0 !important;
        }}

        div[data-testid="column"]:has(.theme-topbar-anchor) [data-testid="stHorizontalBlock"] {{
            align-items: center !important;
            justify-content: flex-start;
            gap: 12px;
            min-height: 50px;
        }}

        div[data-testid="column"]:has(.theme-topbar-anchor) [data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child {{
            display: flex;
            align-items: center;
            justify-content: center;
            flex: 0 0 auto;
            max-width: 50px;
            min-width: 50px;
            padding-right: 2px;
            box-sizing: border-box;
        }}

        div[data-testid="column"]:has(.theme-topbar-anchor) [data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child {{
            flex: 1 1 auto;
            min-width: 0;
            display: flex;
            align-items: center;
            justify-content: flex-start;
        }}

        div[data-testid="column"]:has(.theme-topbar-anchor) [data-testid="stHorizontalBlock"] > div[data-testid="column"]:last-child [data-testid="stVerticalBlock"] {{
            display: flex;
            align-items: center;
            min-height: 50px;
        }}

        .theme-toggle-shell {{
            background: var(--theme-card-bg);
            border: 1px solid var(--theme-card-border);
            border-radius: 18px;
            padding: 12px 14px;
            box-shadow: var(--shadow-soft);
            margin-bottom: 0.55rem;
        }}

        .theme-toggle-shell__row {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
        }}

        .theme-toggle-shell [data-testid="stHorizontalBlock"] {{
            align-items: center;
        }}

        .theme-toggle-shell__copy {{
            min-width: 0;
        }}

        .results-shell,
        .download-shell,
        .auth-shell {{
            background: var(--auth-bg);
            border: 1px solid var(--line);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow);
        }}

        .brand-lockup {{
            display: flex;
            align-items: center;
            gap: 10px;
            color: var(--topbar-text);
        }}

        .brand-mark {{
            width: 34px;
            height: 34px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            background: rgba(17, 213, 196, 0.14);
            border: 1px solid rgba(17, 213, 196, 0.22);
            color: var(--accent);
            font-size: 1rem;
            font-weight: 800;
        }}

        .brand-title {{
            margin: 0;
            color: var(--topbar-text);
            font-size: 0.96rem;
            font-weight: 800;
            line-height: 1.1;
            letter-spacing: -0.02em;
        }}

        .brand-subtitle {{
            margin: 2px 0 0;
            color: var(--topbar-muted);
            font-size: 0.78rem;
            line-height: 1.2;
        }}

        .nav-links {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 18px;
            min-height: 34px;
            flex-wrap: nowrap;
            color: var(--topbar-muted);
            font-size: 0.86rem;
            font-weight: 700;
        }}

        .nav-links span {{
            white-space: nowrap;
        }}

        .nav-links .is-active {{
            color: var(--topbar-active);
            position: relative;
        }}

        .nav-links .is-active::after {{
            content: "";
            position: absolute;
            left: 50%;
            bottom: -12px;
            transform: translateX(-50%);
            width: 68px;
            height: 2px;
            border-radius: 999px;
            background: var(--accent);
        }}

        .theme-slot-label {{
            color: var(--topbar-muted);
            font-size: 0.74rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 0 0 8px;
        }}

        .theme-slot-label--topbar {{
            margin-top: 10px;
        }}

        .theme-inline-state {{
            display: flex;
            align-items: center;
            min-height: 44px;
            color: var(--topbar-text);
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.2;
            white-space: nowrap;
            margin-top: 0;
        }}

        .theme-slot-label--compact {{
            margin: 0 0 4px;
            font-size: 0.70rem;
            letter-spacing: 0.10em;
        }}

        .theme-inline-state--compact {{
            min-height: auto;
            font-size: 1.02rem;
        }}

        .theme-state-stack {{
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-self: center;
            min-height: 50px;
            width: 100%;
        }}

        .theme-state-stack__hint {{
            color: var(--topbar-muted);
            font-size: 0.74rem;
            line-height: 1.25;
            margin-top: 4px;
            margin-bottom: 0;
        }}

        .appearance-shell {{
            margin-bottom: 1rem;
            padding: 16px 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
        }}

        .appearance-shell__copy {{
            min-width: 0;
        }}

        .appearance-shell__kicker {{
            color: var(--topbar-muted);
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin: 0 0 4px;
        }}

        .appearance-shell__title {{
            color: var(--topbar-text);
            font-size: 1.02rem;
            font-weight: 800;
            line-height: 1.2;
            margin: 0;
        }}

        .appearance-shell__hint {{
            color: var(--topbar-muted);
            font-size: 0.8rem;
            line-height: 1.25;
            margin: 4px 0 0;
        }}

        .theme-mode-badge {{
            display: inline-flex;
            align-items: center;
            gap: 10px;
            color: var(--topbar-text);
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.2;
        }}

        .theme-mode-badge--label-only {{
            gap: 0;
        }}

        .theme-mode-label {{
            color: var(--topbar-text);
            white-space: nowrap;
        }}

        .theme-mode-icon {{
            width: 38px;
            height: 38px;
            border-radius: 12px;
            display: grid;
            place-items: center;
            background: var(--theme-icon-bg);
            border: 1px solid var(--theme-icon-border);
            color: var(--theme-icon-text);
            font-size: 1rem;
        }}

        div[data-testid="column"]:has(.theme-topbar-anchor) div.stButton {{
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0;
        }}

        div[data-testid="column"]:has(.theme-topbar-anchor) div.stButton > button {{
            min-width: 44px;
            width: 44px;
            height: 44px;
            padding: 0;
            border-radius: 12px;
            background: var(--theme-icon-bg);
            border: 1px solid var(--theme-icon-border);
            color: var(--theme-icon-text);
            font-size: 1rem;
            font-weight: 800;
        }}

        div[data-testid="column"]:has(.theme-topbar-anchor) div.stButton > button:hover {{
            background: var(--accent-soft);
            border-color: var(--accent);
            color: var(--accent-strong);
        }}

        div[data-testid="column"]:has(.theme-topbar-anchor) div.stButton > button p {{
            font-size: 1rem !important;
            line-height: 1 !important;
        }}

        div[data-testid="column"]:has(.theme-mode-toggle-anchor) div.stButton {{
            margin-top: 2px;
        }}

        div[data-testid="column"]:has(.theme-mode-toggle-anchor) div.stButton > button {{
            min-height: 44px;
            padding: 0 14px;
            border-radius: 10px;
            background: var(--mode-toggle-bg);
            border: 1px solid var(--mode-toggle-border);
            color: var(--mode-toggle-text);
            font-weight: 700;
            box-shadow: none;
        }}

        div[data-testid="column"]:has(.theme-mode-toggle-anchor) div.stButton > button:hover {{
            background: var(--mode-toggle-hover-bg);
            border-color: var(--mode-toggle-hover-border);
            color: var(--mode-toggle-hover-text);
        }}

        div[data-testid="column"]:has(.theme-mode-toggle-anchor) div.stButton > button p {{
            color: inherit !important;
            font-weight: inherit !important;
        }}

        .hero-shell {{
            padding: 34px 30px;
            margin-bottom: 1rem;
            text-align: center;
        }}

        .hero-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 12px;
            border-radius: 999px;
            background: rgba(17, 213, 196, 0.08);
            border: 1px solid var(--hero-line);
            color: var(--accent);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .section-badge {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 12px;
            border-radius: 999px;
            background: var(--badge-bg);
            border: 1px solid var(--badge-border);
            color: var(--badge-text);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .hero-shell h1,
        .steps-shell h1 {{
            margin: 14px 0 0;
            color: var(--hero-text);
            font-size: clamp(2.3rem, 5vw, 4rem);
            line-height: 1.05;
            letter-spacing: -0.04em;
            font-weight: 800;
        }}

        .hero-shell h1 .accent {{
            color: var(--accent);
        }}

        .hero-copy {{
            margin: 18px auto 0;
            max-width: 760px;
            color: var(--hero-muted);
            line-height: 1.7;
            font-size: 1.02rem;
        }}

        .hero-actions {{
            margin-top: 22px;
            display: flex;
            justify-content: center;
            gap: 14px;
            flex-wrap: wrap;
        }}

        .hero-btn {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 46px;
            padding: 0 24px;
            border-radius: 12px;
            border: 1px solid var(--tool-border);
            font-weight: 800;
            font-size: 0.95rem;
        }}

        .hero-btn.primary {{
            background: var(--accent);
            border-color: transparent;
            color: #062027;
        }}

        .hero-btn.ghost {{
            background: rgba(8, 20, 27, 0.36);
            color: var(--accent);
        }}

        .hero-benefits {{
            margin: 24px auto 2px;
            max-width: 860px;
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
        }}

        .hero-benefit {{
            border: 1px solid var(--hero-line);
            border-radius: 14px;
            padding: 14px 15px;
            text-align: left;
            background: rgba(255, 255, 255, 0.02);
            color: var(--hero-muted);
            font-size: 0.82rem;
            line-height: 1.45;
        }}

        .hero-benefit strong {{
            display: block;
            color: var(--hero-text);
            margin-bottom: 3px;
            font-size: 0.95rem;
        }}

        .steps-shell {{
            padding: 28px;
            margin-bottom: 1.2rem;
        }}

        .steps-head {{
            text-align: center;
            margin-bottom: 20px;
        }}

        .steps-head h2 {{
            margin: 14px 0 0;
            color: var(--hero-text);
            font-size: clamp(2rem, 4vw, 3.2rem);
            letter-spacing: -0.03em;
            line-height: 1.06;
        }}

        .steps-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
            margin-top: 4px;
        }}

        .step-card {{
            position: relative;
            min-height: 152px;
            border-radius: 16px;
            border: 1px solid var(--hero-line);
            background: linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.01));
            padding: 18px 16px 16px;
        }}

        .step-card::after {{
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            width: 46px;
            height: 30px;
            border-radius: 0 16px 0 16px;
            background: rgba(255,255,255,0.04);
        }}

        .step-icon {{
            width: 32px;
            height: 32px;
            border-radius: 10px;
            display: grid;
            place-items: center;
            background: rgba(17, 213, 196, 0.10);
            border: 1px solid rgba(17, 213, 196, 0.16);
            color: var(--accent);
            font-size: 0.88rem;
            margin-bottom: 14px;
        }}

        .step-kicker {{
            color: var(--accent);
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}

        .step-title {{
            color: var(--hero-text);
            font-size: 0.98rem;
            font-weight: 800;
            margin-bottom: 8px;
            line-height: 1.3;
        }}

        .step-copy {{
            color: var(--hero-muted);
            font-size: 0.82rem;
            line-height: 1.6;
        }}

        .parser-intro {{
            text-align: center;
            padding: 1rem 0 1.2rem;
        }}

        .parser-heading {{
            display: inline-flex;
            flex-direction: column;
            align-items: center;
        }}

        .parser-heading h2 {{
            margin: 14px 0 0;
            color: var(--display-heading);
            font-size: clamp(2rem, 4vw, 3rem);
            line-height: 1.06;
            letter-spacing: -0.04em;
            font-weight: 800;
        }}

        .parser-copy {{
            margin: 12px 0 0;
            max-width: 760px;
            color: var(--display-copy);
            line-height: 1.75;
            font-size: 0.97rem;
        }}

        .workspace-grid {{
            display: grid;
            grid-template-columns: minmax(260px, 0.9fr) minmax(0, 1.4fr);
            gap: 16px;
            align-items: start;
            margin-bottom: 1.2rem;
        }}

        .progress-shell {{
            padding: 18px;
            min-height: 350px;
        }}

        .progress-title {{
            color: var(--progress-title);
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 16px;
        }}

        .progress-steps {{
            display: flex;
            flex-direction: column;
            gap: 14px;
        }}

        .progress-step {{
            display: grid;
            grid-template-columns: 30px 1fr;
            gap: 12px;
            align-items: start;
        }}

        .progress-index {{
            width: 30px;
            height: 30px;
            border-radius: 999px;
            display: grid;
            place-items: center;
            font-size: 0.82rem;
            font-weight: 800;
            border: 1px solid var(--progress-border);
            color: var(--progress-copy);
            background: var(--hero-ghost);
        }}

        .progress-step.is-active .progress-index {{
            background: var(--accent);
            color: #082126;
            border-color: transparent;
            box-shadow: 0 0 0 6px rgba(17, 213, 196, 0.10);
        }}

        .progress-step-title {{
            color: var(--progress-subtle);
            font-size: 0.93rem;
            font-weight: 700;
            line-height: 1.2;
            margin-top: 4px;
        }}

        .progress-step-copy {{
            color: var(--progress-copy);
            font-size: 0.78rem;
            line-height: 1.5;
            margin-top: 4px;
        }}

        .progress-divider {{
            height: 1px;
            background: var(--progress-border);
            margin: 18px 0 14px;
        }}

        .progress-footer {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            color: var(--progress-copy);
            font-size: 0.82rem;
            font-weight: 700;
        }}

        .mini-pill {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 6px 12px;
            border-radius: 999px;
            background: var(--progress-pill-bg);
            color: var(--progress-pill-text);
            font-size: 0.76rem;
            font-weight: 800;
            min-width: 72px;
        }}

        .tool-shell {{
            padding: 18px;
        }}

        .tool-card {{
            border: 1px solid var(--line);
            background: var(--surface);
            border-radius: 18px;
            padding: 14px 16px;
            margin: 0 0 0.55rem;
            box-shadow: var(--shadow-soft);
        }}

        .tool-card__head {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 0;
        }}

        .tool-card__icon {{
            width: 28px;
            height: 28px;
            border-radius: 10px;
            display: grid;
            place-items: center;
            background: var(--tool-icon-bg);
            border: 1px solid var(--tool-icon-border);
            color: var(--tool-icon-text);
            font-size: 0.9rem;
            flex: none;
        }}

        .tool-card__title {{
            color: var(--tool-title);
            font-size: 0.94rem;
            font-weight: 800;
            line-height: 1.2;
        }}

        .tool-card__copy {{
            color: var(--tool-copy);
            font-size: 0.78rem;
            line-height: 1.45;
            margin-top: 2px;
        }}

        .file-chip-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 0.5rem 0 0.85rem;
        }}

        .file-chip {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border-radius: 999px;
            background: rgba(255,255,255,0.94);
            color: #1d323d;
            border: 1px solid rgba(17,39,51,0.10);
            font-size: 0.8rem;
            font-weight: 700;
        }}

        .file-chip.is-encrypted {{
            background: #fff4ea;
            color: #9a4e08;
            border-color: #f2c7a0;
        }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 14px;
            margin-bottom: 1rem;
        }}

        .metric-card {{
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 14px 15px;
            box-shadow: var(--shadow-soft);
        }}

        .metric-card__label {{
            color: var(--muted);
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}

        .metric-card__value {{
            color: var(--text-strong);
            font-size: 1.02rem;
            font-weight: 800;
            line-height: 1.35;
        }}

        .status-card {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            padding: 16px 18px;
            border-radius: 18px;
            border: 1px solid var(--line);
            background: var(--surface);
            margin-bottom: 1rem;
        }}

        .status-card__group {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .status-card__dot {{
            width: 12px;
            height: 12px;
            border-radius: 999px;
            background: var(--muted);
            box-shadow: 0 0 0 7px rgba(82,102,116,0.10);
            flex: none;
        }}

        .status-card.is-running .status-card__dot {{
            background: var(--accent);
            box-shadow: 0 0 0 7px rgba(17,213,196,0.12);
        }}

        .status-card.is-stopped .status-card__dot {{
            background: #f0a24b;
            box-shadow: 0 0 0 7px rgba(240,162,75,0.12);
        }}

        .status-card__title {{
            color: var(--text-strong);
            font-size: 0.95rem;
            font-weight: 800;
        }}

        .status-card__copy {{
            color: var(--muted);
            font-size: 0.84rem;
            line-height: 1.5;
            margin-top: 2px;
        }}

        .status-pill {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-width: 82px;
            padding: 8px 12px;
            border-radius: 999px;
            font-size: 0.77rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        .status-card.is-idle .status-pill {{
            background: var(--status-idle-bg);
            color: var(--status-idle-text);
        }}

        .status-card.is-running .status-pill {{
            background: var(--status-running-bg);
            color: var(--status-running-text);
        }}

        .status-card.is-stopped .status-pill {{
            background: var(--status-stopped-bg);
            color: var(--status-stopped-text);
        }}

        .results-shell,
        .download-shell {{
            padding: 18px;
            margin-bottom: 1rem;
        }}

        .section-head {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            padding: 16px 18px;
            margin: 0 0 14px;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: var(--surface);
            box-shadow: var(--shadow-soft);
        }}

        .section-title {{
            margin: 0;
            color: var(--text-strong);
            font-size: 1.08rem;
            font-weight: 800;
            letter-spacing: -0.02em;
        }}

        .section-copy {{
            margin: 0;
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.6;
        }}

        .auth-shell {{
            max-width: 760px;
            margin: 7vh auto 0;
            padding: 24px 24px 20px;
        }}

        .auth-shell__logo {{
            margin-bottom: 18px;
        }}

        .auth-shell h1 {{
            margin: 14px 0 0;
            color: var(--auth-heading);
            font-size: clamp(1.8rem, 4vw, 2.5rem);
            line-height: 1.08;
            letter-spacing: -0.04em;
            font-weight: 800;
        }}

        .auth-copy {{
            margin: 12px 0 0;
            color: var(--auth-copy);
            line-height: 1.75;
            font-size: 0.98rem;
        }}

        .auth-footer-note {{
            margin-top: 12px;
            color: var(--muted);
            text-align: center;
            font-size: 0.88rem;
        }}

        div[data-testid="stForm"] {{
            background: var(--surface);
            border: 1px solid var(--line);
            box-shadow: var(--shadow-soft);
            padding: 20px 18px 18px;
            margin: 1rem auto 0;
            max-width: 760px;
            border-radius: 22px;
        }}

        div[data-testid="stWidgetLabel"] p,
        div[data-testid="stTextInput"] label p,
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stFileUploader"] label p,
        div[data-testid="stTextArea"] label p {{
            color: var(--form-label) !important;
            font-size: 0.92rem;
            font-weight: 700 !important;
            opacity: 1 !important;
        }}

        .tool-shell div[data-testid="stWidgetLabel"] p,
        .tool-shell div[data-testid="stTextInput"] label p,
        .tool-shell div[data-testid="stSelectbox"] label p,
        .tool-shell div[data-testid="stFileUploader"] label p,
        .tool-shell div[data-testid="stTextArea"] label p {{
            color: var(--tool-title) !important;
            font-size: 0.92rem !important;
            font-weight: 700 !important;
            opacity: 1 !important;
        }}

        .tool-shell [data-testid="stMarkdownContainer"] p,
        .tool-shell small {{
            color: var(--tool-copy);
        }}

        div[data-baseweb="input"],
        div[data-baseweb="select"] > div {{
            border-radius: var(--radius-md) !important;
        }}

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {{
            min-height: 52px;
            border: 1px solid var(--input-border);
            background: var(--input-bg);
            box-shadow: none;
            transition: border-color 160ms ease, box-shadow 160ms ease, background 160ms ease;
        }}

        div[data-baseweb="input"] > div:hover,
        div[data-baseweb="select"] > div:hover {{
            border-color: var(--line-strong);
        }}

        div[data-baseweb="input"] > div:focus-within,
        div[data-baseweb="select"] > div:focus-within {{
            border-color: var(--accent);
            box-shadow: 0 0 0 3px rgba(17, 213, 196, 0.14);
        }}

        div[data-baseweb="input"] input,
        div[data-baseweb="select"] input,
        div[data-baseweb="select"] span,
        div[data-baseweb="textarea"] textarea {{
            color: var(--input-text) !important;
            -webkit-text-fill-color: var(--input-text) !important;
        }}

        div[data-baseweb="input"] input::placeholder,
        div[data-baseweb="textarea"] textarea::placeholder {{
            color: var(--placeholder) !important;
            opacity: 1 !important;
        }}

        .tool-shell div[data-baseweb="input"] > div,
        .tool-shell div[data-baseweb="select"] > div {{
            background: var(--tool-input-bg);
            border-color: var(--tool-input-border);
        }}

        .tool-shell div[data-baseweb="input"] input,
        .tool-shell div[data-baseweb="select"] input,
        .tool-shell div[data-baseweb="select"] span {{
            color: var(--tool-input-text) !important;
            -webkit-text-fill-color: var(--tool-input-text) !important;
        }}

        .tool-shell div[data-baseweb="input"] input::placeholder {{
            color: var(--tool-placeholder) !important;
        }}

        div[data-testid="stSelectbox"],
        div[data-testid="stTextInput"],
        div[data-testid="stTextArea"],
        div[data-testid="stFileUploader"] {{
            margin-bottom: 1rem;
        }}

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-baseweb="textarea"] > div {{
            background: var(--surface) !important;
            border-color: var(--line-strong) !important;
            box-shadow: var(--shadow-soft);
        }}

        div[data-baseweb="select"] *,
        div[data-baseweb="input"] *,
        div[data-baseweb="textarea"] * {{
            color: var(--input-text) !important;
            -webkit-text-fill-color: var(--input-text) !important;
            opacity: 1 !important;
        }}

        div[data-baseweb="select"] svg,
        div[data-baseweb="select"] path {{
            fill: var(--input-text) !important;
            color: var(--input-text) !important;
            stroke: var(--input-text) !important;
            opacity: 0.78;
        }}

        div[data-baseweb="select"] > div > div {{
            color: var(--input-text) !important;
        }}

        body div[data-baseweb="popover"] {{
            background: transparent !important;
        }}

        body div[data-baseweb="popover"] > div,
        body div[data-baseweb="popover"] > div > div,
        body div[data-baseweb="popover"] > div > div > div,
        body div[data-baseweb="popover"] div[role="presentation"],
        body div[data-baseweb="menu"],
        body ul[role="listbox"],
        body div[role="listbox"] {{
            background: var(--select-menu-bg) !important;
            background-color: var(--select-menu-bg) !important;
            border: 1px solid var(--select-menu-border) !important;
            border-radius: 16px !important;
            box-shadow: var(--select-menu-shadow) !important;
        }}

        body ul[role="listbox"],
        body div[role="listbox"],
        body div[data-baseweb="menu"] {{
            background: var(--select-menu-surface) !important;
            background-color: var(--select-menu-surface) !important;
            padding: 8px !important;
            overflow: hidden !important;
        }}

        body div[role="option"],
        body li[role="option"] {{
            background: var(--select-menu-row-bg) !important;
            background-color: var(--select-menu-row-bg) !important;
            border-radius: 12px !important;
            margin: 2px 0 !important;
        }}

        body div[role="option"],
        body li[role="option"],
        body div[role="option"] *,
        body li[role="option"] *,
        body div[role="option"] span,
        body li[role="option"] span,
        body div[role="option"] p,
        body li[role="option"] p {{
            color: var(--select-menu-text) !important;
            -webkit-text-fill-color: var(--select-menu-text) !important;
            opacity: 1 !important;
        }}

        body div[role="option"]:hover,
        body li[role="option"]:hover,
        body div[role="option"][aria-selected="true"],
        body li[role="option"][aria-selected="true"],
        body div[role="option"][data-highlighted="true"],
        body li[role="option"][data-highlighted="true"] {{
            background: var(--select-menu-hover-bg) !important;
            background-color: var(--select-menu-hover-bg) !important;
            color: var(--select-menu-hover-text) !important;
        }}

        body div[role="option"]:hover *,
        body li[role="option"]:hover *,
        body div[role="option"][aria-selected="true"] *,
        body li[role="option"][aria-selected="true"] *,
        body div[role="option"][data-highlighted="true"] *,
        body li[role="option"][data-highlighted="true"] * {{
            color: var(--select-menu-hover-text) !important;
            -webkit-text-fill-color: var(--select-menu-hover-text) !important;
        }}

        body ul[role="listbox"]::-webkit-scrollbar,
        body div[role="listbox"]::-webkit-scrollbar {{
            width: 10px;
        }}

        body ul[role="listbox"]::-webkit-scrollbar-thumb,
        body div[role="listbox"]::-webkit-scrollbar-thumb {{
            background: var(--line-strong);
            border-radius: 999px;
        }}

        div[data-testid="stFileUploader"] > section {{
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: var(--shadow-soft);
        }}

        div[data-testid="stFileUploaderFileList"] {{
            background: transparent;
        }}

        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] p {{
            color: var(--muted);
        }}

        div[data-testid="stFileUploaderDropzone"] {{
            border: 1.5px dashed var(--line-strong);
            border-radius: 16px;
            background: var(--input-bg);
            padding: 18px;
        }}

        .tool-shell div[data-testid="stFileUploader"] > section {{
            background: var(--tool-uploader-shell-bg);
            border: 1px solid var(--tool-uploader-shell-border);
            border-radius: 14px;
            padding: 12px 14px;
            box-shadow: none;
        }}

        .tool-shell div[data-testid="stFileUploaderFileList"] {{
            background: transparent;
        }}

        .tool-shell div[data-testid="stFileUploader"] small,
        .tool-shell div[data-testid="stFileUploader"] span,
        .tool-shell div[data-testid="stFileUploader"] p {{
            color: var(--tool-uploader-copy);
        }}

        .tool-shell div[data-testid="stFileUploaderDropzone"] {{
            background: var(--tool-input-bg);
            border-color: var(--tool-input-border);
        }}

        div[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p {{
            color: var(--muted);
        }}

        .tool-shell div[data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"] p {{
            color: var(--tool-copy);
        }}

        div[data-testid="stFileUploader"] section button,
        div.stButton > button,
        div.stDownloadButton > button,
        div[data-testid="stFormSubmitButton"] > button {{
            min-height: 46px;
            border-radius: 12px;
            font-weight: 700;
            border: 1px solid var(--line-strong);
            background: var(--surface);
            color: var(--text-strong);
            box-shadow: var(--shadow-soft);
        }}

        .tool-shell div.stButton > button,
        .tool-shell div[data-testid="stFileUploader"] section button {{
            background: var(--tool-button-bg);
            color: var(--tool-button-text);
            border-color: var(--tool-button-border);
        }}

        div.stButton > button:hover,
        div.stDownloadButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover,
        div[data-testid="stFileUploader"] section button:hover {{
            border-color: var(--accent);
            color: var(--text-strong);
        }}

        .tool-shell div.stButton > button:hover,
        .tool-shell div[data-testid="stFileUploader"] section button:hover {{
            color: var(--tool-title);
            background: var(--tool-button-hover-bg);
        }}

        div.stButton > button[kind="primary"],
        div[data-testid="stFormSubmitButton"] > button[kind="primary"] {{
            background: var(--accent);
            color: #082126;
            border-color: transparent;
        }}

        .tool-shell div.stButton > button[kind="primary"] {{
            background: var(--tool-primary-bg);
            color: var(--tool-primary-text);
            border-color: transparent;
        }}

        div[data-testid="stDataFrame"] {{
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid var(--line);
            background: var(--table-bg);
        }}

        div[data-testid="stDataFrame"] [role="grid"] {{
            background: var(--table-bg);
            color: var(--table-text);
        }}

        div[data-testid="stAlert"] {{
            border-radius: 14px;
            border: 1px solid var(--line);
        }}

        div[data-testid="stProgressBar"] > div > div {{
            background-color: rgba(17, 213, 196, 0.16);
        }}

        div[data-testid="stProgressBar"] div[role="progressbar"] {{
            background-color: var(--accent);
        }}

        @media (max-width: 980px) {{
            .steps-grid,
            .workspace-grid {{
                grid-template-columns: 1fr 1fr;
            }}

            .hero-benefits {{
                grid-template-columns: 1fr;
            }}
        }}

        @media (max-width: 760px) {{
            .block-container {{
                padding-top: 0.8rem;
            }}

            .topbar-shell,
            .theme-toggle-shell,
            .hero-shell,
            .steps-shell,
            .progress-shell,
            .tool-shell,
            .results-shell,
            .download-shell,
            .auth-shell {{
                padding: 16px;
            }}

            .steps-grid,
            .workspace-grid {{
                grid-template-columns: 1fr;
            }}

            .hero-shell h1,
            .parser-heading h2 {{
                font-size: 2.2rem;
            }}

            .nav-links {{
                justify-content: flex-start;
                gap: 14px;
                flex-wrap: wrap;
            }}
        }}
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
                    <div class="brand-subtitle">Bank statement parser</div>
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
                <span>How it works</span>
                <span>Features</span>
                <span>FAQ</span>
                <span class="is-active">Contact</span>
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
                <p class="appearance-shell__title">Get Started</p>
                <p class="appearance-shell__hint">Upload a statement and run parser</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    mode_button_label = "☀️ Light mode" if is_light else "🌙 Dark mode"
    st.markdown('<div class="theme-mode-toggle-anchor"></div>', unsafe_allow_html=True)
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
            <h1>Access the parser workspace</h1>
            <p class="auth-copy">Sign in to continue to the parser workspace. The visual design is refreshed, while the authentication and parser functionality remain unchanged.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_app_hero() -> None:
    st.markdown(
        """
        <section class="hero-shell">
            <span class="hero-badge">Bank statement parser · Powered by KreditLab</span>
            <h1>Turn Bank Statements Into <span class="accent">Clear Financial Insights</span></h1>
            <p class="hero-copy">Upload any bank statement PDF and get structured transaction data, summaries, and export-ready reports in seconds.</p>
            <div class="hero-actions">
                <span class="hero-btn primary">⇪&nbsp; Get Started — Upload Now</span>
                <span class="hero-btn ghost">See How It Works →</span>
            </div>
            <div class="hero-benefits">
                <div class="hero-benefit"><strong>Multi-Bank Support</strong>Parse statements from all major Malaysian banks instantly.</div>
                <div class="hero-benefit"><strong>Secure Processing</strong>Bank-grade handling with privacy-first processing.</div>
                <div class="hero-benefit"><strong>Instant Insights</strong>Extract key financial patterns and cash-flow trends quickly.</div>
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
                <span class="section-badge">How it works</span>
                <h2>Four steps to financial clarity</h2>
            </div>
            <div class="steps-grid">
                <div class="step-card">
                    <div class="step-icon">▣</div>
                    <div class="step-kicker">Step 1</div>
                    <div class="step-title">Select Your Bank</div>
                    <div class="step-copy">Choose the bank that issued your statement from the supported list.</div>
                </div>
                <div class="step-card">
                    <div class="step-icon">⤴</div>
                    <div class="step-kicker">Step 2</div>
                    <div class="step-title">Upload Statement</div>
                    <div class="step-copy">Drag and drop or browse your PDF bank statement file.</div>
                </div>
                <div class="step-card">
                    <div class="step-icon">∿</div>
                    <div class="step-kicker">Step 3</div>
                    <div class="step-title">Process & Analyse</div>
                    <div class="step-copy">The engine extracts and structures the transaction data automatically.</div>
                </div>
                <div class="step-card">
                    <div class="step-icon">▥</div>
                    <div class="step-kicker">Step 4</div>
                    <div class="step-title">View Results</div>
                    <div class="step-copy">Inspect transactions, summaries, and export-ready outputs.</div>
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
                <h2>Upload & Parse Your Statement</h2>
                <p class="parser-copy">Select your bank, upload the PDF statement, and let the parser extract structured financial data into review-ready outputs.</p>
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
