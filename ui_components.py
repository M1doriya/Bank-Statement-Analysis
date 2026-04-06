import html
import inspect
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import streamlit as st

_TEMPLATE_DIR = Path(__file__).resolve().parent / "template"


@lru_cache(maxsize=None)
def _read_template(name: str) -> str:
    return (_TEMPLATE_DIR / name).read_text(encoding="utf-8")


@lru_cache(maxsize=1)
def _converted_style_block() -> str:
    template = _read_template("ui_components_converted.html")
    match = re.search(r"<style>(.*?)</style>", template, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError("template/ui_components_converted.html is missing a <style> block")
    return match.group(1)


@lru_cache(maxsize=1)
def _section_map() -> dict:
    raw = _read_template("ui_sections.html")
    pattern = re.compile(
        r"<!--\s*SECTION:(?P<name>[a-z0-9_\-]+)\s*-->(?P<body>.*?)<!--\s*ENDSECTION:\1\s*-->",
        re.DOTALL | re.IGNORECASE,
    )
    sections = {m.group("name"): m.group("body").strip() for m in pattern.finditer(raw)}
    if not sections:
        raise ValueError("No sections found in template/ui_sections.html")
    return sections


def _render_section(name: str, **context: str) -> str:
    sections = _section_map()
    if name not in sections:
        raise KeyError(f"Section '{name}' not found in template/ui_sections.html")
    return sections[name].format(**context)


def inject_global_styles(theme_mode: str = "Dark") -> None:
    theme = "light" if str(theme_mode or "Dark").strip().lower() == "light" else "dark"
    css = _converted_style_block()
    st.markdown(
        f"""
        <style>{css}</style>
        <script>
            const __setTheme = () => {{
                document.body.setAttribute('data-theme', '{theme}');
            }};
            __setTheme();
            window.addEventListener('load', __setTheme, {{ once: true }});
        </script>
        """,
        unsafe_allow_html=True,
    )


def render_top_bar() -> None:
    st.markdown('<div class="topbar-row-anchor"></div>', unsafe_allow_html=True)
    left, middle, right = columns_compat([1.32, 1.38, 1.30], gap="medium", vertical_alignment="center")

    left.markdown(_render_section("topbar_brand"), unsafe_allow_html=True)
    middle.markdown(_render_section("topbar_nav"), unsafe_allow_html=True)

    is_light = st.session_state.get("ui_theme_light", False)
    theme_state = "Light mode" if is_light else "Dark mode"
    right.markdown(_render_section("topbar_appearance", theme_state=html.escape(theme_state)), unsafe_allow_html=True)

    mode_button_label = "☀️ Light mode" if is_light else "🌙 Dark mode"
    st.markdown('<div class="theme-mode-toggle-anchor"></div>', unsafe_allow_html=True)
    if st.button(mode_button_label, key="theme_mode_button", help="Switch between light and dark interface modes"):
        st.session_state.ui_theme_light = not is_light
        st.session_state.ui_theme_mode = "Light" if st.session_state.ui_theme_light else "Dark"
        st.rerun()


def render_auth_shell() -> None:
    st.markdown(_render_section("auth_shell"), unsafe_allow_html=True)


def render_app_hero() -> None:
    st.markdown(_render_section("app_hero"), unsafe_allow_html=True)


def render_steps_showcase() -> None:
    st.markdown(_render_section("steps_showcase"), unsafe_allow_html=True)


def render_parser_intro() -> None:
    st.markdown(_render_section("parser_intro"), unsafe_allow_html=True)


def render_tool_card_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(
        _render_section(
            "tool_card_header",
            icon=html.escape(icon),
            title=html.escape(title),
            subtitle=html.escape(subtitle),
        ),
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
        _render_section(
            "progress_step",
            active_class=" is-active" if idx == current_step else "",
            idx=str(idx),
            title=html.escape(title),
            copy=html.escape(copy),
        )
        for idx, (title, copy) in enumerate(steps, start=1)
    )
    st.markdown(
        _render_section("progress_panel", steps_html=steps_html, status_label=html.escape(status_label)),
        unsafe_allow_html=True,
    )


def render_section_header(label: str, title: str, subtitle: str) -> None:
    st.markdown(
        _render_section(
            "section_header",
            label=html.escape(label),
            title=html.escape(title),
            subtitle=html.escape(subtitle),
        ),
        unsafe_allow_html=True,
    )


def render_status_card(status: str) -> None:
    status_key = (status or "idle").strip().lower()
    status_copy = {
        "idle": "Ready to accept uploads and begin parsing.",
        "running": "Processing uploaded statements and generating outputs.",
        "stopped": "Run paused. You can resume or reset the workspace.",
    }
    status_label = {"idle": "Idle", "running": "Running", "stopped": "Stopped"}.get(status_key, status.upper())
    st.markdown(
        _render_section(
            "status_card",
            status_key=html.escape(status_key),
            status_copy=html.escape(status_copy.get(status_key, "Workspace updated.")),
            status_label=html.escape(status_label),
        ),
        unsafe_allow_html=True,
    )


def render_file_chips(uploaded_files: List, encrypted_files: List[str]) -> None:
    if not uploaded_files:
        return
    encrypted_set = set(encrypted_files or [])
    chips = []
    for uploaded_file in uploaded_files:
        name = getattr(uploaded_file, "name", str(uploaded_file))
        chips.append(
            _render_section(
                "file_chip",
                extra_class=" is-encrypted" if name in encrypted_set else "",
                icon="🔒" if name in encrypted_set else "📎",
                name=html.escape(name),
            )
        )
    st.markdown(_render_section("file_chip_row", chips_html="".join(chips)), unsafe_allow_html=True)


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
        _render_section("metric_card", label=html.escape(label), value=html.escape(value))
        for label, value in items
    )
    st.markdown(_render_section("metric_grid", cards_html=cards_html), unsafe_allow_html=True)
