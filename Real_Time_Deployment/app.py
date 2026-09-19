# =============================================================================
# RT-DeepNIDS — app.py  (Frontend / Streamlit entry point)
# A Real-Time Hybrid Network Intrusion Detection System
# Bangladesh University of Business and Technology (BUBT)
# Department of Computer Science and Engineering

# Run:  streamlit run app.py
# =============================================================================


import os
import html as html_mod
from datetime import datetime


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# All ML / networking / feature-engineering logic lives in backend.py
from backend import (
   IS_ADMIN, SCAPY_AVAILABLE, DL_MODELS, DATASET_COLS,
   EXCLUDE_COLS, MODEL_FILE_MAP,
   FlowTracker, PortScanDetector,
   get_friendly_interfaces, load_framework_assets, load_csv_source,
   get_feature_count, clean_label, safe_int,
   resolve_dataset_dir, resolve_model_file, set_min_flow_sec,
)


# Scapy layers — imported only when Scapy is available
if SCAPY_AVAILABLE:
   from scapy.all import sniff, IP, TCP, UDP

# PAGE CONFIG
st.set_page_config(
   page_title="RT-DeepNIDS",
   layout="wide",
   initial_sidebar_state="expanded",
)

# THEME SYSTEM
if "theme" not in st.session_state:
   st.session_state.theme = "light"


T_DARK = {
   "page_bg":         "#050810",
   "sidebar_bg":      "linear-gradient(180deg,#080d1a 0%,#050810 100%)",
   "sidebar_border":  "#1e3a5f",
   "card_bg":         "linear-gradient(145deg,#0f1729 0%,#0a1020 60%,#060c18 100%)",
   "card_border":     "#1e3a5f",
   "card_border_t":   "#2a4a7f",
   "card_shadow":     "0 8px 32px rgba(0,0,0,0.6),0 2px 8px rgba(14,165,233,0.08),inset 0 1px 0 rgba(255,255,255,0.04)",
   "label_color":     "#64748b",
   "value_color":     "#e2e8f0",
   "text_pri":        "#e2e8f0",
   "text_sec":        "#94a3b8",
   "text_muted":      "#475569",
   "accent":          "#38bdf8",
   "accent_dim":      "rgba(56,189,248,0.08)",
   "border":          "#1e3a5f",
   "grid":            "rgba(30,58,95,0.5)",
   "tick":            "#475569",
   "threat_box_bg":   "linear-gradient(145deg,#090e1c,#060b16)",
   "progress_track":  "#0f1a2e",
   "tab_border":      "#1e3a5f",
   "tab_color":       "#64748b",
   "input_bg":        "#0a1428",
   "input_border":    "#1e3a5f",
   "nav_bg":          "#050810",
   "nav_border":      "#1e3a5f",
   "nav_text":        "#e2e8f0",
   "btn_start_bg":    "linear-gradient(145deg,#0ea5e9,#0284c7)",
   "btn_start_color": "#ffffff",
   "btn_start_shadow":"0 4px 12px rgba(14,165,233,0.4),inset 0 1px 0 rgba(255,255,255,0.15)",
   "btn_stop_bg":     "linear-gradient(145deg,#1e293b,#0f172a)",
   "btn_stop_color":  "#94a3b8",
   "btn_stop_border": "#1e3a5f",
   "btn_reset_bg":    "#0a1428",
   "btn_reset_color": "#64748b",
   "btn_reset_border":"#1e3a5f",
   "df_bg":           "#080d1a",
   "df_border":       "#1e3a5f",
   "radio_color":     "#94a3b8",
}


T_LIGHT = {
   "page_bg":         "#f1f5f9",
   "sidebar_bg":      "linear-gradient(180deg,#ffffff 0%,#f8fafc 100%)",
   "sidebar_border":  "#cbd5e1",
   "card_bg":         "linear-gradient(145deg,#ffffff 0%,#f8fafc 60%,#f1f5f9 100%)",
   "card_border":     "#e2e8f0",
   "card_border_t":   "#cbd5e1",
   "card_shadow":     "0 4px 16px rgba(0,0,0,0.06),0 1px 4px rgba(14,165,233,0.04),inset 0 1px 0 rgba(255,255,255,0.9)",
   "label_color":     "#64748b",
   "value_color":     "#0f172a",
   "text_pri":        "#0f172a",
   "text_sec":        "#334155",
   "text_muted":      "#64748b",
   "accent":          "#0284c7",
   "accent_dim":      "rgba(2,132,199,0.08)",
   "border":          "#e2e8f0",
   "grid":            "rgba(148,163,184,0.25)",
   "tick":            "#94a3b8",
   "threat_box_bg":   "linear-gradient(145deg,#ffffff,#f8fafc)",
   "progress_track":  "#e2e8f0",
   "tab_border":      "#e2e8f0",
   "tab_color":       "#64748b",
   "input_bg":        "#ffffff",
   "input_border":    "#cbd5e1",
   "nav_bg":          "#ffffff",
   "nav_border":      "#e2e8f0",
   "nav_text":        "#0f172a",
   "btn_start_bg":    "linear-gradient(145deg,#0ea5e9,#0284c7)",
   "btn_start_color": "#ffffff",
   "btn_start_shadow":"0 4px 12px rgba(14,165,233,0.3),inset 0 1px 0 rgba(255,255,255,0.2)",
   "btn_stop_bg":     "linear-gradient(145deg,#f1f5f9,#e2e8f0)",
   "btn_stop_color":  "#475569",
   "btn_stop_border": "#cbd5e1",
   "btn_reset_bg":    "#f8fafc",
   "btn_reset_color": "#64748b",
   "btn_reset_border":"#e2e8f0",
   "df_bg":           "#ffffff",
   "df_border":       "#e2e8f0",
   "radio_color":     "#334155",
}


C = T_DARK if st.session_state.theme == "dark" else T_LIGHT

# GLOBAL CSS
ICON_SHIELD = (
   '<svg viewBox="0 0 24 24" width="100%" height="100%" fill="none" stroke="currentColor" '
   'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
   '<path d="M12 2 4 5v6c0 5 3.5 8.5 8 11 4.5-2.5 8-6 8-11V5l-8-3z"/>'
   '<path d="m9 12 2 2 4-4"/></svg>'
)

# GLOBAL CSS
def _build_css(c: dict) -> str:
   return f"""<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600;700&family=Exo+2:wght@300;400;600;700;800&display=swap');


/* --- SUPPRESS STREAMLIT CHROME (keep header + menu visible) --- */
[data-testid="stStatusWidget"]{{visibility:hidden!important;display:none!important;}}
[data-testid="stDecoration"]{{display:none!important;}}
/* Make the native header transparent so it blends with the page,
   but keep the three-dot menu and sidebar controls usable. */
[data-testid="stHeader"]{{background:transparent!important;height:2.5rem!important;}}
[data-testid="stToolbar"]{{right:1rem!important;}}


/* --- PREVENT RERUN FLASH / OPACITY SHAKE --- */
div[data-stale],.stElementContainer,.stPlotlyChart,.stDataFrame,
div[data-testid="stMarkdownContainer"],
[data-testid="stApp"],[data-testid="stAppViewContainer"],
[data-testid="stMain"],.main{{
   opacity:1!important;filter:none!important;
   transition:none!important;animation:none!important;transform:none!important;
}}
div[data-stale="true"],div[data-stale="true"] *{{
   opacity:1!important;filter:none!important;
   transition:none!important;animation:none!important;
}}


/* --- GLOBAL --- */
html,body{{background-color:{c["page_bg"]}!important;overflow-x:hidden!important;}}
[data-testid="stAppViewContainer"]{{background-color:{c["page_bg"]}!important;}}
[data-testid="stAppViewContainer"] *{{
   color:{c["text_pri"]}!important;
   font-family:'Exo 2',sans-serif!important;
}}
/* Don't override the icon font on Material Symbols glyphs — otherwise the
   sidebar/header icons fall back to showing their ligature names as text. */
[data-testid="stAppViewContainer"] [class*="material-symbols"],
[data-testid="stAppViewContainer"] [class*="material-icons"],
[data-testid="stAppViewContainer"] .material-symbols-rounded,
[data-testid="stAppViewContainer"] span[data-testid="stIconMaterial"],
[data-testid="stIconMaterial"]{{
   font-family:'Material Symbols Rounded','Material Symbols Outlined',
               'Material Icons'!important;
}}
.main .block-container{{
   padding-top:0.5rem!important;
   padding-bottom:5rem!important;
   background:transparent!important;
   max-width:100%!important;
}}
/* Kill the top margin/padding on the very first block so content starts
   right under the slim header instead of leaving a big empty gap. */
.main .block-container > div:first-child{{margin-top:0!important;padding-top:0!important;}}
[data-testid="stMain"] [data-testid="stVerticalBlock"]{{gap:0.75rem!important;}}


/* --- RESPONSIVE COLUMNS --- */
@media (max-width: 992px) {{
   /* Tablet: landing hero + cards breathe a little less */
   .lp-wrap{{padding:24px 24px 28px!important;}}
   .lp-title{{font-size:26px!important;letter-spacing:2px!important;}}
}}
@media (max-width: 768px) {{
   .main .block-container{{padding-left:0.75rem!important;padding-right:0.75rem!important;}}
   [data-testid="stMetric"]{{height:auto!important;min-height:90px!important;}}
   /* KPI custom cards: two columns on tablets/large phones */
   .kpi-row{{grid-template-columns:repeat(2,1fr)!important;}}
   /* Landing hero scales down */
   .lp-wrap{{padding:20px 16px 24px!important;}}
   .lp-title{{font-size:22px!important;letter-spacing:1.5px!important;}}
   .lp-subtitle{{font-size:12px!important;}}
   .lp-shield{{width:46px!important;height:46px!important;}}
   /* Pipeline steps stack into a tidy wrap; hide the inline arrows */
   .pipeline{{padding:16px!important;}}
   .p-arrow{{display:none!important;}}
   .p-step{{min-width:42%!important;}}
   /* Dashboard header: let the status block drop below the title cleanly */
   .nids-header{{padding:14px 16px!important;}}
   /* Contribution + team cards already wrap via auto-fit/flex */
}}
@media (max-width: 640px) {{
   /* Force Streamlit horizontal column groups (charts, START/STOP) to stack */
   [data-testid="stHorizontalBlock"]{{flex-direction:column!important;}}
   [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]{{width:100%!important;flex:1 1 100%!important;}}
   /* KPI custom cards: single column on small phones */
   .kpi-row{{grid-template-columns:1fr!important;}}
   /* Tables can scroll horizontally instead of squashing */
   [data-testid="stDataFrame"]{{overflow-x:auto!important;}}
   /* Tabs wrap instead of overflowing */
   .stTabs [data-baseweb="tab-list"]{{flex-wrap:wrap!important;}}
   .stTabs [data-baseweb="tab"]{{padding:8px 12px!important;font-size:12px!important;}}
}}


/* --- KPI METRIC CARDS --- */
[data-testid="stMetric"]{{
   background:{c["card_bg"]}!important;
   border:1px solid {c["card_border"]}!important;
   border-top:2px solid {c["card_border_t"]}!important;
   border-radius:10px!important;
   padding:16px 18px 16px 22px!important;
   height:110px!important;
   box-shadow:{c["card_shadow"]}!important;
   transform:translateZ(0);
   position:relative!important;overflow:hidden!important;
   transition:transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease!important;
}}
[data-testid="stMetric"]::before{{
   content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
   background:linear-gradient(180deg,{c["accent"]},transparent);
   opacity:0.85;
}}
[data-testid="stMetric"]:hover{{
   transform:translateY(-3px)!important;
   border-color:{c["accent"]}!important;
   box-shadow:{c["card_shadow"]},0 10px 28px {c["accent_dim"]}!important;
}}
[data-testid="stMetricLabel"]>div{{
   font-size:10px!important;font-weight:700!important;
   color:{c["label_color"]}!important;text-transform:uppercase!important;
   letter-spacing:1.2px!important;font-family:'JetBrains Mono',monospace!important;
}}
[data-testid="stMetricValue"]>div{{
   font-size:28px!important;font-weight:800!important;
   color:{c["value_color"]}!important;font-family:'Rajdhani',sans-serif!important;
   letter-spacing:1px!important;
}}


/* --- SIDEBAR --- */
[data-testid="stSidebar"]{{
   background:{c["sidebar_bg"]}!important;
   border-right:1px solid {c["sidebar_border"]}!important;
   box-shadow:4px 0 24px rgba(0,0,0,0.08)!important;
}}
[data-testid="stSidebar"] *{{color:{c["text_sec"]}!important;}}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3{{color:{c["text_pri"]}!important;}}
[data-testid="stSidebar"] hr{{border-color:{c["border"]}!important;opacity:0.6!important;}}

/* Sidebar header card */
.sb-head{{
   position:relative;overflow:hidden;border-radius:12px;
   padding:16px 14px 14px;margin-bottom:14px;text-align:center;
   background:{c["card_bg"]};border:1px solid {c["card_border"]};
   border-top:2px solid {c["card_border_t"]};box-shadow:{c["card_shadow"]};
}}
.sb-head::before{{
   content:'';position:absolute;top:0;left:0;right:0;height:2px;
   background:linear-gradient(90deg,transparent,{c["accent"]},transparent);opacity:0.6;
}}
.sb-shield{{
   width:38px;height:38px;margin:0 auto 8px;border-radius:10px;
   display:flex;align-items:center;justify-content:center;color:{c["accent"]};
   background:{c["accent_dim"]};border:1px solid {c["accent"]}55;
}}
.sb-shield svg{{width:20px;height:20px;display:block;}}
.sb-title{{
   font-family:'Rajdhani',sans-serif;font-size:18px;font-weight:800;
   color:{c["text_pri"]}!important;text-transform:uppercase;letter-spacing:2px;line-height:1.1;
}}
.sb-sub{{
   font-size:8.5px;color:{c["accent"]}!important;font-weight:700;letter-spacing:2.5px;
   font-family:'JetBrains Mono',monospace;margin-top:4px;
}}

/* Numbered section heading */
.sb-sec{{
   display:flex;align-items:center;gap:9px;margin:6px 0 2px;
}}
.sb-sec-num{{
   flex-shrink:0;width:20px;height:20px;border-radius:6px;
   display:flex;align-items:center;justify-content:center;
   background:{c["accent_dim"]};border:1px solid {c["accent"]}44;
   font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;
   color:{c["accent"]}!important;
}}
.sb-sec-txt{{
   font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;
   letter-spacing:1.5px;text-transform:uppercase;color:{c["text_sec"]}!important;
}}
.sb-sec-line{{
   flex:1;height:1px;background:linear-gradient(90deg,{c["border"]},transparent);
}}

/* Sidebar footer */
.sb-foot{{
   margin-top:16px;padding-top:12px;border-top:1px solid {c["border"]};
   text-align:center;
}}
.sb-foot-t{{
   font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
   letter-spacing:1.5px;color:{c["text_muted"]}!important;text-transform:uppercase;
}}
.sb-foot-s{{
   font-family:'Exo 2',sans-serif;font-size:9.5px;color:{c["text_muted"]}!important;
   margin-top:3px;
}}


/* --- SIDEBAR RADIO --- */
[data-testid="stSidebar"] [data-testid="stRadio"] label{{color:{c["radio_color"]}!important;}}
[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{{color:{c["accent"]}!important;}}
[data-testid="stSidebar"] [role="radio"][aria-checked="true"] + div{{
   color:{c["accent"]}!important;font-weight:600!important;
}}


/* --- SIDEBAR SELECT / TEXT INPUT --- */
[data-testid="stSidebar"] [data-baseweb="select"]>div,
[data-testid="stSidebar"] [data-testid="stTextInput"] input{{
   background:{c["input_bg"]}!important;
   border:1px solid {c["input_border"]}!important;
   color:{c["text_pri"]}!important;border-radius:6px!important;
}}
[data-testid="stSidebar"] [data-baseweb="select"]>div:hover,
[data-testid="stSidebar"] [data-testid="stTextInput"] input:focus{{
   border-color:{c["accent"]}!important;
   box-shadow:0 0 0 2px {c["accent_dim"]}!important;
}}
[data-baseweb="popover"],[data-baseweb="popover"] *{{
   background:{c["input_bg"]}!important;color:{c["text_pri"]}!important;
}}


/* --- BUTTONS BASE --- */
.stButton>button{{
   font-weight:700!important;border-radius:7px!important;padding:9px 16px!important;
   font-family:'Exo 2',sans-serif!important;font-size:13px!important;
   letter-spacing:0.5px!important;text-transform:uppercase!important;width:100%!important;
   transition:box-shadow 0.15s ease,transform 0.15s ease!important;
}}


/* --- START button --- */
.st-key-start_btn button{{
   background:{c["btn_start_bg"]}!important;color:{c["btn_start_color"]}!important;
   border:none!important;box-shadow:{c["btn_start_shadow"]}!important;
}}
.st-key-start_btn button:hover{{
   box-shadow:0 6px 20px rgba(14,165,233,0.55)!important;transform:translateY(-1px)!important;
}}
.st-key-start_btn button:disabled{{opacity:0.5!important;cursor:not-allowed!important;}}


/* --- STOP button --- */
.st-key-stop_btn button{{
   background:{c["btn_stop_bg"]}!important;color:{c["btn_stop_color"]}!important;
   border:1px solid {c["btn_stop_border"]}!important;box-shadow:none!important;
}}
.st-key-stop_btn button:disabled{{opacity:0.4!important;cursor:not-allowed!important;}}


/* --- RESET button --- */
.st-key-reset_btn button{{
   background:{c["btn_reset_bg"]}!important;color:{c["btn_reset_color"]}!important;
   border:1px solid {c["btn_reset_border"]}!important;box-shadow:none!important;
}}


/* --- SIDEBAR COLLAPSE / EXPAND BUTTONS ---
   Streamlit renders these with a Material-Symbols icon font. When that font
   fails to load, the raw ligature text ("keyboard_double_arrow_left") shows
   instead. We hide any text/svg inside the control and draw a clean chevron
   via ::after so the button always looks right regardless of font loading. */

/* Hide the inner icon text/svg for BOTH the collapse (in-sidebar) and the
   expand (collapsed-state) controls. */
[data-testid="stSidebarCollapseButton"] button > *,
[data-testid="stExpandSidebarButton"] button > *,
[data-testid="stSidebarCollapsedControl"] button > *{{
   font-size:0!important;color:transparent!important;
   width:0!important;height:0!important;overflow:hidden!important;
}}

/* Shared button shell */
[data-testid="stSidebarCollapseButton"] button,
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapsedControl"] button{{
   background:{c["nav_bg"]}!important;border:1px solid {c["nav_border"]}!important;
   border-radius:8px!important;
   box-shadow:0 2px 8px rgba(0,0,0,0.12)!important;
   width:34px!important;height:34px!important;padding:0!important;min-width:unset!important;
   display:flex!important;align-items:center!important;justify-content:center!important;
   overflow:hidden!important;
   transition:background 0.15s ease,border-color 0.15s ease,transform 0.15s ease!important;
}}
[data-testid="stSidebarCollapseButton"] button:hover,
[data-testid="stExpandSidebarButton"] button:hover,
[data-testid="stSidebarCollapsedControl"] button:hover{{
   background:{c["accent_dim"]}!important;border-color:{c["accent"]}!important;
   transform:translateY(-1px)!important;
}}

/* Collapse chevron (sidebar open) */
[data-testid="stSidebarCollapseButton"] button::after{{
   content:'\\276E';font-size:14px;font-weight:700;line-height:1;
   color:{c["nav_text"]}!important;display:block;
}}
/* Expand glyph (sidebar collapsed) — hamburger so it's obvious it opens */
[data-testid="stExpandSidebarButton"] button::after,
[data-testid="stSidebarCollapsedControl"] button::after{{
   content:'\\2630';font-size:15px;font-weight:700;line-height:1;
   color:{c["nav_text"]}!important;display:block;
}}
[data-testid="stSidebarCollapseButton"] button:hover::after,
[data-testid="stExpandSidebarButton"] button:hover::after,
[data-testid="stSidebarCollapsedControl"] button:hover::after{{
   color:{c["accent"]}!important;
}}


/* --- ABOUT BUTTON --- */
.st-key-about_sidebar_btn button{{
   background:{c["accent_dim"]}!important;color:{c["accent"]}!important;
   border:1px solid {c["border"]}!important;border-radius:6px!important;
   font-size:12px!important;font-weight:600!important;letter-spacing:0.3px!important;
   box-shadow:none!important;text-transform:none!important;
   padding:8px 14px!important;width:100%!important;
}}
.st-key-about_sidebar_btn button:hover{{
   border-color:{c["accent"]}!important;background:{c["accent"]}!important;color:#ffffff!important;
}}


/* --- TABS --- */
.stTabs [data-baseweb="tab-list"]{{
   background:transparent!important;border-bottom:1px solid {c["tab_border"]}!important;gap:4px!important;
}}
.stTabs [data-baseweb="tab"]{{
   background:transparent!important;color:{c["tab_color"]}!important;
   font-family:'Exo 2',sans-serif!important;font-size:13px!important;font-weight:600!important;
   border-bottom:2px solid transparent!important;padding:10px 20px!important;
   text-transform:uppercase!important;letter-spacing:0.8px!important;
}}
.stTabs [aria-selected="true"]{{
   color:{c["accent"]}!important;border-bottom:2px solid {c["accent"]}!important;
   background:{c["accent_dim"]}!important;
}}


/* --- DATAFRAME --- */
[data-testid="stDataFrame"]>div{{
   background:{c["df_bg"]}!important;
   border:1px solid {c["df_border"]}!important;border-radius:8px!important;
}}


/* --- SLIDER --- */
div[data-testid="stSlider"] [role="slider"]{{
   box-shadow:none!important;outline:none!important;background:{c["accent"]}!important;
}}


/* --- ALERTS --- */
[data-testid="stAlert"]{{
   background:{c["card_bg"]}!important;
   border:1px solid {c["border"]}!important;border-radius:8px!important;
   margin-top:0!important;
}}
[data-testid="stAlert"] *{{color:{c["text_sec"]}!important;}}


/* --- DIALOG --- */
[data-testid="stDialog"]>div{{
   background:{c["df_bg"]}!important;
   border:1px solid {c["border"]}!important;border-radius:12px!important;
   box-shadow:0 25px 60px rgba(0,0,0,0.4)!important;
}}


/* --- SCROLLBAR --- */
::-webkit-scrollbar{{width:6px;height:6px;}}
::-webkit-scrollbar-track{{background:{c["page_bg"]};}}
::-webkit-scrollbar-thumb{{background:{c["border"]};border-radius:3px;}}


/* --- RESPONSIVE CHART CONTAINER --- */
.chart-wrapper{{width:100%;overflow-x:auto;}}


/* --- LIVE STATUS PULSE DOT --- */
@keyframes nids-pulse{{
   0%{{box-shadow:0 0 0 0 var(--pulse-c);}}
   70%{{box-shadow:0 0 0 7px rgba(0,0,0,0);}}
   100%{{box-shadow:0 0 0 0 rgba(0,0,0,0);}}
}}
.nids-dot{{
   display:inline-block;width:7px;height:7px;border-radius:50%;
   margin-right:7px;vertical-align:middle;
}}
.nids-dot.live{{animation:nids-pulse 1.6s infinite;}}


/* --- HEADER SCANLINE GLOW --- */
@keyframes nids-scan{{
   0%{{transform:translateX(-100%);}}
   100%{{transform:translateX(100%);}}
}}
.nids-header{{
   position:relative;overflow:hidden;border-radius:12px;
   padding:18px 22px;margin-bottom:16px;
   background:{c['card_bg']};border:1px solid {c['card_border']};
   border-top:2px solid {c['card_border_t']};box-shadow:{c['card_shadow']};
}}
.nids-header::before{{
   content:'';position:absolute;top:0;left:0;height:2px;width:40%;
   background:linear-gradient(90deg,transparent,{c['accent']},transparent);
   animation:nids-scan 3.2s linear infinite;
}}


/* --- CUSTOM KPI CARDS (animated) --- */
.kpi-row{{
   display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:6px;
}}
@media (max-width:768px){{.kpi-row{{grid-template-columns:repeat(2,1fr);}}}}
.kpi-card{{
   position:relative;overflow:hidden;border-radius:10px;
   padding:16px 18px 16px 22px;min-height:104px;
   background:{c['card_bg']};border:1px solid {c['card_border']};
   border-top:2px solid {c['card_border_t']};box-shadow:{c['card_shadow']};
   transition:transform 0.18s ease,box-shadow 0.18s ease,border-color 0.18s ease;
}}
.kpi-card::before{{
   content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
   background:linear-gradient(180deg,var(--kpi-accent,{c['accent']}),transparent);
   opacity:0.9;
}}
.kpi-card:hover{{
   transform:translateY(-3px);border-color:var(--kpi-accent,{c['accent']});
   box-shadow:{c['card_shadow']},0 10px 28px {c['accent_dim']};
}}
.kpi-label{{
   font-size:10px;font-weight:700;color:{c['label_color']};
   text-transform:uppercase;letter-spacing:1.2px;
   font-family:'JetBrains Mono',monospace;margin-bottom:8px;
}}
.kpi-value{{
   font-size:30px;font-weight:800;color:var(--kpi-accent,{c['value_color']});
   font-family:'Rajdhani',sans-serif;letter-spacing:1px;line-height:1;
}}
</style>"""


st.markdown(_build_css(C), unsafe_allow_html=True)

# SESSION STATE INITIALISATION
_DEFAULT_STATE = {
   "running":            False,
   "csv_index":          0,
   "metrics":            {"evaluated": 0, "safe": 0, "attacks": 0},
   "threat_vectors":     {},
   "telemetry_logs":     [],
   "trend_history":      [],
   "last_attack_vector": None,
   "last_attack_label":  None,
   "xai_background_pool":[],
   "xai_normal_pool":    [],
   "debug_log":          [],
   "last_non_ip_count":  0,
}
for _k, _v in _DEFAULT_STATE.items():
   if _k not in st.session_state:
       st.session_state[_k] = _v


if "flow_tracker" not in st.session_state:
   st.session_state.flow_tracker = FlowTracker()
if "port_scan_detector" not in st.session_state:
   st.session_state.port_scan_detector = PortScanDetector()

# ABOUT DIALOG
@st.dialog("About the Research — RT-DeepNIDS")
def _show_abstract():
   c = C
   st.markdown(f"""
<div style="text-align:center;padding:4px 0 16px;border-bottom:1px solid {c['border']};margin-bottom:18px;">
 <div style="width:46px;height:46px;margin:0 auto 10px;border-radius:12px;
             display:flex;align-items:center;justify-content:center;color:{c['accent']};
             background:{c['accent_dim']};border:1px solid {c['accent']}55;">
   <span style="display:flex;width:23px;height:23px;">{ICON_SHIELD}</span></div>
 <div style="font-family:'Rajdhani',sans-serif;font-size:22px;font-weight:800;
             color:{c['text_pri']};text-transform:uppercase;letter-spacing:2px;">RT-DeepNIDS</div>
 <div style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
             letter-spacing:2px;color:{c['accent']};margin-top:4px;">
   HYBRID ML / DL INTRUSION DETECTION
 </div>
</div>

<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:18px;">
 <div style="background:{c['threat_box_bg']};border:1px solid {c['card_border']};
             border-left:3px solid #059669;border-radius:9px;padding:12px 14px;">
   <div style="font-family:'Rajdhani',sans-serif;font-size:21px;font-weight:800;color:#059669;line-height:1;">99.94%</div>
   <div style="font-family:'JetBrains Mono',monospace;font-size:8.5px;font-weight:700;
               letter-spacing:1px;text-transform:uppercase;color:{c['text_muted']};margin-top:4px;">Intra-Domain Accuracy</div>
 </div>
 <div style="background:{c['threat_box_bg']};border:1px solid {c['card_border']};
             border-left:3px solid {c['accent']};border-radius:9px;padding:12px 14px;">
   <div style="font-family:'Rajdhani',sans-serif;font-size:21px;font-weight:800;color:{c['accent']};line-height:1;">96.6%</div>
   <div style="font-family:'JetBrains Mono',monospace;font-size:8.5px;font-weight:700;
               letter-spacing:1px;text-transform:uppercase;color:{c['text_muted']};margin-top:4px;">Cross-Domain (RF)</div>
 </div>
 <div style="background:{c['threat_box_bg']};border:1px solid {c['card_border']};
             border-left:3px solid #f59e0b;border-radius:9px;padding:12px 14px;">
   <div style="font-family:'Rajdhani',sans-serif;font-size:21px;font-weight:800;color:#f59e0b;line-height:1;">~10&times;</div>
   <div style="font-family:'JetBrains Mono',monospace;font-size:8.5px;font-weight:700;
               letter-spacing:1px;text-transform:uppercase;color:{c['text_muted']};margin-top:4px;">Faster w/ Tomek+IHT</div>
 </div>
</div>

<div style="background:{c['card_bg']};border-left:3px solid {c['accent']};
           border-radius:6px;padding:18px 22px;box-shadow:0 4px 20px rgba(0,0,0,0.12);">
 <h4 style="font-family:'Rajdhani',sans-serif;font-size:13px;font-weight:700;
            color:{c['accent']};margin-top:0;margin-bottom:10px;text-transform:uppercase;letter-spacing:2px;
            font-family:'JetBrains Mono',monospace;">
   Abstract
 </h4>
 <p style="font-family:'Exo 2',sans-serif;color:{c['text_sec']};font-size:13.5px;line-height:1.75;margin-bottom:0;">
   The rapid expansion of digital communication, IoT architectures, and cloud-based services has led to an unprecedented surge in sophisticated cyberattacks. 
   Traditional signature-based Intrusion Detection Systems (IDS) rely on predefined rules, rendering them ineffective against zero-day vulnerabilities and polymorphic malware while generating excessive false positives. 
   To address these limitations, this paper proposes an intelligent Network Intrusion Detection System (NIDS) leveraging advanced Machine Learning (ML) and Deep Learning (DL) methodologies. 
   The framework evaluates hybrid CNN-GRU and CNN+Transformer architectures alongside Decision Tree, Random Forest, and XGBoost baselines across three benchmark datasets: CIC-IDS-2017, CIC-IDS-2018, and ToN-IoT-v3. 
   Two distinct evaluation pipelines are implemented: an intra-domain pipeline addressing class imbalance via SMOTE and a hybrid Tomek Links with Instance Hardness Threshold (Tomek+IHT) balancing strategy, 
   and a cross-domain transfer learning pipeline where models trained on IT network traffic are deployed directly onto unseen IoT telemetry to assess zero-day generalization. 
   Intra-domain results achieved near-perfect detection accuracies exceeding 99.8%, with Tomek+IHT reducing deep learning training latency by nearly tenfold at less than 2% accuracy trade-off. 
   In the cross-domain evaluation, Random Forest and XGBoost achieved robust accuracies of 96.60% and 96.50%, respectively. 
   SHAP-based Explainable AI (XAI) was integrated to interpret model decisions and identify critical network features driving anomaly detection. 
   The proposed models were further deployed into a lightweight real-time web-based monitoring prototype, demonstrating practical applicability. 
   This research delivers an accurate, explainable, and scalable NIDS solution capable of autonomous threat detection across both IT infrastructures and resource-constrained IoT environments.
 </p>
</div>""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
   c = C

   def _sb_section(num: str, text: str):
       st.markdown(
           f'<div class="sb-sec"><span class="sb-sec-num">{num}</span>'
           f'<span class="sb-sec-txt">{text}</span>'
           f'<span class="sb-sec-line"></span></div>',
           unsafe_allow_html=True,
       )

   st.markdown(f"""
<div class="sb-head">
 <div class="sb-shield">{ICON_SHIELD}</div>
 <div class="sb-title">RT-DeepNIDS</div>
 <div class="sb-sub">SYSTEM CONTROL PANEL</div>
</div>""", unsafe_allow_html=True)


   if st.button("About the Research", key="about_sidebar_btn", use_container_width=True):
       _show_abstract()


   st.markdown('<div style="margin:10px 0 2px;"></div>', unsafe_allow_html=True)
   _sb_section("01", "Detection Setup")


   monitoring_mode = st.radio(
       "Monitoring Mode",
       ["Intra-Domain (Standard)", "Cross-Domain (Robustness Test)"],
       disabled=st.session_state.running,
       key="mode_selector",
   )
   st.markdown('<div style="margin:8px 0 2px;"></div>', unsafe_allow_html=True)
   _sb_section("02", "Model & Dataset")


   # Dataset / balancing / model selectors
   if monitoring_mode == "Intra-Domain (Standard)":
       dataset_choice   = st.selectbox(
           "Dataset",
           ["— Select —", "CIC-IDS-2017", "CIC-IDS-2018", "ToN-IoT-v3"],
           disabled=st.session_state.running, key="ds_intra",
       )
       balancing_choice = st.selectbox(
           "Balancing Technique",
           ["— Select —", "SMOTE", "Tomek+IHT"],
           disabled=st.session_state.running, key="bal_intra",
       )
       model_choice = st.selectbox(
           "Model",
           ["— Select —", "Hybrid CNN-GRU", "Random Forest", "XGBoost", "Decision Tree"],
           disabled=st.session_state.running, key="model_intra",
           help="CNN+Transformer is only available in Cross-Domain mode.",
       )
   else:
       st.text_input("Dataset",             value="Source (CIC-IDS) to Target (ToN-IoT)", disabled=True, key="ds_cross")
       st.text_input("Balancing Technique", value="Tomek+IHT",                            disabled=True, key="bal_cross")
       dataset_choice   = "Cross-Domain"
       balancing_choice = "Tomek+IHT"
       model_choice = st.selectbox(
           "Model",
           ["— Select —", "Decision Tree", "Random Forest", "XGBoost", "Hybrid CNN-GRU", "CNN+Transformer"],
           disabled=st.session_state.running, key="model_cross",
       )


   st.markdown('<div style="margin:8px 0 2px;"></div>', unsafe_allow_html=True)
   _sb_section("03", "Traffic Source")


   # Traffic source
   traffic_source = st.selectbox(
       "Traffic Source",
       ["— Select —", "CSV Simulation", "Live Network Interface"],
       disabled=st.session_state.running,
       key="traffic_source_selector",
   )


   network_interface       = "Wi-Fi"
   network_interface_label = "Wi-Fi"
   demo_mode               = False


   if traffic_source == "Live Network Interface":
       if not SCAPY_AVAILABLE:
           st.error("Scapy not installed. Run:  pip install scapy")
       if not IS_ADMIN:
           st.error(
               "Administrator privileges required for live packet capture.\n\n"
               "Windows: Run terminal as Administrator.\n"
               "Linux / macOS: Run with  sudo streamlit run app.py"
           )
       else:
           st.success("Running with administrator privileges.")


       # Network interface dropdown
       _iface_pairs  = get_friendly_interfaces()
       _iface_labels = [p[0] for p in _iface_pairs]
       _iface_ids    = [p[1] for p in _iface_pairs]


       _sel_idx = st.selectbox(
           "Network Interface",
           options=range(len(_iface_labels)),
           index=0,
           format_func=lambda i: _iface_labels[i],
           disabled=st.session_state.running,
           key="net_iface_select",
           help="Select the adapter connected to your test/Kali machine.",
       )
       network_interface       = _iface_ids[_sel_idx]    if _iface_pairs else "Wi-Fi"
       network_interface_label = _iface_labels[_sel_idx]  if _iface_pairs else "Wi-Fi"
       st.caption("Select the active adapter connected to your Kali machine.")


       demo_mode = st.toggle(
           "High Sensitivity / Demo Mode",
           value=False,
           disabled=st.session_state.running,
           key="demo_mode_toggle",
           help="Lowers MIN_FLOW_SEC to 10 ms and PortScan threshold to 5 ports. "
                "Use for hping3 / nmap demos with short-lived flows.",
       )
       if demo_mode:
           st.caption("Demo Mode active — ultra-low thresholds, catches hping3/nmap immediately.")


   st.markdown('<div style="margin:8px 0 2px;"></div>', unsafe_allow_html=True)
   _sb_section("04", "Control")


   # Apply demo-mode thresholds to backend singletons
   if demo_mode:
       set_min_flow_sec(0.01)
       st.session_state.port_scan_detector.PORT_SCAN_THRESHOLD = 5
   else:
       set_min_flow_sec(0.1)
       st.session_state.port_scan_detector.PORT_SCAN_THRESHOLD = 15


   # Configuration readiness guard
   _unselected    = "— Select —"
   _config_ready  = not any(
       v == _unselected for v in [dataset_choice, balancing_choice, model_choice, traffic_source]
   )
   if not _config_ready and not st.session_state.running:
       st.warning("Select all options above before starting.")


   col1, col2 = st.columns(2)
   with col1:
       if st.button(
           "START", use_container_width=True,
           disabled=(st.session_state.running or not _config_ready),
           key="start_btn",
       ):
           st.session_state.running = True
           st.rerun()
   with col2:
       if st.button(
           "STOP", use_container_width=True,
           disabled=not st.session_state.running,
           key="stop_btn",
       ):
           st.session_state.running = False
           st.rerun()


   if st.button("RESET", use_container_width=True, key="reset_btn"):
       # Reset all state keys to defaults
       for _k, _v in _DEFAULT_STATE.items():
           import copy
           st.session_state[_k] = copy.deepcopy(_v)
       st.session_state.flow_tracker         = FlowTracker()
       st.session_state.port_scan_detector   = PortScanDetector()
       st.rerun()


   st.markdown('<div style="margin:8px 0 2px;"></div>', unsafe_allow_html=True)
   _sb_section("05", "Simulation Settings")
   throttle_speed   = st.slider("Simulation Speed (sec / packet)", 0.01, 1.0, 0.05, step=0.01)
   render_interval  = st.slider("Refresh Interval (sec)",          0.1,  2.0, 0.5,  step=0.1)
   telemetry_filter = st.radio("Log Filter", ["All Logs", "Normal Only", "Attacks Only"])

   st.markdown("""
<div class="sb-foot">
 <div class="sb-foot-t">RT-DeepNIDS &middot; v1.0</div>
 <div class="sb-foot-s">BUBT &mdash; Dept. of CSE</div>
</div>""", unsafe_allow_html=True)

# BACKEND PATH RESOLUTION & ASSET LOADING
current_dataset_dir = resolve_dataset_dir(monitoring_mode, balancing_choice, dataset_choice)
target_model_file   = resolve_model_file(monitoring_mode, model_choice)


target_model_path  = os.path.join(current_dataset_dir, target_model_file) if target_model_file else ""
target_scaler_path = os.path.join(current_dataset_dir, "scaler.pkl")
_model_missing     = bool(target_model_file and not os.path.exists(target_model_path))
_scaler_missing    = not os.path.exists(target_scaler_path)


def _show_asset_guide(missing_model: bool, missing_scaler: bool,
                     model_file: str, dataset_dir: str) -> None:
   """Render an asset-resolution failure card with the expected folder tree."""
   st.markdown(f"""
<div style="background:linear-gradient(145deg,#1a0808,#120606);
           border:1px solid #7f1d1d;border-left:3px solid #ef4444;
           border-radius:10px;padding:20px 24px;margin-bottom:18px;">
 <div style="font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;
             color:#ef4444;letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">
   Asset Resolution Failure
 </div>
 <p style="font-family:'Exo 2',sans-serif;font-size:13px;color:#fca5a5;margin:0 0 12px;">
   One or more model asset files could not be located.
   Verify your <code>Real_Time_Export</code> folder matches the structure below.
 </p>
 <div style="font-family:'JetBrains Mono',monospace;font-size:11.5px;color:#f87171;
             background:#0d0303;border:1px solid #7f1d1d;border-radius:6px;
             padding:14px 16px;line-height:1.9;white-space:pre;overflow-x:auto;">
Real_Time_Export/
├── SMOTE/
│   ├── CIC_IDS_2017/
│   │   ├── CNN_GRU_model.h5
│   │   ├── RF_model.pkl  |  DT_model.pkl  |  XGB_model.pkl
│   │   ├── scaler.pkl
│   │   └── live_traffic_sample.csv
│   ├── CIC_IDS_2018/   (same structure)
│   └── TON_IOT_V3/     (same structure)
├── Tomek_IHT/          (same structure as SMOTE/)
└── Cross_Validation/
   ├── CNN_Transformer_model.h5
   ├── CNN_GRU_model.h5
   ├── RF_model.pkl  |  DT_model.pkl  |  XGB_model.pkl
   ├── scaler.pkl
   └── live_traffic_sample.csv
 </div>
 <p style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#fca5a5;margin-top:12px;line-height:1.7;">
   <b>Expected path :</b> {dataset_dir}<br>
   <b>Model file    :</b> {model_file or "(none selected)"}
   {'  —  NOT FOUND' if missing_model  else '  —  Found'}<br>
   <b>Scaler file   :</b> scaler.pkl
   {'  —  NOT FOUND' if missing_scaler else '  —  Found'}
 </p>
</div>""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading model assets...")
def _cached_load(dataset_dir: str, model_filename: str):
   return load_framework_assets(dataset_dir, model_filename)

@st.cache_data(show_spinner=False)
def _cached_csv(csv_path: str):
   return load_csv_source(csv_path)

active_model, active_scaler, classes = _cached_load(current_dataset_dir, target_model_file)
csv_path  = os.path.join(current_dataset_dir, "live_traffic_sample.csv")
df_source = _cached_csv(csv_path)


# Stop immediately if model is missing while running
if active_model is None and st.session_state.running:
   _show_asset_guide(_model_missing, _scaler_missing, target_model_file, current_dataset_dir)
   st.session_state.running = False
   st.stop()


# Passive warning when idle and files are absent
if not st.session_state.running and target_model_file and _model_missing:
   _show_asset_guide(_model_missing, _scaler_missing, target_model_file, current_dataset_dir)


_feature_count = get_feature_count(active_scaler, active_model, dataset_choice)


PLOT_BG = "rgba(0,0,0,0)"

def _base_layout(c: dict, **kw) -> dict:
   return dict(
       paper_bgcolor=PLOT_BG,
       plot_bgcolor=PLOT_BG,
       font=dict(family="'JetBrains Mono', monospace", color=c["tick"], size=11),
       margin=dict(l=10, r=40, t=35, b=10),
       uirevision="constant",
       **kw,
   )

# LANDING PAGE (shown before first run)
def _landing_page_html(c: dict) -> str:
   # Minimal monochrome line icons (inherit currentColor) — professional,
   # consistent rendering across all platforms (unlike emoji).
   _ic_shield = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                 '<path d="M12 2 4 5v6c0 5 3.5 8.5 8 11 4.5-2.5 8-6 8-11V5l-8-3z"/>'
                 '<path d="m9 12 2 2 4-4"/></svg>')
   _ic_super  = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                 '<path d="M12 2 9.5 8 3 8.5l5 4-1.5 6.5L12 16l5.5 3L16 12.5l5-4L14.5 8 12 2z"/></svg>')
   _ic_team   = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                 '<circle cx="9" cy="8" r="3.2"/><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6"/>'
                 '<path d="M16 4.5a3.2 3.2 0 0 1 0 6.3"/><path d="M18 14c2.2.7 3.8 2.7 3.8 5"/></svg>')
   _ic_flow   = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                 '<rect x="3" y="4" width="6" height="6" rx="1"/><rect x="15" y="14" width="6" height="6" rx="1"/>'
                 '<path d="M9 7h6a3 3 0 0 1 3 3v4"/></svg>')
   _ic_data   = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                 '<ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5"/>'
                 '<path d="M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></svg>')
   _ic_brain  = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                 '<rect x="3" y="6" width="6" height="6" rx="1.5"/><rect x="15" y="4" width="6" height="5" rx="1.5"/>'
                 '<rect x="14" y="14" width="7" height="6" rx="1.5"/><path d="M9 9h3v8h2"/><path d="M18 9v5"/></svg>')
   _ic_xfer   = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                 '<path d="M4 8h13l-3-3M20 16H7l3 3"/></svg>')
   _ic_balance= ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                 '<path d="M12 3v18M5 7h14"/><path d="M5 7 2 13a3 3 0 0 0 6 0L5 7z"/>'
                 '<path d="M19 7l-3 6a3 3 0 0 0 6 0l-3-6z"/></svg>')
   _ic_xai    = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                 '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>'
                 '<path d="M11 8v.01M11 11v3"/></svg>')
   _ic_deploy = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                 '<path d="M5 13c0-5 3-9 7-9s7 4 7 9l-3 2H8l-3-2z"/><circle cx="12" cy="9" r="1.5"/>'
                 '<path d="M9 17l-1 4M15 17l1 4"/></svg>')
   _ic_globe  = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
                 'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                 '<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/>'
                 '<path d="M12 3c2.5 2.5 3.8 5.7 3.8 9s-1.3 6.5-3.8 9c-2.5-2.5-3.8-5.7-3.8-9s1.3-6.5 3.8-9z"/></svg>')
   return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@700&family=JetBrains+Mono:wght@600;700&family=Exo+2:wght@400;600;700;800&display=swap');
.lp-wrap{{
   background:{c['card_bg']};border:1px solid {c['card_border']};
   border-top:2px solid {c['card_border_t']};border-radius:14px;
   padding:30px 44px 36px;margin-top:8px;
   box-shadow:{c['card_shadow']};position:relative;overflow:hidden;
}}
.lp-wrap::before{{
   content:'';position:absolute;top:0;left:0;height:2px;width:38%;
   background:linear-gradient(90deg,transparent,{c['accent']},transparent);
   animation:lp-scan 3.4s linear infinite;
}}
@keyframes lp-scan{{0%{{transform:translateX(-120%);}}100%{{transform:translateX(360%);}}}}
.lp-shield{{
   width:54px;height:54px;margin:0 auto 12px;border-radius:14px;
   display:flex;align-items:center;justify-content:center;color:{c['accent']};
   background:{c['accent_dim']};border:1px solid {c['accent']}55;
   box-shadow:0 0 22px {c['accent_dim']};
}}
.lp-shield svg{{width:28px;height:28px;display:block;}}
.lp-title{{
   font-family:'Rajdhani',sans-serif;font-size:32px;font-weight:700;
   color:{c['text_pri']};text-align:center;margin-bottom:4px;
   text-transform:uppercase;letter-spacing:3px;
}}
.lp-subtitle{{
   font-family:'Exo 2',sans-serif;font-size:13px;color:{c['accent']};
   text-align:center;letter-spacing:1px;margin-bottom:6px;font-weight:600;
}}
.lp-inst{{
   font-family:'Exo 2',sans-serif;font-size:13px;color:{c['text_muted']};
   text-align:center;margin-bottom:22px;line-height:1.6;
}}
.lp-inst b{{color:{c['accent']};}}

/* Key Contributions grid */
.contrib-grid{{
   display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px;
}}
.contrib-card{{
   position:relative;overflow:hidden;display:flex;gap:12px;align-items:flex-start;
   background:{c['card_bg']};border:1px solid {c['card_border']};
   border-top:2px solid {c['card_border_t']};border-radius:11px;
   padding:16px 18px;box-shadow:{c['card_shadow']};
   transition:transform 0.18s ease,border-color 0.18s ease,box-shadow 0.18s ease;
}}
.contrib-card::before{{
   content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
   background:linear-gradient(180deg,{c['accent']},transparent);opacity:0.9;
}}
.contrib-card:hover{{
   transform:translateY(-4px);border-color:{c['accent']};
   box-shadow:{c['card_shadow']},0 12px 30px {c['accent_dim']};
}}
.contrib-ico{{
   flex-shrink:0;width:38px;height:38px;border-radius:10px;
   display:flex;align-items:center;justify-content:center;color:{c['accent']};
   background:{c['accent_dim']};border:1px solid {c['accent']}44;
}}
.contrib-ico svg{{width:19px;height:19px;display:block;}}
.contrib-body{{min-width:0;}}
.contrib-tag{{
   font-family:'JetBrains Mono',monospace;font-size:8.5px;font-weight:700;
   letter-spacing:1.5px;text-transform:uppercase;color:{c['accent']};margin-bottom:3px;
}}
.contrib-title{{
   font-family:'Exo 2',sans-serif;font-size:14px;font-weight:700;
   color:{c['text_pri']};line-height:1.25;margin-bottom:5px;
}}
.contrib-desc{{
   font-family:'Exo 2',sans-serif;font-size:11.5px;color:{c['text_muted']};line-height:1.55;
}}
.contrib-desc b{{color:{c['text_sec']};font-weight:700;}}

/* Stat chips strip */
.lp-stats{{
   display:flex;justify-content:center;gap:10px;flex-wrap:wrap;
   margin:0 0 8px;
}}
.lp-stat{{
   flex:1;min-width:120px;max-width:180px;text-align:center;
   background:{c['threat_box_bg']};border:1px solid {c['card_border']};
   border-radius:10px;padding:14px 10px;box-shadow:{c['card_shadow']};
   transition:transform 0.18s ease,border-color 0.18s ease;
}}
.lp-stat:hover{{transform:translateY(-3px);border-color:{c['accent']};}}
.lp-stat-v{{
   font-family:'Rajdhani',sans-serif;font-size:24px;font-weight:800;
   color:{c['accent']};line-height:1;
}}
.lp-stat-l{{
   font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
   letter-spacing:1.2px;text-transform:uppercase;color:{c['text_muted']};margin-top:5px;
}}

/* Section heading (matches sidebar style) */
.lp-sec{{
   display:flex;align-items:center;gap:9px;margin:26px 0 14px;
}}
.lp-sec-ico{{
   flex-shrink:0;width:22px;height:22px;border-radius:6px;
   display:flex;align-items:center;justify-content:center;
   background:{c['accent_dim']};border:1px solid {c['accent']}44;color:{c['accent']};
}}
.lp-sec-ico svg{{width:13px;height:13px;display:block;}}
.lp-sec-txt{{
   font-family:'JetBrains Mono',monospace;font-size:11px;font-weight:700;
   color:{c['accent']};text-transform:uppercase;letter-spacing:2px;
}}
.lp-sec-line{{flex:1;height:1px;background:linear-gradient(90deg,{c['border']},transparent);}}

.team-grid{{
   display:flex;flex-wrap:wrap;justify-content:center;gap:10px;
}}
.team-card{{
   position:relative;overflow:hidden;flex:1 1 180px;max-width:230px;
   background:{c['card_bg']};border:1px solid {c['card_border']};
   border-radius:9px;padding:14px;text-align:center;box-shadow:{c['card_shadow']};
   transition:transform 0.18s ease,border-color 0.18s ease,box-shadow 0.18s ease;
}}
.team-card::before{{
   content:'';position:absolute;left:0;top:0;bottom:0;width:3px;
   background:linear-gradient(180deg,{c['accent']},transparent);opacity:0.85;
}}
.team-card:hover{{
   transform:translateY(-3px);border-color:{c['accent']};
   box-shadow:{c['card_shadow']},0 10px 24px {c['accent_dim']};
}}
.sup-card{{
   position:relative;overflow:hidden;
   background:{c['threat_box_bg']};border:1px solid {c['card_border']};
   border-left:3px solid {c['accent']};border-radius:10px;
   padding:18px 22px;text-align:center;box-shadow:{c['card_shadow']};
   max-width:560px;margin:0 auto;
}}
.team-name{{font-family:'Exo 2',sans-serif;font-weight:700;color:{c['text_pri']};font-size:13.5px;margin-bottom:4px;}}
.team-id{{font-family:'JetBrains Mono',monospace;font-size:11px;color:{c['accent']};font-weight:600;}}
.pipeline{{
   display:flex;justify-content:space-between;align-items:center;
   background:{c['threat_box_bg']};border:1px solid {c['border']};border-radius:10px;
   padding:20px 22px;flex-wrap:wrap;gap:8px;
}}
.p-step{{
   text-align:center;flex:1;min-width:100px;padding:8px 4px;border-radius:8px;
   transition:background 0.18s ease,transform 0.18s ease;
}}
.p-step:hover{{background:{c['accent_dim']};transform:translateY(-2px);}}
.p-num{{
   width:30px;height:30px;background:linear-gradient(145deg,#0ea5e9,#0284c7);
   border-radius:50%;display:flex;align-items:center;justify-content:center;
   font-family:'JetBrains Mono',monospace;font-size:12px;font-weight:700;
   color:white;margin:0 auto 8px;box-shadow:0 0 12px rgba(14,165,233,0.4);
}}
.p-title{{font-family:'Exo 2',sans-serif;font-size:12px;font-weight:700;color:{c['text_pri']};margin-bottom:2px;}}
.p-desc{{font-family:'Exo 2',sans-serif;font-size:11px;color:{c['text_muted']};}}
.p-arrow{{color:{c['accent']};font-size:18px;font-weight:bold;opacity:0.5;}}
.lp-table{{
   width:100%;border-collapse:collapse;margin-top:8px;
   border:1px solid {c['border']};border-radius:9px;overflow:hidden;
   box-shadow:{c['card_shadow']};
}}
.lp-table th{{
   background:{c['card_bg']};color:{c['accent']};padding:12px 16px;text-align:left;
   font-family:'JetBrains Mono',monospace;font-size:10px;font-weight:700;
   letter-spacing:1px;text-transform:uppercase;border-bottom:1px solid {c['border']};
}}
.lp-table td{{
   padding:11px 16px;border-bottom:1px solid {c['border']};
   font-family:'Exo 2',sans-serif;font-size:13px;color:{c['text_sec']};
}}
.lp-table tbody tr{{transition:background 0.15s ease;}}
.lp-table tbody tr:hover{{background:{c['accent_dim']};}}
.lp-table tr:last-child td{{border-bottom:none;}}
.badge{{
   display:inline-block;background:{c['accent_dim']};border:1px solid {c['border']};
   color:{c['accent']};padding:2px 8px;border-radius:4px;
   font-size:11px;font-family:'JetBrains Mono',monospace;font-weight:600;
}}
</style>
<div class="lp-wrap">
 <div class="lp-shield">{_ic_shield}</div>
 <div class="lp-title">RT-DeepNIDS</div>
 <div class="lp-subtitle">A Real-Time Hybrid Network Intrusion Detection System for IT and IoT Environments</div>
 <div class="lp-inst">
   Bangladesh University of Business and Technology (BUBT)<br>
   <b>Department of Computer Science and Engineering</b>
 </div>


 <div class="lp-sec"><span class="lp-sec-ico">{_ic_super}</span><span class="lp-sec-txt">Proposed Solution &amp; Novelty</span><span class="lp-sec-line"></span></div>
 <div style="font-family:'Exo 2',sans-serif;font-size:12.5px;color:{c['text_muted']};
             margin:-6px 0 14px 2px;font-style:italic;">
   Five core innovations underpinning the proposed intrusion detection framework
 </div>
 <div class="contrib-grid">
   <div class="contrib-card">
     <div class="contrib-ico">{_ic_brain}</div>
     <div class="contrib-body">
       <div class="contrib-tag">Novelty 01</div>
       <div class="contrib-title">Hybrid Architecture</div>
       <div class="contrib-desc">Advanced hybrid deep learning models <b>(CNN+GRU, CNN+Transformer)</b> developed alongside robust ML baselines such as <b>XGBoost and Random Forest</b>.</div>
     </div>
   </div>
   <div class="contrib-card">
     <div class="contrib-ico">{_ic_flow}</div>
     <div class="contrib-body">
       <div class="contrib-tag">Novelty 02</div>
       <div class="contrib-title">Novel Dual-Pipeline Approach</div>
       <div class="contrib-desc">Parallel class-balancing strategies &mdash; <b>SMOTE versus Tomek Links with IHT</b> &mdash; evaluated systematically across large-scale network traffic data.</div>
     </div>
   </div>
   <div class="contrib-card">
     <div class="contrib-ico">{_ic_xai}</div>
     <div class="contrib-body">
       <div class="contrib-tag">Novelty 03</div>
       <div class="contrib-title">XAI Integration</div>
       <div class="contrib-desc"><b>SHAP (Shapley Additive Explanations)</b> values applied throughout to ensure complete transparency in every model decision.</div>
     </div>
   </div>
   <div class="contrib-card">
     <div class="contrib-ico">{_ic_globe}</div>
     <div class="contrib-body">
       <div class="contrib-tag">Novelty 04</div>
       <div class="contrib-title">Zero-Day Cross-Domain Evaluation</div>
       <div class="contrib-desc">Models trained exclusively on traditional IT network traffic <b>(CIC-IDS 2017/2018)</b> and evaluated directly on unseen IoT telemetry <b>(ToN-IoT-v3)</b>.</div>
     </div>
   </div>
   <div class="contrib-card">
     <div class="contrib-ico">{_ic_deploy}</div>
     <div class="contrib-body">
       <div class="contrib-tag">Novelty 05</div>
       <div class="contrib-title">Real-Time Deployment</div>
       <div class="contrib-desc">Trained models deployed live in an interactive <b>Streamlit dashboard</b> supporting <b>both CSV replay and Scapy-based live packet capture</b> &mdash; raw traffic to prediction in a single pipeline &mdash; with on-demand <b>SHAP explanations</b> for every verdict. <span style="color:{c['text_muted']};">(This hosted demo runs in CSV-simulation mode; live capture requires a local deployment.)</span></div>
     </div>
   </div>
 </div>


 <div class="lp-sec"><span class="lp-sec-ico">{_ic_data}</span><span class="lp-sec-txt">Project Highlights</span><span class="lp-sec-line"></span></div>
 <div class="lp-stats">
   <div class="lp-stat"><div class="lp-stat-v">3</div><div class="lp-stat-l">Benchmark Datasets</div></div>
   <div class="lp-stat"><div class="lp-stat-v">5</div><div class="lp-stat-l">ML / DL Models</div></div>
   <div class="lp-stat"><div class="lp-stat-v">99.94%</div><div class="lp-stat-l">Peak Accuracy</div></div>
   <div class="lp-stat"><div class="lp-stat-v">2</div><div class="lp-stat-l">Eval Pipelines</div></div>
 </div>


 <div class="lp-sec"><span class="lp-sec-ico">{_ic_flow}</span><span class="lp-sec-txt">System Methodology Pipeline</span><span class="lp-sec-line"></span></div>
 <div class="pipeline">
   <div class="p-step">
     <div class="p-num">01</div>
     <div class="p-title">Data Ingestion</div>
     <div class="p-desc">Raw PCAP / CSV</div>
   </div>
   <div class="p-arrow">&#8250;</div>
   <div class="p-step">
     <div class="p-num">02</div>
     <div class="p-title">Preprocessing</div>
     <div class="p-desc">Cleaning and Scaling</div>
   </div>
   <div class="p-arrow">&#8250;</div>
   <div class="p-step">
     <div class="p-num">03</div>
     <div class="p-title">Balancing</div>
     <div class="p-desc">SMOTE / Tomek+IHT</div>
   </div>
   <div class="p-arrow">&#8250;</div>
   <div class="p-step">
     <div class="p-num">04</div>
     <div class="p-title">Model Training</div>
     <div class="p-desc">ML and DL Architectures</div>
   </div>
   <div class="p-arrow">&#8250;</div>
   <div class="p-step">
     <div class="p-num">05</div>
     <div class="p-title">Prediction</div>
     <div class="p-desc">Real-Time Telemetry</div>
   </div>
 </div>


 <div class="lp-sec"><span class="lp-sec-ico">{_ic_data}</span><span class="lp-sec-txt">Dataset Benchmarks</span><span class="lp-sec-line"></span></div>
 <table class="lp-table">
   <thead>
     <tr><th>Dataset</th><th>Feature Space</th><th>Total Samples (Approx.)</th></tr>
   </thead>
   <tbody>
     <tr>
       <td><strong>CIC-IDS-2017</strong></td>
       <td><span class="badge">78 Features</span></td>
       <td>2.8 M Samples</td>
     </tr>
     <tr>
       <td><strong>CIC-IDS-2018</strong></td>
       <td><span class="badge">80 Features</span></td>
       <td>16 M Samples</td>
     </tr>
     <tr>
       <td><strong>ToN-IoT-v3</strong></td>
       <td><span class="badge">~470 Features</span></td>
       <td>2.4 M Samples</td>
     </tr>
   </tbody>
 </table>


 <div class="lp-sec"><span class="lp-sec-ico">{_ic_super}</span><span class="lp-sec-txt">Supervised By</span><span class="lp-sec-line"></span></div>
 <div class="sup-card">
   <div style="font-family:'Rajdhani',sans-serif;font-size:18px;font-weight:700;
               color:{c['text_pri']};letter-spacing:0.5px;">Md. Saifur Rahman</div>
   <div style="font-family:'Exo 2',sans-serif;font-size:12.5px;
               color:{c['accent']};font-weight:600;margin:3px 0 2px;">Assistant Professor</div>
   <div style="font-family:'Exo 2',sans-serif;font-size:12px;color:{c['text_muted']};line-height:1.6;">
     Department of Computer Science &amp; Engineering<br>
     Bangladesh University of Business and Technology (BUBT), Dhaka, Bangladesh
   </div>
 </div>


 <div class="lp-sec"><span class="lp-sec-ico">{_ic_team}</span><span class="lp-sec-txt">Research Team</span><span class="lp-sec-line"></span></div>
 <div class="team-grid">
   <div class="team-card"><div class="team-name">Ayesha Siddika</div><div class="team-id">ID: 22234103099</div></div>
   <div class="team-card"><div class="team-name">Sanjida Khanom</div><div class="team-id">ID: 22234103103</div></div>
   <div class="team-card"><div class="team-name">Ihsanul Hossain Rafsan</div><div class="team-id">ID: 22234103112</div></div>
   <div class="team-card"><div class="team-name">Sadia Mehrin Rahi</div><div class="team-id">ID: 22234103122</div></div>
   <div class="team-card"><div class="team-name">Istiyak Hasan Maruf</div><div class="team-id">ID: 22234103130</div></div>
 </div>
</div><br>"""

# Show landing page only on first load
if (not st.session_state.running
       and st.session_state.csv_index == 0
       and not st.session_state.telemetry_logs):
   st.info("Select your configuration in the control panel and press Start to begin monitoring.")
   st.markdown(_landing_page_html(C), unsafe_allow_html=True)
   st.stop()

# XAI / SHAP PANEL
def _render_xai_panel(c: dict, model, scaler, m_choice: str,
                     ds_choice: str, n_feats: int) -> None:
   """
   Render the Explainable AI panel for the most recent Attack prediction.
   Uses SHAP when available; falls back to sklearn feature_importances_.
   """
   st.markdown(f"""
<div style="background:{c['threat_box_bg']};border:1px solid {c['border']};
           border-left:3px solid {c['accent']};border-radius:10px;
           padding:16px 20px;margin-bottom:14px;">
 <div style="font-size:10px;font-weight:700;color:{c['accent']};
             font-family:'JetBrains Mono',monospace;letter-spacing:2px;margin-bottom:6px;">
   EXPLAINABLE AI — DECISION TRANSPARENCY (SHAP)
 </div>
 <div style="font-size:12px;color:{c['text_sec']};font-family:'Exo 2',sans-serif;line-height:1.6;">
   SHapley Additive exPlanations — top features that drove the most recent
   <b style="color:#ef4444;">Attack</b> classification decision.
 </div>
</div>""", unsafe_allow_html=True)


   if model is None:
       st.warning("Model not loaded. Select a valid configuration and press Start, then enable XAI.")
       return


   # Resolve feature column labels from DATASET_COLS
   _raw_cols   = DATASET_COLS.get(ds_choice, [])
   _col_labels = [str(col).strip()[:35] for col in _raw_cols]
   while len(_col_labels) < n_feats:
       _col_labels.append(f"feat_{len(_col_labels)}")


   explain_vec = st.session_state.get("last_attack_vector", None)
   last_label  = st.session_state.get("last_attack_label",  "Attack")


   if explain_vec is None:
       st.info(
           "No attack prediction recorded yet. Start monitoring — "
           "the XAI panel will populate on the first attack detection."
       )
       return


   explain_vec = np.array(explain_vec, dtype=float).flatten()
   explain_vec = (
       np.pad(explain_vec, (0, n_feats - len(explain_vec)))
       if len(explain_vec) < n_feats
       else explain_vec[:n_feats]
   )
   explain_2d = explain_vec.reshape(1, -1)


   # Build SHAP background from Normal-only pool (or fallback to mixed pool)
   pool      = st.session_state.get("xai_background_pool", [])
   norm_pool = st.session_state.get("xai_normal_pool",     [])


   if len(norm_pool) >= 5:
       _bg_rows  = norm_pool[-100:]
       _bg_arr   = np.array(
           [np.pad(r, (0, max(0, n_feats - len(r))))[:n_feats] for r in _bg_rows],
           dtype=float,
       )
       try:
           import shap as _shap_bg
           _k         = min(10, len(_bg_arr))
           background = _shap_bg.kmeans(_bg_arr, _k)
           _bg_label  = f"shap.kmeans(k={_k}, {len(_bg_arr)} Normal vectors)"
       except Exception:
           background = _bg_arr
           _bg_label  = f"Raw Normal background ({len(_bg_arr)} rows)"
   elif len(pool) >= 5:
       _bg_rows  = pool[-30:]
       background = np.array(
           [np.pad(r, (0, max(0, n_feats - len(r))))[:n_feats] for r in _bg_rows],
           dtype=float,
       )
       _bg_label = f"Mixed pool fallback ({len(_bg_rows)} rows)"
   else:
       background = explain_2d
       _bg_label  = "Explain-vec fallback (pool too small)"


   _DL           = {"Hybrid CNN-GRU", "CNN+Transformer"}
   _shap_vals    = None
   _imp_vals     = None
   _method_label = "N/A"
   _has_shap     = False


   try:
       import shap as _shap
       _has_shap = True
   except ImportError:
       pass


   if _has_shap:
       try:
           if m_choice in _DL:
               def _pred_fn(x):
                   inp = x.reshape(x.shape[0], 1, x.shape[1])
                   out = model.predict(inp, verbose=0)
                   return out if out.shape[-1] > 1 else np.hstack([1 - out, out])


               _exp          = _shap.KernelExplainer(_pred_fn, background)
               _raw          = _exp.shap_values(explain_2d, nsamples=128, silent=True)
               _method_label = f"KernelSHAP (DL)  |  background: {_bg_label}"
           else:
               _exp          = _shap.TreeExplainer(model)
               _raw          = _exp.shap_values(explain_2d)
               _method_label = "TreeSHAP (exact)"


           # Normalise SHAP output shape
           if isinstance(_raw, list):
               _raw = _raw[-1]
           _raw       = np.array(_raw, dtype=float).squeeze()
           if _raw.ndim == 2: _raw = _raw[0]
           if _raw.ndim != 1: _raw = _raw.flatten()
           _shap_vals = np.abs(_raw).astype(float)


       except Exception as _shap_e:
           st.warning(f"SHAP computation failed: {_shap_e}  —  using feature_importances_ fallback.")
           if hasattr(model, "feature_importances_"):
               _imp_vals     = np.array(model.feature_importances_, dtype=float).flatten()
               _method_label = f"Feature Importance (SHAP error: {str(_shap_e)[:60]})"
           else:
               st.error(f"SHAP failed and model has no feature_importances_.\n{_shap_e}")
               return
   else:
       if hasattr(model, "feature_importances_"):
           _imp_vals     = np.array(model.feature_importances_, dtype=float).flatten()
           _method_label = "Feature Importance (sklearn)  —  install shap for SHAP values"
           st.info("shap not installed. Showing sklearn feature_importances_. Run: pip install shap")
       else:
           st.error("shap not installed and this model has no feature_importances_. Run: pip install shap")
           return


   _importance = _shap_vals if _shap_vals is not None else _imp_vals
   if _importance is None:
       st.error("No importance values available.")
       return


   _importance = np.array(_importance, dtype=float).flatten()
   n_avail     = min(len(_importance), len(_col_labels))
   _importance = _importance[:n_avail]
   _names      = np.array(_col_labels[:n_avail], dtype=object)


   top_n   = min(15, n_avail)
   top_idx = np.argsort(_importance)[::-1][:top_n]
   t_vals  = _importance[top_idx].astype(float)
   t_names = [str(s) for s in _names[top_idx].tolist()]
   _max    = float(t_vals.max()) if t_vals.max() > 0 else 1.0
   t_norm  = (t_vals / _max).tolist()


   clrs = [
       f"rgba(239,68,68,{0.4 + 0.6*v:.2f})" if v > 0.5
       else f"rgba(56,189,248,{0.3 + 0.5*v:.2f})"
       for v in t_norm
   ]
   explain_vals = [f"{float(explain_vec[i]):.4f}" for i in top_idx.tolist()]


   fig_xai = go.Figure(go.Bar(
       x=t_vals[::-1].tolist(),
       y=[n[:32] for n in t_names[::-1]],
       orientation="h",
       marker=dict(color=clrs[::-1],
                   line=dict(color=c["accent"], width=0.5)),
       text=[f"{float(v):.4f}" for v in t_vals[::-1]],
       textposition="outside",
       textfont=dict(family="JetBrains Mono", size=10, color=c["text_sec"]),
       customdata=explain_vals[::-1],
       hovertemplate=(
           "<b>%{y}</b><br>"
           "SHAP |value|: %{x:.4f}<br>"
           "Feature value: %{customdata}<extra></extra>"
       ),
   ))
   fig_xai.update_layout(
       paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
       font=dict(family="'JetBrains Mono',monospace", color=c["tick"], size=11),
       margin=dict(l=10, r=60, t=20, b=10),
       uirevision="xai_chart",
       height=max(360, top_n * 28),
       xaxis=dict(
           showgrid=True, gridcolor=c["grid"], zeroline=False,
           title=dict(text="SHAP |value| / Importance",
                      font=dict(size=11, color=c["text_muted"], family="JetBrains Mono")),
           tickfont=dict(color=c["tick"], family="JetBrains Mono"),
       ),
       yaxis=dict(
           showgrid=False, type="category", automargin=True,
           tickfont=dict(color=c["tick"], family="JetBrains Mono", size=11),
       ),
       bargap=0.3,
   )
   # --- Summary stat strip + top-3 driver badges (above the chart) ---
   _is_shap   = _shap_vals is not None
   _method_short = "SHAP (Shapley values)" if _is_shap else "Feature Importance"
   _method_clr   = "#059669" if _is_shap else "#f59e0b"
   _lbl_clr      = "#059669" if str(last_label).lower() in ("normal", "benign") else "#ef4444"

   def _stat(label, value, vclr):
       return (
           f'<div style="flex:1;min-width:120px;background:{c["card_bg"]};'
           f'border:1px solid {c["card_border"]};border-radius:9px;padding:11px 14px;'
           f'box-shadow:{c["card_shadow"]};">'
           f'<div style="font-family:JetBrains Mono,monospace;font-size:8.5px;font-weight:700;'
           f'letter-spacing:1.2px;text-transform:uppercase;color:{c["text_muted"]};margin-bottom:4px;">{label}</div>'
           f'<div style="font-family:Rajdhani,sans-serif;font-size:17px;font-weight:800;'
           f'color:{vclr};line-height:1.05;">{value}</div></div>'
       )

   st.markdown(
       '<div style="display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px;">'
       + _stat("Predicted Label", str(last_label), _lbl_clr)
       + _stat("Method", _method_short, _method_clr)
       + _stat("Features Analyzed", f"{n_avail:,}", c["accent"])
       + _stat("Top Driver", t_names[0][:18] if t_names else "—", c["text_pri"])
       + "</div>",
       unsafe_allow_html=True,
   )

   # Top-3 driver badges
   _medals = ["#1", "#2", "#3"]
   _badge_parts = []
   for _r in range(min(3, len(t_names))):
       _fval = float(explain_vec[int(top_idx[_r])]) if int(top_idx[_r]) < len(explain_vec) else 0.0
       _badge_parts.append(
           f'<div style="display:inline-flex;align-items:center;gap:8px;'
           f'background:{c["accent_dim"]};border:1px solid {c["accent"]}44;'
           f'border-radius:8px;padding:7px 12px;margin:0 8px 8px 0;">'
           f'<span style="font-family:JetBrains Mono,monospace;font-size:11px;font-weight:700;'
           f'color:{c["accent"]};">{_medals[_r]}</span>'
           f'<span style="font-family:Exo 2,sans-serif;font-size:12px;font-weight:600;'
           f'color:{c["text_pri"]};">{t_names[_r][:26]}</span>'
           f'<span style="font-family:JetBrains Mono,monospace;font-size:10px;'
           f'color:{c["text_muted"]};">val {_fval:.3f}</span></div>'
       )
   st.markdown(
       f'<div style="font-size:9px;font-weight:700;color:{c["text_muted"]};'
       f'font-family:JetBrains Mono,monospace;letter-spacing:1.5px;margin-bottom:8px;">'
       f'TOP CONTRIBUTING FEATURES</div>'
       + '<div style="display:flex;flex-wrap:wrap;">' + "".join(_badge_parts) + "</div>",
       unsafe_allow_html=True,
   )

   st.plotly_chart(fig_xai, use_container_width=True, theme=None,
                   config={"displayModeBar": False}, key="xai_chart")


   if t_names:
       top1_name = t_names[0]
       top1_val  = float(explain_vec[int(top_idx[0])]) if int(top_idx[0]) < len(explain_vec) else 0.0
       _share = (float(t_vals[0]) / float(t_vals.sum()) * 100) if t_vals.sum() > 0 else 0.0
       st.markdown(f"""
<div style="font-size:12.5px;color:{c['text_sec']};font-family:'Exo 2',sans-serif;line-height:1.7;
           padding:14px 18px;background:{c['threat_box_bg']};border:1px solid {c['border']};
           border-left:3px solid {c['accent']};border-radius:10px;margin-top:10px;">
 <div style="font-family:'JetBrains Mono',monospace;font-size:9px;font-weight:700;
             letter-spacing:1.5px;color:{c['accent']};text-transform:uppercase;margin-bottom:8px;">
   How To Read This
 </div>
 The most influential feature was
 <code style="color:{c['text_pri']};font-size:11.5px;background:{c['accent_dim']};
              padding:1px 6px;border-radius:4px;">{top1_name}</code>
 (observed value <b style="color:{c['text_pri']};">{top1_val:.4f}</b>), accounting for roughly
 <b style="color:{c['accent']};">{_share:.0f}%</b> of the combined importance of the top {top_n} features
 in classifying this flow as <b style="color:{_lbl_clr};">{last_label}</b>.
 <div style="margin-top:8px;font-size:11.5px;color:{c['text_muted']};">
   <span style="color:#ef4444;font-weight:700;">&#9632; Red bars</span> = stronger influence on the decision &nbsp;&middot;&nbsp;
   <span style="color:#38bdf8;font-weight:700;">&#9632; Blue bars</span> = comparatively weaker influence.
   Computed via <b style="color:{_method_clr};">{_method_short}</b>.
 </div>
</div>""", unsafe_allow_html=True)

# LIVE DASHBOARD FRAGMENT
@st.fragment(run_every=render_interval if st.session_state.running else None)
def _render_live_dashboard() -> None:
   c             = T_DARK if st.session_state.theme == "dark" else T_LIGHT
   feature_count = _feature_count


   # BATCH PROCESSING
   if st.session_state.running:


       # --- CSV Simulation ---------------------------------------------------
       if traffic_source == "CSV Simulation" and df_source is not None:
           batch_size   = max(1, int(render_interval / max(throttle_speed, 0.001)))
           feature_cols = [col for col in df_source.columns if col not in EXCLUDE_COLS]
           ts           = datetime.now().strftime("%H:%M:%S")
           batch_n = batch_a = 0


           for _ in range(batch_size):
               if st.session_state.csv_index >= len(df_source):
                   st.session_state.running = False
                   break


               row = df_source.iloc[st.session_state.csv_index]
               raw = row[feature_cols].values.astype(float)
               n   = min(len(raw), feature_count)
               feat = raw[:n].reshape(1, -1)
               if n < feature_count:
                   feat = np.pad(feat, ((0, 0), (0, feature_count - n)))


               scaled = feat
               if active_scaler:
                   try:    scaled = active_scaler.transform(feat)
                   except: scaled = feat


               pred_class = 0
               if model_choice in DL_MODELS:
                   try:
                       inp  = scaled.reshape((1, 1, scaled.shape[1]))
                       prob = active_model.predict(inp, verbose=0)
                       pred_class = (
                           int(prob[0][0] > 0.5) if prob.shape[-1] == 1
                           else int(np.argmax(prob, axis=1)[0])
                       )
                   except: pred_class = 0
               else:
                   try:    pred_class = int(active_model.predict(scaled)[0])
                   except: pred_class = 0


               label = (
                   clean_label(str(classes[pred_class]))
                   if classes is not None and 0 <= pred_class < len(classes)
                   else ("Normal" if pred_class == 0 else "Attack")
               )


               src_ip = row.get("Source IP",
                   f"10.0.{np.random.randint(0,255)}.{np.random.randint(1,254)}")
               port  = safe_int(row.get("Dest Port", row.get(" Destination Port", 80)), 80)
               proto = safe_int(row.get("Protocol", 6), 6)


               _sv = scaled.flatten()
               pool = st.session_state.xai_background_pool
               pool.append(_sv)
               if len(pool) > 100: pool.pop(0)


               st.session_state.metrics["evaluated"] += 1
               if label == "Normal":
                   st.session_state.metrics["safe"] += 1
                   batch_n += 1
                   np_pool = st.session_state.xai_normal_pool
                   np_pool.append(_sv.copy())
                   if len(np_pool) > 100: np_pool.pop(0)
               else:
                   st.session_state.metrics["attacks"] += 1
                   batch_a += 1
                   tv = st.session_state.threat_vectors
                   tv[label] = tv.get(label, 0) + 1
                   st.session_state.last_attack_vector = _sv.copy()
                   st.session_state.last_attack_label  = label


               st.session_state.telemetry_logs.append({
                   "Time": ts, "Source IP": src_ip,
                   "Port": port, "Protocol": proto, "Status": label,
               })
               st.session_state.csv_index += 1


           st.session_state.trend_history.append(
               {"Time": ts, "Normal": batch_n, "Attack": batch_a}
           )


       # --- Live Network Interface -------------------------------------------
       elif traffic_source == "Live Network Interface" and SCAPY_AVAILABLE:
           ts      = datetime.now().strftime("%H:%M:%S")
           batch_n = batch_a = _non_ip_skipped = 0
           flow_tracker: FlowTracker      = st.session_state.flow_tracker
           ps_detector:  PortScanDetector = st.session_state.port_scan_detector
           debug_log: list = st.session_state.setdefault("debug_log", [])


           if not IS_ADMIN:
               debug_log.append(
                   f"[{ts}] Permission denied — administrator/sudo required."
               )
               if len(debug_log) > 200: debug_log.pop(0)
               st.session_state.running = False
           else:
               try:
                   sniff_count = 10 if demo_mode else 5
                   packets = sniff(
                       iface=network_interface,
                       count=sniff_count,
                       timeout=1.0,
                       store=True,
                       filter="ip",
                   )
                   for pkt in packets:
                       if not pkt.haslayer(IP):
                           _non_ip_skipped += 1
                           continue


                       src_ip    = pkt[IP].src
                       proto     = pkt[IP].proto
                       tcp_flags = int(pkt[TCP].flags) if pkt.haslayer(TCP) else 0
                       port      = (
                           pkt[TCP].dport if pkt.haslayer(TCP)
                           else (pkt[UDP].dport if pkt.haslayer(UDP) else 0)
                       )


                       # Port-scan heuristic (runs before ML model)
                       if ps_detector.observe(src_ip, port, tcp_flags):
                           label = "Port-Scan"
                           st.session_state.metrics["evaluated"] += 1
                           st.session_state.metrics["attacks"]   += 1
                           batch_a += 1
                           tv = st.session_state.threat_vectors
                           tv[label] = tv.get(label, 0) + 1
                           st.session_state.telemetry_logs.append({
                               "Time": ts, "Source IP": src_ip,
                               "Port": port, "Protocol": proto, "Status": label,
                           })
                           continue


                       # Feature engineering + ML inference
                       feat = flow_tracker.get_features(pkt, dataset_choice, feature_count)
                       scaled = feat
                       if active_scaler:
                           try:    scaled = active_scaler.transform(feat)
                           except: scaled = feat


                       pred_class = 0
                       if active_model is not None:
                           if model_choice in DL_MODELS:
                               try:
                                   inp  = scaled.reshape((1, 1, scaled.shape[1]))
                                   prob = active_model.predict(inp, verbose=0)
                                   pred_class = (
                                       int(prob[0][0] > 0.5) if prob.shape[-1] == 1
                                       else int(np.argmax(prob, axis=1)[0])
                                   )
                               except: pred_class = 0
                           else:
                               try:    pred_class = int(active_model.predict(scaled)[0])
                               except: pred_class = 0


                       label = (
                           clean_label(str(classes[pred_class]))
                           if classes is not None and 0 <= pred_class < len(classes)
                           else ("Normal" if pred_class == 0 else "Attack")
                       )


                       _sv = scaled.flatten()
                       pool = st.session_state.xai_background_pool
                       pool.append(_sv)
                       if len(pool) > 100: pool.pop(0)


                       st.session_state.metrics["evaluated"] += 1
                       if label == "Normal":
                           st.session_state.metrics["safe"] += 1
                           batch_n += 1
                           np_pool = st.session_state.xai_normal_pool
                           np_pool.append(_sv.copy())
                           if len(np_pool) > 100: np_pool.pop(0)
                       else:
                           st.session_state.metrics["attacks"] += 1
                           batch_a += 1
                           tv = st.session_state.threat_vectors
                           tv[label] = tv.get(label, 0) + 1
                           st.session_state.last_attack_vector = _sv.copy()
                           st.session_state.last_attack_label  = label


                       st.session_state.telemetry_logs.append({
                           "Time": ts, "Source IP": src_ip,
                           "Port": port, "Protocol": proto, "Status": label,
                       })


               except PermissionError as e:
                   debug_log.append(f"[{ts}] PermissionError: {e} — run as Administrator/sudo.")
                   if len(debug_log) > 200: debug_log.pop(0)
                   st.session_state.running = False
               except OSError as e:
                   debug_log.append(f"[{ts}] OSError (interface '{network_interface}'): {e}")
                   if len(debug_log) > 200: debug_log.pop(0)
               except Exception as e:
                   debug_log.append(f"[{ts}] Sniff error: {type(e).__name__}: {e}")
                   if len(debug_log) > 200: debug_log.pop(0)


           if _non_ip_skipped > 0:
               st.session_state["last_non_ip_count"] = _non_ip_skipped


           st.session_state.trend_history.append(
               {"Time": ts, "Normal": batch_n, "Attack": batch_a}
           )


   # DASHBOARD RENDERING
   m_total  = st.session_state.metrics["evaluated"]
   m_safe   = st.session_state.metrics["safe"]
   m_attack = st.session_state.metrics["attacks"]
   m_ratio  = f"{(m_attack / m_total * 100):.1f}%" if m_total > 0 else "0.0%"
   progress = (
       st.session_state.csv_index
       / max(len(df_source) if df_source is not None else 1, 1)
       * 100
   )


   current_dt   = datetime.now().strftime("%d %b %Y,  %I:%M:%S %p")
   is_running   = st.session_state.running
   status_color = c["accent"] if is_running else c["text_muted"]
   status_label = "MONITORING ACTIVE" if is_running else "SYSTEM PAUSED"
   source_badge = (
       "Live Interface" if traffic_source == "Live Network Interface"
       else "CSV Simulation"
   )


   # --- Header ---
   st.markdown(f"""
<div class="nids-header" style="display:flex;justify-content:space-between;align-items:flex-start;
           flex-wrap:wrap;gap:12px;">
 <div style="min-width:0;flex:1;">
   <h1 style="font-family:'Rajdhani',sans-serif;font-size:24px;font-weight:800;
              color:{c['text_pri']};margin:0;text-transform:uppercase;letter-spacing:2px;">
     Real-Time Traffic Monitoring
   </h1>
   <p style="margin:5px 0 0;font-size:12px;color:{c['text_muted']};
             font-family:'Exo 2',sans-serif;line-height:1.6;flex-wrap:wrap;">
     <b style="color:{c['text_sec']};">Mode:</b> {monitoring_mode} &nbsp;&middot;&nbsp;
     <b style="color:{c['text_sec']};">Dataset:</b> {dataset_choice} &nbsp;&middot;&nbsp;
     <b style="color:{c['text_sec']};">Model:</b> {model_choice} &nbsp;&middot;&nbsp;
     <b style="color:{c['text_sec']};">Balancing:</b> {balancing_choice} &nbsp;&middot;&nbsp;
     <b style="color:{c['text_sec']};">Source:</b> {source_badge}
   </p>
 </div>
 <div style="text-align:right;flex-shrink:0;padding-top:2px;">
   <div style="font-size:11px;color:{c['text_muted']};
               font-family:'JetBrains Mono',monospace;margin-bottom:6px;">
     {current_dt}
   </div>
   <span style="background:{c['accent_dim']};border:1px solid {status_color}55;
                color:{status_color};padding:5px 14px 5px 12px;border-radius:20px;
                font-family:'JetBrains Mono',monospace;font-weight:700;
                font-size:10px;letter-spacing:1px;">
     <span class="nids-dot {'live' if is_running else ''}"
           style="background:{status_color};--pulse-c:{status_color}99;"></span>{status_label}
   </span>
 </div>
</div>""", unsafe_allow_html=True)


   # --- KPI Cards (animated count-up) ---
   _prev = st.session_state.get("kpi_prev", {"t": 0, "s": 0, "a": 0})
   _ratio_num = (m_attack / m_total * 100) if m_total > 0 else 0.0
   _atk_accent = "#dc2626" if _ratio_num >= 1 else c["accent"]

   def _kpi(label, accent, from_v, to_v, suffix=""):
       return (
           f'<div class="kpi-card" style="--kpi-accent:{accent};">'
           f'<div class="kpi-label">{label}</div>'
           f'<div class="kpi-value" data-from="{from_v}" data-to="{to_v}" '
           f'data-suffix="{suffix}">{to_v:,}{suffix}</div></div>'
       )

   st.markdown(
       '<div class="kpi-row">'
       + _kpi("Total Packets Evaluated", c["accent"], _prev["t"], m_total)
       + _kpi("Normal Traffic", "#059669", _prev["s"], m_safe)
       + _kpi("Attack Traffic", "#dc2626", _prev["a"], m_attack)
       + f'<div class="kpi-card" style="--kpi-accent:{_atk_accent};">'
         f'<div class="kpi-label">Attack Ratio</div>'
         f'<div class="kpi-value">{m_ratio}</div></div>'
       + '</div>',
       unsafe_allow_html=True,
   )

   st.markdown("""
<script>
(function(){
  function run(doc){
    const els = doc.querySelectorAll('.kpi-value[data-to]');
    els.forEach(el=>{
      const to=parseInt(el.dataset.to||'0',10);
      const from=parseInt(el.dataset.from||'0',10);
      const suf=el.dataset.suffix||'';
      if(to===from){el.textContent=to.toLocaleString()+suf;return;}
      const dur=600, t0=performance.now();
      function step(now){
        const p=Math.min((now-t0)/dur,1);
        const e=1-Math.pow(1-p,3);
        const val=Math.round(from+(to-from)*e);
        el.textContent=val.toLocaleString()+suf;
        if(p<1)requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    });
  }
  try{ run(document); }catch(e){}
  try{ if(window.parent && window.parent.document!==document) run(window.parent.document); }catch(e){}
})();
</script>""", unsafe_allow_html=True)

   st.session_state.kpi_prev = {"t": m_total, "s": m_safe, "a": m_attack}


   # --- Progress bar ---
   if traffic_source == "CSV Simulation":
       st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin:14px 0;">
 <span style="font-size:9px;font-weight:700;color:{c['text_muted']};
              font-family:'JetBrains Mono',monospace;letter-spacing:1.5px;
              white-space:nowrap;width:160px;">
   PROCESSING PROGRESS
 </span>
 <div style="flex:1;height:5px;background:{c['progress_track']};
             border-radius:3px;overflow:hidden;min-width:60px;">
   <div style="width:{progress:.1f}%;height:100%;
               background:linear-gradient(90deg,#0ea5e9,#38bdf8,#7dd3fc);
               box-shadow:0 0 8px rgba(56,189,248,0.5);"></div>
 </div>
 <span style="font-size:12px;font-weight:800;color:{c['accent']};
              width:46px;text-align:right;font-family:'JetBrains Mono',monospace;">
   {progress:.1f}%
 </span>
</div>""", unsafe_allow_html=True)
   else:
       _non_ip = st.session_state.get("last_non_ip_count", 0)
       _non_ip_badge = (
           f'<span style="font-size:10px;color:#f59e0b;background:rgba(245,158,11,0.08);'
           f'border:1px solid rgba(245,158,11,0.3);border-radius:4px;padding:2px 8px;margin-left:8px;">'
           f'{_non_ip} non-IP frame(s) skipped last tick</span>'
           if _non_ip > 0 else ""
       )
       st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin:14px 0;flex-wrap:wrap;">
 <span style="font-size:9px;font-weight:700;color:{c['text_muted']};
              font-family:'JetBrains Mono',monospace;letter-spacing:1.5px;white-space:nowrap;">
   LIVE CAPTURE &nbsp;&middot;&nbsp;
   INTERFACE: <span style="color:{c['accent']};">{network_interface_label}</span>
   {_non_ip_badge}
 </span>
 <div style="flex:1;height:5px;min-width:60px;
             background:linear-gradient(90deg,#0ea5e9,#38bdf8,#7dd3fc,#38bdf8,#0ea5e9);
             background-size:200% 100%;border-radius:3px;
             animation:pulse-bar 2s linear infinite;
             box-shadow:0 0 8px rgba(56,189,248,0.4);"></div>
 <span style="font-size:10px;font-weight:700;color:#10b981;
              font-family:'JetBrains Mono',monospace;">LIVE</span>
</div>
<style>@keyframes pulse-bar{{0%{{background-position:0% 0%}}100%{{background-position:200% 0%}}}}</style>
""", unsafe_allow_html=True)


   # --- Threat signature badges ---
   badge_parts = []
   for k, v in st.session_state.threat_vectors.items():
       safe_k = html_mod.escape(str(k))
       badge_parts.append(
           f'<div style="background:rgba(220,38,38,0.08);color:#ef4444;'
           f'border:1px solid rgba(220,38,38,0.25);padding:5px 10px;border-radius:6px;'
           f'font-size:12px;font-weight:600;display:inline-flex;align-items:center;'
           f'margin:0 6px 8px 0;font-family:Exo 2,sans-serif;">'
           f'{safe_k}'
           f'<span style="background:#dc2626;color:white;border-radius:4px;'
           f'padding:2px 7px;margin-left:8px;font-size:11px;'
           f'font-family:JetBrains Mono,monospace;">{v}</span></div>'
       )
   badges_html = "".join(badge_parts)
   empty_msg = (
       f'<span style="font-size:13px;color:{c["text_muted"]};font-style:italic;">'
       f'Scanning data stream for threat signatures...</span>'
   )
   st.markdown(f"""
<div style="background:{c['threat_box_bg']};border:1px solid {c['border']};
           border-radius:10px;padding:14px 18px;margin-bottom:16px;
           box-shadow:inset 0 2px 8px rgba(0,0,0,0.06);">
 <div style="font-size:9px;font-weight:700;color:{c['accent']};
             font-family:'JetBrains Mono',monospace;letter-spacing:2px;margin-bottom:10px;">
   DETECTED THREAT SIGNATURES
 </div>
 <div style="min-height:42px;display:flex;flex-wrap:wrap;align-content:flex-start;">
   {badges_html if badges_html else empty_msg}
 </div>
</div>""", unsafe_allow_html=True)


   # --- Charts ---
   g1, g2 = st.columns([1, 1.8])


   with g1:
       st.markdown(
           f'<div style="font-size:12px;font-weight:700;color:{c["text_sec"]};'
           f'font-family:JetBrains Mono,monospace;letter-spacing:1px;'
           f'text-transform:uppercase;margin-bottom:4px;">Threat Classification</div>',
           unsafe_allow_html=True,
       )
       if m_total > 0:
           labels = ["Normal"] + list(st.session_state.threat_vectors.keys())
           vals   = [m_safe]   + list(st.session_state.threat_vectors.values())
           colors = ["#059669"] + ["#dc2626"] * len(st.session_state.threat_vectors)
           fig_bar = go.Figure(go.Bar(
               x=vals, y=labels, orientation="h",
               marker=dict(color=colors, line=dict(color="rgba(0,0,0,0.1)", width=1), opacity=0.9),
               text=vals, textposition="outside",
               textfont=dict(family="JetBrains Mono", size=11, color=c["text_sec"]),
           ))
           fig_bar.update_layout(
               **_base_layout(c, height=290),
               xaxis=dict(showgrid=True, gridcolor=c["grid"], showticklabels=False, zeroline=False),
               yaxis=dict(showgrid=False, type="category", automargin=True,
                          tickfont=dict(color=c["tick"], family="JetBrains Mono", size=11)),
               bargap=0.35,
           )
           st.plotly_chart(fig_bar, use_container_width=True, theme=None,
                           config={"displayModeBar": False}, key="bar_chart")
       else:
           st.info("Awaiting traffic data...")


   with g2:
       st.markdown(
           f'<div style="font-size:12px;font-weight:700;color:{c["text_sec"]};'
           f'font-family:JetBrains Mono,monospace;letter-spacing:1px;'
           f'text-transform:uppercase;margin-bottom:4px;">Live Traffic Activity</div>',
           unsafe_allow_html=True,
       )
       if st.session_state.trend_history:
           df_trend = pd.DataFrame(st.session_state.trend_history).tail(45)
           fig_line = go.Figure()
           fig_line.add_trace(go.Scatter(
               x=df_trend["Time"], y=df_trend["Normal"], name="Normal",
               mode="lines", line=dict(color="#059669", width=2),
               fill="tozeroy", fillcolor="rgba(5,150,105,0.07)",
           ))
           fig_line.add_trace(go.Scatter(
               x=df_trend["Time"], y=df_trend["Attack"], name="Attack",
               mode="lines", line=dict(color="#dc2626", width=2),
               fill="tozeroy", fillcolor="rgba(220,38,38,0.07)",
           ))
           fig_line.update_layout(
               **_base_layout(c, height=290),
               yaxis=dict(showgrid=True, gridcolor=c["grid"], zeroline=False,
                          tickfont=dict(color=c["tick"], family="JetBrains Mono")),
               xaxis=dict(showgrid=False, tickangle=-30, nticks=10,
                          tickfont=dict(color=c["tick"], family="JetBrains Mono")),
               legend=dict(
                   orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1,
                   font=dict(color=c["text_sec"], family="JetBrains Mono", size=11),
                   bgcolor="rgba(0,0,0,0)",
               ),
           )
           st.plotly_chart(fig_line, use_container_width=True, theme=None,
                           config={"displayModeBar": False}, key="line_chart")
       else:
           st.info("Awaiting time-series data...")


   # --- Log tables ---
   st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
   tab1, tab2 = st.tabs(["Live Network Logs", "Security Incident Alerts"])


   if st.session_state.telemetry_logs:
       df_logs = pd.DataFrame(st.session_state.telemetry_logs)
       with tab1:
           display = df_logs.copy()
           if telemetry_filter == "Normal Only":
               display = display[display["Status"].str.upper() == "NORMAL"]
           elif telemetry_filter == "Attacks Only":
               display = display[display["Status"].str.upper() != "NORMAL"]
           st.dataframe(
               display.tail(15).style.map(
                   lambda v: (
                       "color:#ef4444;font-weight:700;"
                       if isinstance(v, str) and v.upper() != "NORMAL"
                       else "color:#059669;font-weight:600;"
                   ),
                   subset=["Status"],
               ),
               height=380, use_container_width=True, hide_index=True, key="stream_table",
           )
       with tab2:
           df_alerts = df_logs[df_logs["Status"].str.upper() != "NORMAL"].copy()
           if not df_alerts.empty:
               df_alerts = df_alerts.rename(columns={"Status": "Signature", "Source IP": "IP"})
               df_alerts["Severity"] = "CRITICAL"
               st.dataframe(
                   df_alerts[["Time","Signature","Severity","IP","Port","Protocol"]].tail(15)
                   .style.map(
                       lambda v: (
                           "background:rgba(220,38,38,0.12);color:#ef4444;font-weight:700;"
                           if v == "CRITICAL" else ""
                       ),
                       subset=["Severity"],
                   ),
                   height=380, use_container_width=True, hide_index=True, key="alert_table",
               )
           else:
               st.success("No security incidents detected in the current session.")
   else:
       with tab1:
           st.info("No log entries recorded. Start monitoring to capture traffic.")
       with tab2:
           st.info("No alerts detected. System is ready.")


   # --- XAI Panel ---
   st.markdown('<div style="margin-top:10px;"></div>', unsafe_allow_html=True)
   xai_enabled = st.checkbox(
       "Enable Decision Transparency (XAI / SHAP)",
       value=False,
       key="xai_toggle",
       help="Shows the top features that drove the most recent Attack classification "
            "using SHAP values or sklearn feature_importances_.",
   )
   if xai_enabled:
       _render_xai_panel(
           c, active_model, active_scaler,
           model_choice, dataset_choice, feature_count,
       )

# ENTRY POINT
_render_live_dashboard()