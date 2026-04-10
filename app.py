import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pymarc import MARCReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer
from reportlab.graphics.shapes import Circle, Drawing, Line, Polygon, Rect, String
from skimage import color


APP_TITLE = "Hazardous Pigment Screener for 19th-Century Cloth Bindings"
APP_SUBTITLE = (
    "A LAB-based screening aid for triaging cloth-case bindings into historical hazard colour classes, "
    "with chemistry reserved for XRF or Raman confirmation."
)

COLOR_FAMILIES = [
    "Green",
    "Yellow",
    "Orange",
    "Brown",
    "Red",
    "Blue/Purple",
    "Black/Gray",
    "Unknown",
]

VIVIDNESS_OPTIONS = [
    "Vivid/bright",
    "Moderate",
    "Muted/olive",
    "Unknown",
]

FADING_EVIDENCE_OPTIONS = [
    "Unknown",
    "Possible fading/browning",
    "No obvious fading",
]

BINDING_TYPES = [
    "Original cloth-case binding",
    "Likely original cloth-case binding",
    "Cloth binding, date or originality uncertain",
    "Paper-covered or mixed-material binding",
    "Rebound / later case / not original",
]

REGIONS = [
    "Unknown",
    "United Kingdom / Ireland",
    "United States / Canada",
    "Continental Europe",
    "Australia / New Zealand",
    "Other",
]

ELEMENT_OPTIONS = ["As", "Cu", "Cr", "Pb", "Hg", "Fe", "Zn", "Ca", "S"]
HAZARD_ELEMENTS = {"As", "Pb", "Cr", "Hg"}
DEFAULT_BATCH_SWATCH_HEX = "#8c8271"
DATA_DIR = Path(__file__).resolve().parent / "data"
ISCC_NBS_CSV_PATH = DATA_DIR / "iscc_nbs_lab_colors.csv"
MARC_COLOR_NOTE_FIELDS = ("340", "563", "500", "590")
BINDING_CONTEXT_WORDS = (
    "binding",
    "bound",
    "cloth",
    "bookcloth",
    "case",
    "cover",
    "covers",
    "covering",
    "spine",
    "board",
    "boards",
)

SOURCE_LINKS = [
    (
        "Gil et al. 2023, Analytical Methods: vis-NIR detection of emerald green in 19th-century book bindings",
        "https://pubs.rsc.org/en/content/articlehtml/2023/ay/d3ay01329d",
    ),
    (
        "Vermeulen et al. 2023, Journal of Hazardous Materials: arsenic-based pigments, degradation, and transfer",
        "https://www.sciencedirect.com/science/article/pii/S0304389423007367",
    ),
    (
        "Turner 2023, Journal of Hazardous Materials: lead and mercury in historical books",
        "https://www.sciencedirect.com/science/article/pii/S0304389423012645",
    ),
    (
        "Poison Book Project: safer handling and storage guidance",
        "https://sites.udel.edu/poisonbookproject/handling-and-safety-tips/",
    ),
    (
        "Poison Book Project: emerald green identification methods",
        "https://sites.udel.edu/poisonbookproject/resources/submit-data-to-poison-book-project/",
    ),
    (
        "Poison Book Project: chrome yellow bookcloth",
        "https://sites.udel.edu/poisonbookproject/chrome-yellow-bookcloth/",
    ),
    (
        "Otero et al. 2012, RSC Advances: historical chrome yellow reconstructions with colorimetry",
        "https://pubs.rsc.org/en/content/articlehtml/2012/ra/c1ra00614b",
    ),
    (
        "Poison Book Project: identifying arsenic bookbindings",
        "https://sites.udel.edu/poisonbookproject/arsenic-bookbindings/",
    ),
    (
        "University of Kansas Libraries: heavy metals in original 19th-century bookbindings",
        "https://www.lib.ku.edu/heavy-metals",
    ),
    (
        "Tedone and Grayburn 2023, Toxic Tomes: early findings from the Poison Book Project",
        "https://journals.sagepub.com/doi/abs/10.1177/15501906231159040",
    ),
]

LAB_FIT_STRONG_LOCAL = "Strong local match"
LAB_FIT_CLOSE = "Close match"
LAB_FIT_LOOSE = "Loose match"
LAB_FIT_NONE = "No colour match"
LAB_FIT_PROVISIONAL_CLOSE = "Provisional close match"
LAB_FIT_PROVISIONAL_LOOSE = "Provisional loose match"
PROVISIONAL_LAB_FIT_LABELS = frozenset({LAB_FIT_PROVISIONAL_CLOSE, LAB_FIT_PROVISIONAL_LOOSE})
CLOSE_LAB_FIT_LABELS = frozenset({LAB_FIT_CLOSE, LAB_FIT_PROVISIONAL_CLOSE})
LOOSE_LAB_FIT_LABELS = frozenset({LAB_FIT_LOOSE, LAB_FIT_PROVISIONAL_LOOSE})

DELTA_E00_WORKING_BANDS: Tuple[Tuple[float, int, str, str], ...] = (
    (
        2.0,
        26,
        LAB_FIT_STRONG_LOCAL,
        "Observed LAB falls inside the app's tighter provisional ΔE00 band.",
    ),
    (
        5.0,
        14,
        LAB_FIT_CLOSE,
        "Observed LAB falls inside the app's broader provisional ΔE00 flagging band.",
    ),
    (
        8.0,
        4,
        LAB_FIT_LOOSE,
        "Observed LAB remains near this class, but outside the app's current flagging band.",
    ),
    (
        float("inf"),
        -8,
        LAB_FIT_NONE,
        "Observed LAB falls outside the app's current provisional ΔE00 bands for this class.",
    ),
)


@dataclass(frozen=True)
class LABColor:
    name: str
    L: float
    a: float
    b: float

    def to_rgb(self) -> Tuple[int, int, int]:
        lab_array = np.array([[[self.L, self.a, self.b]]], dtype=float)
        rgb_array = color.lab2rgb(lab_array)
        rgb = np.clip(rgb_array[0, 0] * 255, 0, 255).astype(int)
        return int(rgb[0]), int(rgb[1]), int(rgb[2])

    def to_hex(self) -> str:
        red, green, blue = self.to_rgb()
        return f"#{red:02x}{green:02x}{blue:02x}"

    @classmethod
    def from_rgb(cls, name: str, rgb: Tuple[int, int, int]) -> "LABColor":
        rgb_array = np.array([[[channel / 255.0 for channel in rgb]]], dtype=float)
        lab_array = color.rgb2lab(rgb_array)
        lab = lab_array[0, 0]
        return cls(name=name, L=float(lab[0]), a=float(lab[1]), b=float(lab[2]))

    @classmethod
    def from_hex(cls, name: str, hex_color: str) -> "LABColor":
        cleaned = hex_color.strip().lstrip("#")
        if len(cleaned) != 6:
            raise ValueError("Hex colours must be 6 characters long.")
        rgb = tuple(int(cleaned[index:index + 2], 16) for index in (0, 2, 4))
        return cls.from_rgb(name=name, rgb=rgb)  # type: ignore[arg-type]


@dataclass(frozen=True)
class HazardClassProfile:
    key: str
    label: str
    class_basis: str
    risk_label: str
    primary_families: Tuple[str, ...]
    cluster_key: str
    description: str
    xrf_signature: str
    handling_note: str
    color_data_confidence: str
    use_evidence_confidence: str
    strongest_date_signal: str
    exposure_risk_note: str
    caveat: str


@dataclass(frozen=True)
class ScreeningInput:
    title: str
    publication_year: Optional[int]
    region: str
    binding_type: str
    color_family: str
    vividness: str
    spine_browned: Optional[bool]
    stamped_decoration: Optional[bool]
    xrf_elements: Tuple[str, ...]
    observed_lab: LABColor
    measurement_summary: Optional["MeasurementSummary"] = None
    fading_evidence: str = "Unknown"


@dataclass(frozen=True)
class CandidateAssessment:
    profile: HazardClassProfile
    score: int
    confidence: str
    lab_fit_label: str
    evidence: Tuple[str, ...]
    summary: str
    next_step: str
    elemental_supported: bool
    delta_e00: float
    raw_delta_e00: float
    fading_adjustment: float
    delta_e76: float
    nearest_reference_name: str
    reference_source: str
    reference_count: int
    has_local_reference: bool


@dataclass(frozen=True)
class ScreeningOutcome:
    priority_label: str
    priority_explanation: str
    handling_label: str
    handling_advice: str
    xrf_summary: str
    candidates: Tuple[CandidateAssessment, ...]
    color_note: str
    reference_library_note: str


@dataclass(frozen=True)
class ScenarioSOP:
    key: str
    title: str
    summary: str
    steps: Tuple[str, ...]


@dataclass(frozen=True)
class MARCTemplateRecord:
    record_id: str
    title: str
    year: Optional[int]
    oclc: str
    color_terms: Tuple[str, ...] = tuple()


@dataclass(frozen=True)
class ReferenceSample:
    sample_id: str
    cluster_key: str
    color: LABColor
    source: str
    condition: str = ""
    notes: str = ""
    is_local: bool = False


@dataclass(frozen=True)
class MeasurementSummary:
    count: int
    mean_lab: LABColor
    std_l: float
    std_a: float
    std_b: float
    mean_delta_e00: float
    max_delta_e00: float


@dataclass(frozen=True)
class ReferenceMatch:
    sample: ReferenceSample
    effective_delta_e00: float
    raw_delta_e00: float
    delta_e76: float
    fading_adjustment: float = 0.0
    fading_note: str = ""


@dataclass(frozen=True)
class LABClusterModel:
    key: str
    centroid: LABColor
    exemplar_terms: Tuple[str, ...]
    mean_distance: float
    max_distance: float
    std_a: float
    std_b: float
    references: Tuple[ReferenceSample, ...]


LAB_CLASS_EXEMPLARS: Dict[str, Tuple[str, ...]] = {
    "emerald_green": (
        "Vivid green",
        "Brilliant green",
        "Strong green",
        "Vivid bluish green",
        "Brilliant bluish green",
    ),
    "chrome_green": (
        "Strong olive green",
        "Moderate olive green",
        "Dark olive green",
        "Grayish olive green",
        "Deep yellowish green",
        "Moderate yellowish green",
    ),
    "chrome_yellow": (
        "Vivid yellow",
        "Brilliant yellow",
        "Strong yellow",
        "Deep yellow",
        "Brilliant orange yellow",
        "Strong orange yellow",
    ),
    "chrome_orange": (
        "Vivid orange",
        "Strong orange",
        "Deep orange",
        "Vivid reddish orange",
        "Strong reddish orange",
        "Deep reddish orange",
    ),
    "mercury_red": (
        "Vivid red",
        "Strong red",
        "Deep red",
        "Very deep red",
        "Vivid reddish orange",
        "Strong reddish orange",
    ),
}

CHROME_YELLOW_LITERATURE_REFERENCES: Tuple[Tuple[str, float, float, float], ...] = (
    (
        "Historic chrome yellow reconstruction (reported lower Lab bound)",
        76.0,
        23.0,
        71.0,
    ),
    (
        "Historic chrome yellow reconstruction (reported upper Lab bound)",
        85.0,
        30.0,
        89.0,
    ),
)


HAZARD_CLASS_PROFILES: Dict[str, HazardClassProfile] = {
    "emerald_green": HazardClassProfile(
        key="emerald_green",
        label="Emerald-green-like saturated green class",
        class_basis="Saturated to bluish saturated green",
        risk_label="Arsenic-associated visual class",
        primary_families=("Green",),
        cluster_key="emerald_green",
        description=(
            "Use this LAB class to triage vivid Victorian greens. The class is visually associated "
            "with emerald-green-looking cloth, but compound identity remains unconfirmed even when As + Cu are measured."
        ),
        xrf_signature="As + Cu on green cloth; confirm compound identity separately",
        handling_note="Treat as potentially friable and capable of transfer.",
        color_data_confidence="Low: open bookcloth-specific LAB baselines remain sparse.",
        use_evidence_confidence="High: documented on Victorian starch-coated bookcloth in English and American imprints.",
        strongest_date_signal="1840s-1860s, especially vivid green cloth with stamped decoration.",
        exposure_risk_note="High intrinsic toxicity and the clearest practical exposure concern in this app because friable surface pigment can transfer.",
        caveat=(
            "This is a colorimetric class, not a pigment reference standard. XRF provides elemental "
            "support only, and Raman, FTIR, or XRD is needed for compound-level confirmation."
        ),
    ),
    "chrome_green": HazardClassProfile(
        key="chrome_green",
        label="Chrome-green-like olive green class",
        class_basis="Olive to yellowish green",
        risk_label="Lead/chromium-associated visual class",
        primary_families=("Green",),
        cluster_key="chrome_green",
        description=(
            "Use this LAB class for darker, olive, and yellowish greens that are visually closer to "
            "historical chrome-green-looking cloth than to saturated emerald greens, often described as "
            "lead chromate mixed with Prussian blue."
        ),
        xrf_signature="Pb + Cr on green cloth, sometimes with Fe support",
        handling_note="Lead/chromium system; current project guidance suggests lower transfer than emerald green.",
        color_data_confidence="Low: no open mixture-specific LAB dataset was located for chrome-green cloth.",
        use_evidence_confidence="High: bookcloth-specific sources identify chrome green as a common Pb/Cr green pathway.",
        strongest_date_signal="Broad 19th-century signal; interpret alongside Pb + Cr and any Fe support for Prussian blue.",
        exposure_risk_note="High intrinsic heavy-metal hazard, but current handling guidance suggests lower offset risk than friable emerald green.",
        caveat=(
            "This class captures visual similarity only. Pb + Cr is an elemental screen, not compound-level "
            "identification, and the mixture ratio can shift the hue substantially."
        ),
    ),
    "chrome_yellow": HazardClassProfile(
        key="chrome_yellow",
        label="Chrome-yellow-like yellow/orange class",
        class_basis="Strong yellow to orange-yellow",
        risk_label="Lead/chromium-associated visual class",
        primary_families=("Yellow", "Orange", "Brown"),
        cluster_key="chrome_yellow",
        description=(
            "Use this LAB class for late-century yellow and yellow-orange cloth that could warrant "
            "Pb/Cr screening."
        ),
        xrf_signature="Pb + Cr on yellow, orange, or brown cloth",
        handling_note="Lead/chromium system; avoid ingestion and abrasive handling.",
        color_data_confidence="Medium: literature-derived historical reconstruction LAB anchors are available, but they are not bookcloth measurements.",
        use_evidence_confidence="High: directly identified on cloth-case bindings, with stronger reported use in the 1880s-1890s.",
        strongest_date_signal="1880s-1890s yellow, orange, or brown cloth-case bindings.",
        exposure_risk_note="High intrinsic heavy-metal hazard, but current project guidance suggests tighter adhesion and lower transfer than emerald green.",
        caveat="Hue varies widely with formulation and ageing, so this class is a broad visual grouping only.",
    ),
    "mercury_red": HazardClassProfile(
        key="mercury_red",
        label="Mercury-red-like red class",
        class_basis="Strong red to deep red",
        risk_label="Mercury watchlist visual class",
        primary_families=("Red",),
        cluster_key="mercury_red",
        description=(
            "Use this class as a conservative watchlist for red cloth or panels. It is intentionally "
            "weaker than the green and Pb/Cr classes because the cloth-specific evidence base is thinner."
        ),
        xrf_signature="Hg on a red component",
        handling_note="Consult institutional health and safety guidance if Hg is detected.",
        color_data_confidence="Low: bundled anchors are heuristic hue proxies only.",
        use_evidence_confidence="Low to medium: mercury is documented in historical books, but the cloth-specific evidence base is limited.",
        strongest_date_signal="Victorian-era red cloth can justify testing, but the signal is weaker than the green and Pb/Cr classes.",
        exposure_risk_note="Treat intrinsic hazard as meaningful if Hg is detected, but rely on specialist interpretation before inferring exposure risk from colour alone.",
        caveat="This is a watchlist class, not a vermilion identification, and Hg alone does not identify the exact pigment.",
    ),
}


YEAR_PATTERN = re.compile(r"(1[78]\d{2}|19\d{2})")


def inject_css() -> None:
    st.markdown(
        """
        <style>
            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(circle at 12% 18%, rgba(77, 109, 90, 0.18), transparent 28%),
                    radial-gradient(circle at 88% 8%, rgba(153, 106, 43, 0.16), transparent 25%),
                    linear-gradient(180deg, #f7f1e5 0%, #efe3cf 54%, #e8d8bf 100%);
                color: #201914;
            }
            .block-container {
                padding-top: 1.6rem;
                padding-bottom: 2.2rem;
                max-width: 1160px;
            }
            h1, h2, h3 {
                font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
                color: #201914;
                letter-spacing: 0.01em;
            }
            p, li, div, span, label {
                font-family: "Avenir Next", "Gill Sans", "Trebuchet MS", sans-serif;
            }
            .hero {
                background:
                    linear-gradient(135deg, rgba(30, 39, 35, 0.96), rgba(95, 70, 40, 0.92)),
                    linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0));
                border: 1px solid rgba(106, 78, 44, 0.35);
                border-radius: 22px;
                color: #f8f1e5;
                padding: 1.6rem 1.8rem 1.4rem 1.8rem;
                box-shadow: 0 18px 40px rgba(57, 37, 16, 0.12);
                margin-bottom: 1.1rem;
            }
            .hero-kicker {
                display: inline-block;
                font-size: 0.78rem;
                letter-spacing: 0.18em;
                text-transform: uppercase;
                color: #d8c39a;
                margin-bottom: 0.35rem;
            }
            .hero h1 {
                color: #fff7e9;
                margin-bottom: 0.25rem;
            }
            .hero p {
                margin: 0.2rem 0;
                color: #efe4d1;
                max-width: 52rem;
                line-height: 1.45;
            }
            .note-card {
                background: rgba(255, 249, 239, 0.72);
                border: 1px solid rgba(129, 93, 48, 0.18);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                box-shadow: 0 8px 20px rgba(73, 45, 18, 0.06);
            }
            .priority-card {
                position: relative;
                overflow: hidden;
                border-radius: 18px;
                padding: 1.15rem 1.2rem 1.2rem;
                border: 1px solid rgba(76, 59, 33, 0.14);
                box-shadow: 0 14px 30px rgba(66, 42, 18, 0.1);
                margin-bottom: 0.8rem;
            }
            .priority-card::after {
                content: "";
                position: absolute;
                inset: 0;
                background: linear-gradient(180deg, rgba(255,255,255,0.08), rgba(255,255,255,0));
                pointer-events: none;
            }
            .priority-card > * {
                position: relative;
                z-index: 1;
            }
            .priority-card-header {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 0.85rem;
                margin-bottom: 0.45rem;
            }
            .priority-card .smallcaps {
                color: rgba(237, 218, 189, 0.72);
            }
            .priority-card p {
                margin: 0;
                max-width: 44rem;
                color: rgba(255, 247, 234, 0.92);
                line-height: 1.5;
                font-size: 1rem;
            }
            .priority-chip {
                display: inline-flex;
                align-items: center;
                white-space: nowrap;
                border-radius: 999px;
                padding: 0.28rem 0.72rem;
                border: 1px solid rgba(255,255,255,0.18);
                background: rgba(255,255,255,0.08);
                color: rgba(255, 247, 234, 0.88);
                font-size: 0.76rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
                gap: 0.85rem;
                margin-bottom: 0.55rem;
            }
            .metric-grid-compact .metric-card-value {
                font-size: 1.02rem;
            }
            .metric-card {
                background: rgba(255, 250, 242, 0.84);
                border: 1px solid rgba(120, 91, 42, 0.16);
                border-radius: 18px;
                padding: 0.95rem 1rem 0.9rem;
                min-width: 0;
                height: 100%;
                display: flex;
                flex-direction: column;
                justify-content: flex-start;
                box-shadow: 0 8px 18px rgba(66, 42, 18, 0.06);
            }
            .metric-card-label {
                font-size: 0.76rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #7b6141;
                margin-bottom: 0.45rem;
            }
            .metric-card-value {
                font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
                font-size: 1.4rem;
                line-height: 1.12;
                color: #201914;
                overflow-wrap: break-word;
                text-wrap: balance;
            }
            .metric-card-note {
                margin-top: 0.65rem;
                font-size: 0.88rem;
                line-height: 1.4;
                color: #6d5838;
            }
            .metric-card-signal .metric-card-value {
                font-family: "Avenir Next", "Gill Sans", "Trebuchet MS", sans-serif;
                font-size: 1rem;
            }
            .support-note {
                background: rgba(255, 249, 239, 0.68);
                border: 1px solid rgba(129, 93, 48, 0.18);
                border-left: 4px solid rgba(129, 93, 48, 0.34);
                border-radius: 16px;
                padding: 0.9rem 1rem;
                margin: 0.3rem 0 0.9rem;
                box-shadow: 0 8px 20px rgba(73, 45, 18, 0.04);
            }
            .support-note p {
                margin: 0;
                color: #5e4b31;
                line-height: 1.5;
                font-size: 0.93rem;
            }
            .support-note .smallcaps {
                display: block;
                margin-bottom: 0.35rem;
            }
            .support-note p + p {
                margin-top: 0.55rem;
            }
            .priority-card h3 {
                margin-top: 0;
                margin-bottom: 0.7rem;
                font-size: clamp(1.85rem, 2.8vw, 2.35rem);
                line-height: 1.06;
                max-width: 16ch;
            }
            .priority-urgent {
                background: linear-gradient(135deg, rgba(141, 33, 33, 0.95), rgba(92, 21, 21, 0.95));
                color: #fff4f2;
            }
            .priority-high {
                background: linear-gradient(135deg, rgba(120, 74, 22, 0.95), rgba(79, 52, 18, 0.95));
                color: #fff7ea;
            }
            .priority-moderate {
                background: linear-gradient(135deg, rgba(73, 98, 77, 0.95), rgba(42, 64, 50, 0.95));
                color: #f3faef;
            }
            .priority-low {
                background: linear-gradient(135deg, rgba(73, 84, 90, 0.95), rgba(45, 53, 58, 0.95));
                color: #eef5f8;
            }
            .candidate-card {
                background: rgba(255, 250, 242, 0.82);
                border: 1px solid rgba(120, 91, 42, 0.18);
                border-radius: 18px;
                padding: 1rem 1.1rem;
                margin-bottom: 0.85rem;
            }
            .candidate-card h4 {
                margin: 0 0 0.25rem 0;
                font-family: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
            }
            .candidate-meta {
                font-size: 0.92rem;
                color: #59452b;
                margin-bottom: 0.55rem;
            }
            .signal-pill {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                border-radius: 999px;
                padding: 0.18rem 0.62rem;
                font-size: 0.8rem;
                font-weight: 600;
                letter-spacing: 0.04em;
                text-transform: uppercase;
                border: 1px solid transparent;
            }
            .signal-green {
                background: rgba(63, 110, 72, 0.14);
                color: #2d5a38;
                border-color: rgba(63, 110, 72, 0.28);
            }
            .signal-amber {
                background: rgba(153, 106, 43, 0.14);
                color: #8b5a15;
                border-color: rgba(153, 106, 43, 0.3);
            }
            .signal-red {
                background: rgba(146, 49, 49, 0.12);
                color: #8a2d2d;
                border-color: rgba(146, 49, 49, 0.26);
            }
            .smallcaps {
                font-size: 0.78rem;
                letter-spacing: 0.13em;
                text-transform: uppercase;
                color: #7f6640;
            }
            .source-list a {
                color: #184f5d;
                text-decoration: none;
            }
            @media (max-width: 720px) {
                .priority-card {
                    padding: 1rem 1rem 1.05rem;
                }
                .priority-card-header {
                    flex-direction: column;
                    gap: 0.45rem;
                }
                .priority-card h3 {
                    max-width: none;
                }
                .metric-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-kicker">Bibliotoxicology Workflow</div>
            <h1>{APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
            <p>
                This app is deliberately conservative: it helps you prioritise books for safer handling
                and instrumental confirmation, but it does not claim pigment identification from LAB alone.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def parse_optional_year(raw_value: Any) -> Optional[int]:
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    match = YEAR_PATTERN.search(text)
    if not match:
        return None
    year = int(match.group(1))
    if 1700 <= year <= 1999:
        return year
    return None


def to_bool_or_none(raw_value: Any) -> Optional[bool]:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, (int, float)) and not pd.isna(raw_value):
        if raw_value == 1:
            return True
        if raw_value == 0:
            return False
    text = str(raw_value).strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return True
    if text in {"no", "n", "false", "0"}:
        return False
    return None


def parse_xrf_elements(raw_value: Any) -> Tuple[str, ...]:
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return tuple()
    if isinstance(raw_value, (list, tuple, set)):
        tokens = [str(value).strip() for value in raw_value]
    else:
        separators = re.split(r"[,;/|]", str(raw_value))
        tokens = [token.strip() for token in separators]
    normalised = []
    for token in tokens:
        cleaned = token.title()
        if cleaned in ELEMENT_OPTIONS and cleaned not in normalised:
            normalised.append(cleaned)
    return tuple(normalised)


def normalize_color_name(raw_value: Any) -> str:
    text = str(raw_value).strip().lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\bcolou?r\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@st.cache_data(show_spinner=False)
def load_iscc_nbs_lookup() -> pd.DataFrame:
    df = pd.read_csv(ISCC_NBS_CSV_PATH)
    required_columns = {"L", "A", "B", "Color Name"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"ISCC-NBS lookup is missing columns: {', '.join(sorted(missing))}")
    df = df.copy()
    df["Color Name"] = df["Color Name"].astype(str).str.strip()
    df["normalized_name"] = df["Color Name"].map(normalize_color_name)
    return df


def lookup_named_color(color_name: str) -> Optional[LABColor]:
    normalized = normalize_color_name(color_name)
    if not normalized:
        return None
    lookup = load_iscc_nbs_lookup()
    matched = lookup[lookup["normalized_name"] == normalized]
    if matched.empty:
        return None
    row = matched.iloc[0]
    return LABColor(
        name=str(row["Color Name"]),
        L=float(row["L"]),
        a=float(row["A"]),
        b=float(row["B"]),
    )


def infer_named_color_from_lab(observed_lab: LABColor) -> Tuple[LABColor, float]:
    lookup = load_iscc_nbs_lookup()
    lookup_lab = lookup[["L", "A", "B"]].to_numpy(dtype=float).reshape(-1, 1, 3)
    observed_lab_array = np.array([[[observed_lab.L, observed_lab.a, observed_lab.b]]], dtype=float)
    tiled_observed = np.repeat(observed_lab_array, repeats=len(lookup), axis=0)
    distances = pd.Series(
        color.deltaE_ciede2000(lookup_lab, tiled_observed).reshape(-1),
        index=lookup.index,
    )
    index = int(distances.idxmin())
    row = lookup.loc[index]
    return (
        LABColor(
            name=str(row["Color Name"]),
            L=float(row["L"]),
            a=float(row["A"]),
            b=float(row["B"]),
        ),
        float(distances.loc[index]),
    )


def infer_color_family_from_name(color_name: str) -> str:
    normalized = normalize_color_name(color_name)
    if not normalized:
        return "Unknown"
    if "green" in normalized:
        return "Green"
    if any(token in normalized for token in ("pink", "red", "rose")):
        return "Red"
    if any(token in normalized for token in ("blue", "violet", "purple")):
        return "Blue/Purple"
    if any(token in normalized for token in ("gray", "grey", "black", "white")):
        return "Black/Gray"
    if "orange" in normalized:
        return "Orange"
    if "yellow" in normalized:
        return "Yellow"
    if any(token in normalized for token in ("brown", "olive")):
        return "Brown"
    return "Unknown"


def parse_lab_triplet(raw_value: str) -> Tuple[float, float, float]:
    parts = [part for part in re.split(r"[\s,;]+", raw_value.strip()) if part]
    if len(parts) != 3:
        raise ValueError("Each LAB entry must contain exactly three numbers: L, a, b.")
    l_value, a_value, b_value = (float(part) for part in parts)
    if not 0.0 <= l_value <= 100.0:
        raise ValueError("L* must be between 0 and 100.")
    if not -128.0 <= a_value <= 127.0:
        raise ValueError("a* must be between -128 and 127.")
    if not -128.0 <= b_value <= 127.0:
        raise ValueError("b* must be between -128 and 127.")
    return l_value, a_value, b_value


def parse_replicate_measurements_text(raw_text: str, name_prefix: str) -> Tuple[LABColor, ...]:
    measurements: List[LABColor] = []
    for index, line in enumerate(raw_text.splitlines(), start=1):
        cleaned = line.strip()
        if not cleaned:
            continue
        l_value, a_value, b_value = parse_lab_triplet(cleaned)
        measurements.append(
            LABColor(
                name=f"{name_prefix} measurement {index}",
                L=l_value,
                a=a_value,
                b=b_value,
            )
        )
    return tuple(measurements)


def parse_replicate_measurements_field(raw_value: Any, name_prefix: str) -> Tuple[LABColor, ...]:
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return tuple()
    measurements: List[LABColor] = []
    for index, chunk in enumerate(str(raw_value).split("|"), start=1):
        cleaned = chunk.strip()
        if not cleaned:
            continue
        l_value, a_value, b_value = parse_lab_triplet(cleaned)
        measurements.append(
            LABColor(
                name=f"{name_prefix} measurement {index}",
                L=l_value,
                a=a_value,
                b=b_value,
            )
        )
    return tuple(measurements)


def summarize_measurements(measurements: Sequence[LABColor], name: str) -> Optional[MeasurementSummary]:
    if not measurements:
        return None
    lab_values = np.array([[sample.L, sample.a, sample.b] for sample in measurements], dtype=float)
    mean_values = lab_values.mean(axis=0)
    mean_lab = LABColor(
        name=name,
        L=float(mean_values[0]),
        a=float(mean_values[1]),
        b=float(mean_values[2]),
    )
    distances = [delta_e_ciede2000(sample, mean_lab) for sample in measurements]
    return MeasurementSummary(
        count=len(measurements),
        mean_lab=mean_lab,
        std_l=float(lab_values[:, 0].std(ddof=0)),
        std_a=float(lab_values[:, 1].std(ddof=0)),
        std_b=float(lab_values[:, 2].std(ddof=0)),
        mean_delta_e00=float(np.mean(distances)) if distances else 0.0,
        max_delta_e00=float(np.max(distances)) if distances else 0.0,
    )


def example_reference_library_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "reference_id": "eg-001",
                "reference_name": "Measure your confirmed emerald green cloth A",
                "cluster_key": "emerald_green",
                "lab_l": None,
                "lab_a": None,
                "lab_b": None,
                "source": "Your instrument and workflow",
                "condition": "Describe substrate and ageing state",
                "notes": (
                    "Enter same-device LAB measured from a confirmed or institutionally trusted analogue. "
                    "The repository does not ship a built-in local confirmed bookcloth LAB dataset for this class."
                ),
            },
            {
                "reference_id": "cg-001",
                "reference_name": "Measure your confirmed chrome green cloth A",
                "cluster_key": "chrome_green",
                "lab_l": None,
                "lab_a": None,
                "lab_b": None,
                "source": "Your instrument and workflow",
                "condition": "Describe substrate and ageing state",
                "notes": (
                    "Enter same-device LAB measured from a confirmed or institutionally trusted analogue. "
                    "The repository does not ship a built-in local confirmed bookcloth LAB dataset for this class."
                ),
            },
            {
                "reference_id": "cy-001",
                "reference_name": "Measure your confirmed chrome yellow cloth A",
                "cluster_key": "chrome_yellow",
                "lab_l": None,
                "lab_a": None,
                "lab_b": None,
                "source": "Your instrument and workflow",
                "condition": "Describe substrate and ageing state",
                "notes": (
                    "Enter same-device LAB measured from a confirmed or institutionally trusted analogue. "
                    "The built-in fallback for this class uses literature-derived historical chrome-yellow "
                    "anchors, not local bookcloth measurements."
                ),
            },
            {
                "reference_id": "mr-001",
                "reference_name": "Measure your confirmed mercury red cloth A",
                "cluster_key": "mercury_red",
                "lab_l": None,
                "lab_a": None,
                "lab_b": None,
                "source": "Your instrument and workflow",
                "condition": "Describe substrate and ageing state",
                "notes": (
                    "Enter same-device LAB measured from a confirmed or institutionally trusted analogue. "
                    "The repository does not ship a built-in local confirmed bookcloth LAB dataset for this class."
                ),
            },
        ]
    )


def parse_reference_library_csv(content: bytes) -> Tuple[ReferenceSample, ...]:
    df = pd.read_csv(io.BytesIO(content))
    required_columns = {"cluster_key", "reference_name", "lab_l", "lab_a", "lab_b"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Reference library CSV is missing columns: {', '.join(sorted(missing))}"
        )

    samples: List[ReferenceSample] = []
    valid_keys = set(HAZARD_CLASS_PROFILES)
    for index, row in df.iterrows():
        cluster_key = str(row["cluster_key"]).strip()
        if cluster_key not in valid_keys:
            raise ValueError(
                f"Invalid cluster_key on row {index + 2}: {cluster_key}. "
                f"Use one of {', '.join(sorted(valid_keys))}."
            )
        reference_name = str(row["reference_name"]).strip() or f"{cluster_key} reference {index + 1}"
        lab_l = float(pd.to_numeric(row["lab_l"], errors="raise"))
        lab_a = float(pd.to_numeric(row["lab_a"], errors="raise"))
        lab_b = float(pd.to_numeric(row["lab_b"], errors="raise"))
        if not (0.0 <= lab_l <= 100.0 and -128.0 <= lab_a <= 127.0 and -128.0 <= lab_b <= 127.0):
            raise ValueError(
                f"Invalid LAB values on row {index + 2}. Expected L* 0-100 and a*/b* in -128 to 127."
            )
        sample_id = str(row.get("reference_id", "")).strip() or f"{cluster_key}-{index + 1}"
        source = str(row.get("source", "")).strip() or "Uploaded local reference library"
        condition = str(row.get("condition", "")).strip()
        notes = str(row.get("notes", "")).strip()
        samples.append(
            ReferenceSample(
                sample_id=sample_id,
                cluster_key=cluster_key,
                color=LABColor(name=reference_name, L=lab_l, a=lab_a, b=lab_b),
                source=source,
                condition=condition,
                notes=notes,
                is_local=True,
            )
        )
    if not samples:
        raise ValueError("Reference library CSV contained no reference rows.")
    return tuple(samples)


@st.cache_data(show_spinner=False)
def get_default_reference_samples() -> Tuple[ReferenceSample, ...]:
    samples: List[ReferenceSample] = []
    for cluster_key in HAZARD_CLASS_PROFILES:
        if cluster_key == "chrome_yellow":
            for index, (label, lab_l, lab_a, lab_b) in enumerate(
                CHROME_YELLOW_LITERATURE_REFERENCES,
                start=1,
            ):
                samples.append(
                    ReferenceSample(
                        sample_id=f"{cluster_key}-lit-{index}",
                        cluster_key=cluster_key,
                        color=LABColor(name=label, L=lab_l, a=lab_a, b=lab_b),
                        source="Built-in literature-derived historical reconstruction",
                        condition="Historic chrome yellow reconstruction",
                        notes=(
                            "Otero et al. 2012 reported Lab* coordinates for reconstructed 19th-century "
                            "chrome yellows spanning (76, 23, 71) to (85, 30, 89). Used here as "
                            "provisional fallback anchors, not as bookcloth measurements."
                        ),
                        is_local=False,
                    )
                )
            continue
        terms = LAB_CLASS_EXEMPLARS[cluster_key]
        for index, term in enumerate(terms, start=1):
            lab_color = lookup_named_color(term)
            if lab_color is None:
                raise ValueError(f"Bundled reference term not found in ISCC-NBS lookup: {term}")
            samples.append(
                ReferenceSample(
                    sample_id=f"{cluster_key}-{index}",
                    cluster_key=cluster_key,
                    color=lab_color,
                    source="Bundled ISCC-NBS fallback exemplar",
                    condition="Generic lookup term",
                    notes="Use a same-device local reference library when available.",
                    is_local=False,
                )
            )
    return tuple(samples)


def built_in_fallback_anchor_phrase(cluster_key: str) -> str:
    if cluster_key == "chrome_yellow":
        return "built-in literature-derived historical chrome-yellow anchors"
    return "bundled ISCC-NBS hue proxies"


def built_in_fallback_caveat(cluster_key: str) -> str:
    if cluster_key == "chrome_yellow":
        return (
            "Built-in chrome-yellow anchors are drawn from published historical pigment reconstructions, "
            "but they are not same-device bookcloth measurements."
        )
    if cluster_key == "mercury_red":
        return (
            "Bundled mercury-red anchors are heuristic hue proxies because the cloth-specific evidence base "
            "is thin and the app does not ship an empirical LAB set for this class."
        )
    return (
        "Bundled fallback anchors for this class are generic hue proxies rather than chemically confirmed "
        "bookcloth measurements."
    )


def built_in_reference_detail(cluster_key: str) -> str:
    if cluster_key == "chrome_yellow":
        return (
            "Built-in fallback basis: historical chrome-yellow Lab* bounds from Otero et al. 2012, "
            "reported as (76, 23, 71) to (85, 30, 89). These are historical reconstructions rather than "
            "bookcloth measurements."
        )
    if cluster_key == "emerald_green":
        return (
            "Built-in fallback basis: ISCC-NBS proxy hue terms selected to mirror the literature's "
            "vivid to bluish emerald-green bookcloth descriptions because an open cloth-specific LAB dataset "
            "was not located: "
            f"{format_anchor_exemplar_terms(cluster_key)}."
        )
    if cluster_key == "chrome_green":
        return (
            "Built-in fallback basis: ISCC-NBS proxy hue terms selected to mirror published olive, "
            "yellowish, and darker chrome-green bookcloth descriptions. Open mixture-specific chrome-green "
            "LAB data were not located, so these remain proxies: "
            f"{format_anchor_exemplar_terms(cluster_key)}."
        )
    return (
        "Built-in fallback basis: ISCC-NBS proxy hue terms used as a conservative watchlist anchor set: "
        f"{format_anchor_exemplar_terms(cluster_key)}."
    )


def reference_library_note(reference_samples: Sequence[ReferenceSample]) -> str:
    local_count = sum(1 for sample in reference_samples if sample.is_local)
    fallback_count = sum(1 for sample in reference_samples if not sample.is_local)
    if local_count and fallback_count:
        return (
            "Using a mixed reference library: uploaded local references where provided, plus built-in "
            "fallback anchors for any missing classes. The chrome-yellow-like class falls back to "
            "literature-derived historical reconstruction LAB anchors, while the emerald-green-like, "
            "chrome-green-like, and mercury-red-like classes still fall back to ISCC-NBS hue proxies. "
            "Classes without uploaded local references remain provisional because the fallback rows are "
            "not same-device chemically confirmed bookcloth measurements."
        )
    if local_count:
        return (
            "Using uploaded local reference measurements. ΔE00 is calculated to the nearest reference "
            "measured in your own library."
        )
    return (
        "Using built-in fallback references only. The chrome-yellow-like class uses literature-derived "
        "historical reconstruction LAB anchors; the emerald-green-like, chrome-green-like, and "
        "mercury-red-like classes use ISCC-NBS hue proxies. None of these are same-device bookcloth "
        "measurements, so colour matches remain provisional until you upload a same-device local "
        "reference library."
    )


def resolve_reference_samples(reference_library_bytes: Optional[bytes]) -> Tuple[ReferenceSample, ...]:
    default_samples = get_default_reference_samples()
    if not reference_library_bytes:
        return default_samples

    uploaded_samples = parse_reference_library_csv(reference_library_bytes)
    uploaded_keys = {sample.cluster_key for sample in uploaded_samples}
    supplemented_samples = list(uploaded_samples)
    supplemented_samples.extend(
        sample for sample in default_samples if sample.cluster_key not in uploaded_keys
    )
    return tuple(supplemented_samples)


def class_reference_counts(
    cluster_key: str, reference_samples: Sequence[ReferenceSample]
) -> Tuple[int, int]:
    local_count = sum(
        1 for sample in reference_samples if sample.cluster_key == cluster_key and sample.is_local
    )
    fallback_count = sum(
        1 for sample in reference_samples if sample.cluster_key == cluster_key and not sample.is_local
    )
    return local_count, fallback_count


def class_has_local_reference(
    cluster_key: str, reference_samples: Sequence[ReferenceSample]
) -> bool:
    local_count, _ = class_reference_counts(cluster_key, reference_samples)
    return local_count > 0


def reference_basis_label(has_local_reference: bool) -> str:
    return "Uploaded local references" if has_local_reference else "Built-in fallback anchors"


def colour_closeness_signal(assessment: CandidateAssessment) -> Tuple[str, str, str]:
    if assessment.lab_fit_label == LAB_FIT_STRONG_LOCAL:
        return (
            "Green",
            "signal-green",
            "Strong colour proximity to uploaded local references.",
        )
    if assessment.lab_fit_label in CLOSE_LAB_FIT_LABELS:
        if assessment.has_local_reference:
            return (
                "Amber",
                "signal-amber",
                "Colour is near the class references, but still needs confirmatory testing.",
            )
        return (
            "Amber",
            "signal-amber",
            "Colour is near the fallback anchors, but the match is still provisional.",
        )
    if assessment.lab_fit_label in LOOSE_LAB_FIT_LABELS:
        if assessment.has_local_reference:
            return (
                "Amber",
                "signal-amber",
                "Only loose colour similarity to the current class references.",
            )
        return (
            "Amber",
            "signal-amber",
            "Only loose colour similarity to the fallback anchors for this class.",
        )
    return (
        "Red",
        "signal-red",
        "No meaningful colour closeness from the current references alone.",
    )


def prettify_color_name(color_name: str) -> str:
    normalized = normalize_color_name(color_name)
    return normalized.title() if normalized else str(color_name).strip()


def compute_cluster_model(
    cluster_key: str, reference_samples: Sequence[ReferenceSample]
) -> LABClusterModel:
    selected = tuple(sample for sample in reference_samples if sample.cluster_key == cluster_key)
    if not selected:
        raise ValueError(f"No reference samples available for cluster {cluster_key}.")

    lab_values = np.array(
        [[sample.color.L, sample.color.a, sample.color.b] for sample in selected],
        dtype=float,
    )
    centroid = LABColor(
        name=f"{cluster_key.replace('_', ' ').title()} centroid",
        L=float(lab_values[:, 0].mean()),
        a=float(lab_values[:, 1].mean()),
        b=float(lab_values[:, 2].mean()),
    )
    distances = [delta_e_ciede2000(sample.color, centroid) for sample in selected]
    exemplar_terms = tuple(sample.color.name for sample in selected)
    return LABClusterModel(
        key=cluster_key,
        centroid=centroid,
        exemplar_terms=exemplar_terms,
        mean_distance=float(np.mean(distances)) if distances else 0.0,
        max_distance=float(np.max(distances)) if distances else 0.0,
        std_a=float(lab_values[:, 1].std(ddof=0)),
        std_b=float(lab_values[:, 2].std(ddof=0)),
        references=selected,
    )


def get_cluster_models(
    reference_samples: Sequence[ReferenceSample],
) -> Dict[str, LABClusterModel]:
    return {
        cluster_key: compute_cluster_model(cluster_key, reference_samples)
        for cluster_key in HAZARD_CLASS_PROFILES
    }


def get_reference_swatches(reference_samples: Sequence[ReferenceSample]) -> Dict[str, LABColor]:
    return {key: model.centroid for key, model in get_cluster_models(reference_samples).items()}


def format_anchor_exemplar_terms(cluster_key: str) -> str:
    return ", ".join(LAB_CLASS_EXEMPLARS[cluster_key])


def extract_marc_color_terms(record: Any) -> Tuple[str, ...]:
    lookup = load_iscc_nbs_lookup()
    known_terms = sorted(
        (
            (str(row["Color Name"]), str(row["normalized_name"]))
            for _, row in lookup.iterrows()
            if str(row["normalized_name"]).strip()
        ),
        key=lambda item: len(item[1]),
        reverse=True,
    )
    found_terms: List[str] = []

    # Prefer structured genre/form faceting in 655$b when present.
    for field in record.get_fields("655"):
        for value in field.get_subfields("b"):
            normalized_value = normalize_color_name(value)
            if not normalized_value:
                continue
            matched_known_term = False
            for label, normalized_term in known_terms:
                if normalized_value == normalized_term or re.search(
                    rf"\b{re.escape(normalized_term)}\b", normalized_value
                ):
                    if label not in found_terms:
                        found_terms.append(label)
                    matched_known_term = True
            if not matched_known_term and infer_color_family_from_name(normalized_value) != "Unknown":
                pretty_value = prettify_color_name(normalized_value)
                if pretty_value not in found_terms:
                    found_terms.append(pretty_value)

    for tag in MARC_COLOR_NOTE_FIELDS:
        for field in record.get_fields(tag):
            field_text = normalize_color_name(field.format_field())
            if not field_text:
                continue
            if tag in {"500", "590"} and not any(word in field_text for word in BINDING_CONTEXT_WORDS):
                continue
            for label, normalized_term in known_terms:
                if re.search(rf"\b{re.escape(normalized_term)}\b", field_text):
                    if label not in found_terms:
                        found_terms.append(label)

    return tuple(found_terms)


def make_observed_lab(
    name: str,
    swatch_hex: str,
    manual_L: Optional[float] = None,
    manual_a: Optional[float] = None,
    manual_b: Optional[float] = None,
    named_color: Optional[LABColor] = None,
) -> LABColor:
    if manual_L is not None and manual_a is not None and manual_b is not None:
        return LABColor(name=name, L=float(manual_L), a=float(manual_a), b=float(manual_b))
    if named_color is not None:
        return LABColor(name=name, L=named_color.L, a=named_color.a, b=named_color.b)
    return LABColor.from_hex(name=name, hex_color=swatch_hex)


def delta_e_cie76(lab1: LABColor, lab2: LABColor) -> float:
    return float(
        np.sqrt(
            (lab1.L - lab2.L) ** 2
            + (lab1.a - lab2.a) ** 2
            + (lab1.b - lab2.b) ** 2
        )
    )


def delta_e_ciede2000(lab1: LABColor, lab2: LABColor) -> float:
    lab1_array = np.array([[[lab1.L, lab1.a, lab1.b]]], dtype=float)
    lab2_array = np.array([[[lab2.L, lab2.a, lab2.b]]], dtype=float)
    return float(color.deltaE_ciede2000(lab1_array, lab2_array)[0, 0])


def lab_chroma(lab: LABColor) -> float:
    return float(np.hypot(lab.a, lab.b))


def lab_hue_angle(lab: LABColor) -> float:
    angle = float(np.degrees(np.arctan2(lab.b, lab.a)))
    return (angle + 360.0) % 360.0


def hue_angle_difference(lab1: LABColor, lab2: LABColor) -> float:
    diff = abs(lab_hue_angle(lab1) - lab_hue_angle(lab2))
    return float(min(diff, 360.0 - diff))


def format_delta_e00_value(
    effective_delta_e00: float, raw_delta_e00: float, fading_adjustment: float
) -> str:
    if fading_adjustment > 0:
        return (
            f"{effective_delta_e00:.2f} "
            f"(raw {raw_delta_e00:.2f}; fading allowance {fading_adjustment:.2f})"
        )
    return f"{effective_delta_e00:.2f}"


def compute_fading_adjustment(
    observed_lab: LABColor,
    reference_sample: ReferenceSample,
    cluster_key: str,
    inputs: Optional[ScreeningInput],
) -> Tuple[float, str]:
    if inputs is None or inputs.fading_evidence == "No obvious fading":
        return 0.0, ""

    reference_chroma = lab_chroma(reference_sample.color)
    observed_chroma = lab_chroma(observed_lab)
    chroma_drop = reference_chroma - observed_chroma
    if chroma_drop <= 4.0:
        return 0.0, ""

    hue_gap = hue_angle_difference(observed_lab, reference_sample.color)
    max_hue_gap = 28.0 if cluster_key in {"chrome_green", "mercury_red"} else 24.0
    if hue_gap > max_hue_gap:
        return 0.0, ""

    allowance = 0.0
    reasons: List[str] = []

    if inputs.fading_evidence == "Possible fading/browning":
        allowance += 0.9
        reasons.append("visible fading or browning was flagged")
    if inputs.spine_browned is True:
        allowance += 0.6
        reasons.append("spine or edge browning is present")
    if chroma_drop >= 8.0:
        allowance += 0.4
    if chroma_drop >= 14.0:
        allowance += 0.5
        reasons.append(f"observed chroma is {chroma_drop:.1f} below the reference")
    if abs(observed_lab.L - reference_sample.color.L) >= 6.0:
        allowance += 0.2

    max_allowance = 1.0 if reference_sample.is_local else 2.0
    allowance = min(max_allowance, allowance)
    if allowance < 0.5:
        return 0.0, ""

    basis = "same-device local reference" if reference_sample.is_local else "built-in fallback anchor"
    reason_text = "; ".join(reasons) if reasons else "hue stayed near the class while chroma dropped"
    note = (
        f"Applied a capped fading allowance of ΔE00 {allowance:.2f} against the {basis} because "
        f"hue stayed near the reference ({hue_gap:.1f}°) while chroma dropped by {chroma_drop:.1f}; "
        f"{reason_text}."
    )
    return allowance, note


def find_best_reference_match(
    observed_lab: LABColor,
    cluster_key: str,
    reference_samples: Sequence[ReferenceSample],
    inputs: Optional[ScreeningInput],
) -> Tuple[LABClusterModel, ReferenceMatch, bool]:
    model = get_cluster_models(reference_samples)[cluster_key]
    has_local_reference = any(sample.is_local for sample in model.references)
    best_match: Optional[ReferenceMatch] = None

    for sample in model.references:
        raw_delta_e00 = delta_e_ciede2000(observed_lab, sample.color)
        delta_e76 = delta_e_cie76(observed_lab, sample.color)
        fading_adjustment, fading_note = compute_fading_adjustment(
            observed_lab,
            sample,
            cluster_key,
            inputs,
        )
        match = ReferenceMatch(
            sample=sample,
            effective_delta_e00=max(0.0, raw_delta_e00 - fading_adjustment),
            raw_delta_e00=raw_delta_e00,
            delta_e76=delta_e76,
            fading_adjustment=fading_adjustment,
            fading_note=fading_note,
        )
        if best_match is None or (
            match.effective_delta_e00,
            match.raw_delta_e00,
            match.delta_e76,
            match.sample.sample_id,
        ) < (
            best_match.effective_delta_e00,
            best_match.raw_delta_e00,
            best_match.delta_e76,
            best_match.sample.sample_id,
        ):
            best_match = match

    if best_match is None:
        raise ValueError(f"No reference samples available for cluster {cluster_key}.")
    return model, best_match, has_local_reference


def evaluate_lab_reference_distance(
    observed_lab: LABColor,
    cluster_key: str,
    reference_samples: Sequence[ReferenceSample],
    measurement_summary: Optional[MeasurementSummary] = None,
    inputs: Optional[ScreeningInput] = None,
) -> Tuple[LABClusterModel, ReferenceSample, float, float, float, float, int, str, str, bool]:
    model, reference_match, has_local_reference = find_best_reference_match(
        observed_lab,
        cluster_key,
        reference_samples,
        inputs,
    )
    nearest_reference = reference_match.sample
    delta_e00 = reference_match.effective_delta_e00
    raw_delta_e00 = reference_match.raw_delta_e00
    fading_adjustment = reference_match.fading_adjustment
    delta_e76 = reference_match.delta_e76
    band_index = next(
        index
        for index, (threshold, _, _, _) in enumerate(DELTA_E00_WORKING_BANDS)
        if delta_e00 <= threshold
    )
    initial_band_index = band_index
    adjustments: List[str] = []

    if measurement_summary is not None:
        while (
            band_index < len(DELTA_E00_WORKING_BANDS) - 1
            and measurement_summary.max_delta_e00 >= DELTA_E00_WORKING_BANDS[band_index][0]
        ):
            band_index += 1
        if band_index != initial_band_index:
            adjustments.append(
                f"Repeated measurements varied by up to ΔE00 {measurement_summary.max_delta_e00:.2f}, "
                "so the app downgraded the colour band until it exceeded the observed spread."
            )

    if not has_local_reference and band_index == 0:
        band_index = 1
        adjustments.append(
            "This class currently relies on built-in fallback anchors rather than uploaded same-device "
            "local references, so the app will not issue a top-tier local-reference match."
        )

    _, points, fit_label, relation = DELTA_E00_WORKING_BANDS[band_index]
    if not has_local_reference:
        anchor_phrase = built_in_fallback_anchor_phrase(cluster_key)
        if band_index == 1:
            fit_label = LAB_FIT_PROVISIONAL_CLOSE
            relation = f"Observed LAB is near the {anchor_phrase} for this class."
        elif band_index == 2:
            fit_label = LAB_FIT_PROVISIONAL_LOOSE
            relation = f"Observed LAB is only loosely similar to the {anchor_phrase} for this class."
        elif band_index == 3:
            relation = (
                "Observed LAB falls outside the app's current provisional bands for the built-in "
                "fallback anchors for this class."
            )

    evidence_parts = [
        relation,
        (
            f"Nearest reference: {nearest_reference.color.name} "
            f"(ΔE00 {format_delta_e00_value(delta_e00, raw_delta_e00, fading_adjustment)}; "
            f"ΔE76 {delta_e76:.2f}; source: {nearest_reference.source})."
        ),
    ]
    if reference_match.fading_note:
        evidence_parts.append(reference_match.fading_note)
    if not has_local_reference:
        evidence_parts.append(built_in_fallback_caveat(cluster_key))
    evidence_parts.extend(adjustments)
    evidence = " ".join(evidence_parts)
    return (
        model,
        nearest_reference,
        delta_e00,
        raw_delta_e00,
        fading_adjustment,
        delta_e76,
        points,
        evidence,
        fit_label,
        has_local_reference,
    )


def nearest_lab_class_note(
    inputs: ScreeningInput, reference_samples: Sequence[ReferenceSample]
) -> str:
    inferred_name, inferred_distance = infer_named_color_from_lab(inputs.observed_lab)
    cluster_models = get_cluster_models(reference_samples)
    ranked = sorted(
        (
            (
                profile.label,
                find_best_reference_match(
                    inputs.observed_lab,
                    profile.cluster_key,
                    reference_samples,
                    inputs,
                )[1],
                cluster_models[profile.cluster_key],
            )
            for profile in HAZARD_CLASS_PROFILES.values()
        ),
        key=lambda item: item[1].effective_delta_e00,
    )
    label, match, model = ranked[0]
    has_local_reference = any(sample.is_local for sample in model.references)
    basis = reference_basis_label(has_local_reference).lower()
    caveat = (
        "This is a colour-similarity statement only; pigment identification still requires instrumental confirmation."
        if has_local_reference
        else (
            "This is a provisional similarity statement because this class is still using built-in fallback anchors "
            "rather than uploaded same-device references."
        )
    )
    return (
        f"Nearest ISCC-NBS colour name: {inferred_name.name} (ΔE00 {inferred_distance:.2f}). "
        f"Nearest LAB hazard class: {label.lower()} "
        f"(nearest-reference ΔE00 {format_delta_e00_value(match.effective_delta_e00, match.raw_delta_e00, match.fading_adjustment)}; "
        f"class reference count {len(model.references)}; basis: {basis}). "
        f"{caveat}"
    )


def nearest_lab_candidate(
    candidates: Sequence[CandidateAssessment],
) -> CandidateAssessment:
    return min(
        candidates,
        key=lambda candidate: (
            candidate.delta_e00,
            candidate.delta_e76,
            -candidate.score,
            candidate.profile.key,
        ),
    )


def score_emerald_green(
    inputs: ScreeningInput, reference_samples: Sequence[ReferenceSample]
) -> CandidateAssessment:
    score = 0
    evidence: List[str] = []
    xrf = set(inputs.xrf_elements)
    elemental_supported = False
    (
        model,
        nearest_reference,
        delta_e00,
        raw_delta_e00,
        fading_adjustment,
        delta_e76,
        cluster_points,
        cluster_evidence,
        lab_fit_label,
        has_local_reference,
    ) = evaluate_lab_reference_distance(
        inputs.observed_lab,
        "emerald_green",
        reference_samples,
        inputs.measurement_summary,
        inputs,
    )
    score += cluster_points
    evidence.append(cluster_evidence)

    if "cloth-case" in inputs.binding_type.lower():
        score += 15
        evidence.append("Binding type is compatible with publishers' cloth-case production.")
    elif "uncertain" in inputs.binding_type.lower():
        score += 6

    if inputs.color_family == "Green":
        score += 22
        evidence.append("Observed family is green.")

    if inputs.vividness == "Vivid/bright":
        score += 10
        evidence.append("Vivid green is consistent with classic emerald-green screening descriptions.")
    elif inputs.vividness == "Moderate":
        score += 5

    if inputs.spine_browned is True:
        score += 15
        evidence.append("Browned spine or edges support the published darkening behaviour of emerald green.")

    if inputs.stamped_decoration is True:
        score += 6
        evidence.append("Stamped or gilt decoration is common on surviving publishers' cloth examples.")

    if inputs.publication_year is not None:
        if 1840 <= inputs.publication_year <= 1869:
            score += 18
            evidence.append("Date falls in the strongest visual screening window for emerald-green cloth.")
        elif 1837 <= inputs.publication_year <= 1900:
            score += 8
            evidence.append("Date remains plausible for 19th-century arsenical cloth.")

    if inputs.region in {"United Kingdom / Ireland", "United States / Canada"}:
        score += 8
        evidence.append("Published literature on cloth-case examples is strongest for British and North American books.")

    if {"As", "Cu"}.issubset(xrf):
        elemental_supported = True
        score = max(score + 45, 92)
        evidence.append(
            "XRF shows As + Cu, an elemental pattern consistent with an arsenical copper green. "
            "Compound-level identification still requires Raman, FTIR, or XRD."
        )
    elif "As" in xrf:
        score += 20
        evidence.append("Arsenic is present, which raises concern but is not yet chemically specific.")
    elif {"Pb", "Cr"}.issubset(xrf):
        score -= 20
        evidence.append("Pb + Cr shifts the interpretation toward chrome yellow/chrome green instead.")
    elif "Cu" in xrf:
        score -= 6
        evidence.append("Copper without arsenic is less supportive of emerald green.")

    if not elemental_supported and "As" not in xrf:
        score = min(score, 70)

    score = max(0, min(score, 99))
    confidence = score_to_confidence(score, elemental_supported, lab_fit_label)
    if elemental_supported:
        summary = "LAB class and XRF together support an arsenic-associated green screening case."
    elif lab_fit_label == LAB_FIT_STRONG_LOCAL:
        summary = (
            "LAB sits close to uploaded local references for the emerald-green-like class, but colour alone "
            "still cannot establish chemistry."
        )
    elif lab_fit_label in PROVISIONAL_LAB_FIT_LABELS:
        summary = (
            "LAB is near the emerald-green-like fallback anchors, so keep this as a provisional colour "
            "screen until local references or XRF are available."
        )
    elif delta_e00 <= 8.0:
        summary = (
            "LAB shows loose similarity to the emerald-green-like class, but colour alone cannot establish chemistry."
        )
    else:
        summary = (
            "Contextual factors keep the emerald-green-like class in view, but the observed LAB is not close to this class."
        )
    next_step = (
        "Isolate the book, use nitrile gloves, and seek Raman, FTIR, or XRD confirmation if compound-level identification matters."
        if elemental_supported
        else (
            "Treat this as a provisional fallback colour screen. Upload a same-device local reference library "
            "for this class and prioritise qualitative XRF before treatment or broad access."
            if not has_local_reference
            else "Treat this as a colorimetric alert and prioritise qualitative XRF before treatment or broad access."
        )
    )
    return CandidateAssessment(
        profile=HAZARD_CLASS_PROFILES["emerald_green"],
        score=score,
        confidence=confidence,
        lab_fit_label=lab_fit_label,
        evidence=tuple(evidence),
        summary=summary,
        next_step=next_step,
        elemental_supported=elemental_supported,
        delta_e00=delta_e00,
        raw_delta_e00=raw_delta_e00,
        fading_adjustment=fading_adjustment,
        delta_e76=delta_e76,
        nearest_reference_name=nearest_reference.color.name,
        reference_source=nearest_reference.source,
        reference_count=len(model.references),
        has_local_reference=has_local_reference,
    )


def score_chrome_green(
    inputs: ScreeningInput, reference_samples: Sequence[ReferenceSample]
) -> CandidateAssessment:
    score = 0
    evidence: List[str] = []
    xrf = set(inputs.xrf_elements)
    elemental_supported = False
    (
        model,
        nearest_reference,
        delta_e00,
        raw_delta_e00,
        fading_adjustment,
        delta_e76,
        cluster_points,
        cluster_evidence,
        lab_fit_label,
        has_local_reference,
    ) = evaluate_lab_reference_distance(
        inputs.observed_lab,
        "chrome_green",
        reference_samples,
        inputs.measurement_summary,
        inputs,
    )
    score += cluster_points
    evidence.append(cluster_evidence)

    if "cloth-case" in inputs.binding_type.lower():
        score += 15
        evidence.append("Binding type fits 19th-century cloth-case production.")
    elif "uncertain" in inputs.binding_type.lower():
        score += 5

    if inputs.color_family == "Green":
        score += 24
        evidence.append("Observed family is green.")

    if inputs.vividness == "Muted/olive":
        score += 10
        evidence.append("Muted or olive greens are often more consistent with chrome-green mixtures than emerald green.")
    elif inputs.vividness == "Moderate":
        score += 7
    elif inputs.vividness == "Vivid/bright":
        score += 3

    if inputs.spine_browned is False and inputs.color_family == "Green":
        score += 5
        evidence.append("Absence of brown darkening is slightly more consistent with non-arsenical green systems.")

    if inputs.publication_year is not None and 1837 <= inputs.publication_year <= 1900:
        score += 12
        evidence.append("Date is compatible with 19th-century chrome-green use.")

    if {"Pb", "Cr"}.issubset(xrf):
        elemental_supported = True
        score = max(score + 45, 88)
        evidence.append(
            "XRF shows Pb + Cr on green cloth, an elemental pattern consistent with a lead/chromium green "
            "system. Compound-level identification still requires Raman, FTIR, or XRD."
        )
        if "Fe" in xrf:
            score += 4
            evidence.append("Fe can be supportive when Prussian blue is part of the green mixture.")
    elif {"As", "Cu"}.issubset(xrf):
        score -= 20
        evidence.append("As + Cu points more strongly to emerald green.")

    score = max(0, min(score, 99))
    confidence = score_to_confidence(score, elemental_supported, lab_fit_label)
    if elemental_supported:
        summary = "LAB class and XRF together support a lead/chromium-associated green screening case."
    elif lab_fit_label in PROVISIONAL_LAB_FIT_LABELS:
        summary = (
            "LAB is near the chrome-green-like fallback anchors, so keep this as a provisional colour "
            "screen until local references or XRF are available."
        )
    elif delta_e00 <= 5.0:
        summary = (
            "LAB places the binding near the chrome-green-like class, but Pb/Cr chemistry remains unconfirmed."
        )
    elif delta_e00 <= 8.0:
        summary = (
            "LAB shows loose similarity to the chrome-green-like class, but Pb/Cr chemistry remains unconfirmed."
        )
    else:
        summary = (
            "Contextual factors keep the chrome-green-like class in view, but the observed LAB is not close to this class."
        )
    next_step = (
        "Confirm the green component with Raman, FTIR, or XRD if you need to distinguish the mixture from other Pb/Cr systems."
        if elemental_supported
        else (
            "Use qualitative XRF first and build a same-device local reference library before relying heavily on the colour distance for this class."
            if not has_local_reference
            else "Use qualitative XRF first; Pb + Cr is the decisive elemental pattern for this class."
        )
    )
    return CandidateAssessment(
        profile=HAZARD_CLASS_PROFILES["chrome_green"],
        score=score,
        confidence=confidence,
        lab_fit_label=lab_fit_label,
        evidence=tuple(evidence),
        summary=summary,
        next_step=next_step,
        elemental_supported=elemental_supported,
        delta_e00=delta_e00,
        raw_delta_e00=raw_delta_e00,
        fading_adjustment=fading_adjustment,
        delta_e76=delta_e76,
        nearest_reference_name=nearest_reference.color.name,
        reference_source=nearest_reference.source,
        reference_count=len(model.references),
        has_local_reference=has_local_reference,
    )


def score_chrome_yellow(
    inputs: ScreeningInput, reference_samples: Sequence[ReferenceSample]
) -> CandidateAssessment:
    score = 0
    evidence: List[str] = []
    xrf = set(inputs.xrf_elements)
    elemental_supported = False
    (
        model,
        nearest_reference,
        delta_e00,
        raw_delta_e00,
        fading_adjustment,
        delta_e76,
        cluster_points,
        cluster_evidence,
        lab_fit_label,
        has_local_reference,
    ) = evaluate_lab_reference_distance(
        inputs.observed_lab,
        "chrome_yellow",
        reference_samples,
        inputs.measurement_summary,
        inputs,
    )
    score += cluster_points
    evidence.append(cluster_evidence)

    if "cloth-case" in inputs.binding_type.lower():
        score += 15
        evidence.append("Binding type is compatible with original cloth-case production.")
    elif "uncertain" in inputs.binding_type.lower():
        score += 5

    if inputs.color_family in {"Yellow", "Orange", "Brown"}:
        score += 26
        evidence.append("Observed family is yellow, orange, or yellow-brown.")

    if inputs.vividness == "Vivid/bright":
        score += 8
    elif inputs.vividness == "Moderate":
        score += 6
    elif inputs.vividness == "Muted/olive":
        score += 4

    if inputs.publication_year is not None:
        if 1880 <= inputs.publication_year <= 1899:
            score += 18
            evidence.append("Late-century date supports the strong chrome-yellow signal reported in bookcloth studies.")
        elif 1837 <= inputs.publication_year <= 1900:
            score += 8

    if {"Pb", "Cr"}.issubset(xrf):
        elemental_supported = True
        score = max(score + 45, 88)
        evidence.append(
            "XRF shows Pb + Cr, an elemental pattern consistent with a lead-chromate family pigment. "
            "Compound-level identification still requires Raman, FTIR, or XRD."
        )
    elif {"As", "Cu"}.issubset(xrf):
        score -= 18
        evidence.append("As + Cu points away from chrome yellow and toward arsenical copper green.")

    score = max(0, min(score, 99))
    confidence = score_to_confidence(score, elemental_supported, lab_fit_label)
    if elemental_supported:
        summary = "LAB class and XRF together support a lead/chromium-associated yellow-orange screening case."
    elif lab_fit_label in PROVISIONAL_LAB_FIT_LABELS:
        summary = (
            "LAB is near the chrome-yellow-like fallback anchors, so keep this as a provisional colour "
            "screen until local references or XRF are available."
        )
    elif delta_e00 <= 5.0:
        summary = (
            "LAB places the binding near the chrome-yellow-like class, but Pb/Cr chemistry remains unconfirmed."
        )
    elif delta_e00 <= 8.0:
        summary = (
            "LAB shows loose similarity to the chrome-yellow-like class, but Pb/Cr chemistry remains unconfirmed."
        )
    else:
        summary = (
            "Contextual factors keep the chrome-yellow-like class in view, but the observed LAB is not close to this class."
        )
    next_step = (
        "Handle as a lead/chromium binding and confirm with Raman, FTIR, or XRD if the distinction between yellow and orange variants matters."
        if elemental_supported
        else (
            "Use qualitative XRF first and build a same-device local reference library before relying heavily on the colour distance for this class."
            if not has_local_reference
            else "Use qualitative XRF; Pb + Cr should determine whether this LAB alert corresponds to a lead-chromate system."
        )
    )
    return CandidateAssessment(
        profile=HAZARD_CLASS_PROFILES["chrome_yellow"],
        score=score,
        confidence=confidence,
        lab_fit_label=lab_fit_label,
        evidence=tuple(evidence),
        summary=summary,
        next_step=next_step,
        elemental_supported=elemental_supported,
        delta_e00=delta_e00,
        raw_delta_e00=raw_delta_e00,
        fading_adjustment=fading_adjustment,
        delta_e76=delta_e76,
        nearest_reference_name=nearest_reference.color.name,
        reference_source=nearest_reference.source,
        reference_count=len(model.references),
        has_local_reference=has_local_reference,
    )


def score_mercury_red(
    inputs: ScreeningInput, reference_samples: Sequence[ReferenceSample]
) -> CandidateAssessment:
    score = 0
    evidence: List[str] = []
    xrf = set(inputs.xrf_elements)
    elemental_supported = False
    (
        model,
        nearest_reference,
        delta_e00,
        raw_delta_e00,
        fading_adjustment,
        delta_e76,
        cluster_points,
        cluster_evidence,
        lab_fit_label,
        has_local_reference,
    ) = evaluate_lab_reference_distance(
        inputs.observed_lab,
        "mercury_red",
        reference_samples,
        inputs.measurement_summary,
        inputs,
    )
    score += cluster_points
    evidence.append(cluster_evidence)

    if inputs.color_family == "Red":
        score += 24
        evidence.append("Observed family is red.")

    if "cloth" in inputs.binding_type.lower():
        score += 8

    if inputs.publication_year is not None and 1850 <= inputs.publication_year <= 1900:
        score += 14
        evidence.append("Victorian-era date aligns with published mercury findings in historical books.")

    if "Hg" in xrf:
        elemental_supported = True
        score = max(score + 45, 80)
        evidence.append(
            "XRF shows Hg, which creates a meaningful watchlist signal for red historical components, "
            "but does not by itself establish the exact mercury pigment."
        )
    elif {"Pb", "Cr"}.issubset(xrf):
        score -= 12
    elif {"As", "Cu"}.issubset(xrf):
        score -= 16

    if not elemental_supported:
        score = min(score, 55)

    score = max(0, min(score, 99))
    confidence = score_to_confidence(score, elemental_supported, lab_fit_label)
    if elemental_supported:
        summary = "LAB class and XRF together support escalation of a mercury watchlist case."
    elif lab_fit_label in PROVISIONAL_LAB_FIT_LABELS:
        summary = (
            "LAB is near the mercury-red-like fallback anchors, so keep this as a provisional watchlist "
            "screen until local references or XRF are available."
        )
    elif delta_e00 <= 8.0:
        summary = "Keep this as a conservative LAB watchlist class rather than a chemistry claim."
    else:
        summary = "This remains a conservative watchlist class only; the observed LAB is not close to the reference reds."
    next_step = (
        "Escalate to specialist review because Hg in cloth bindings is less mature as a screening literature than emerald green."
        if elemental_supported
        else (
            "Do not infer mercury from LAB alone. Build a same-device local reference library and test before applying any handling escalation beyond standard precautions."
            if not has_local_reference
            else "Do not infer mercury from LAB alone; test before applying any handling escalation beyond standard precautions."
        )
    )
    return CandidateAssessment(
        profile=HAZARD_CLASS_PROFILES["mercury_red"],
        score=score,
        confidence=confidence,
        lab_fit_label=lab_fit_label,
        evidence=tuple(evidence),
        summary=summary,
        next_step=next_step,
        elemental_supported=elemental_supported,
        delta_e00=delta_e00,
        raw_delta_e00=raw_delta_e00,
        fading_adjustment=fading_adjustment,
        delta_e76=delta_e76,
        nearest_reference_name=nearest_reference.color.name,
        reference_source=nearest_reference.source,
        reference_count=len(model.references),
        has_local_reference=has_local_reference,
    )


def score_to_confidence(
    score: int,
    elemental_supported: bool,
    lab_fit_label: str,
) -> str:
    if elemental_supported:
        return "Supported by XRF"
    if lab_fit_label == LAB_FIT_STRONG_LOCAL:
        return "Strong local colour match"
    if lab_fit_label in PROVISIONAL_LAB_FIT_LABELS:
        return "Provisional colour match"
    if lab_fit_label == LAB_FIT_CLOSE:
        return "Colour match"
    if score >= 45:
        return "Possible from context"
    return "Not supported by colour"


def interpret_xrf(elements: Sequence[str], color_family: str) -> str:
    found = set(elements)
    if not found:
        return (
            "No XRF elements entered. This app therefore limits itself to contextual screening "
            "and deliberately caps confidence."
        )

    messages: List[str] = []
    if {"As", "Cu"}.issubset(found):
        messages.append(
            "As + Cu on green cloth is an elemental pattern consistent with an arsenical copper green, "
            "but XRF alone does not establish the exact compound."
        )
    elif "As" in found:
        messages.append(
            "Arsenic is present, but without Cu the result may reflect degradation, migration, or a different arsenical system."
        )

    if {"Pb", "Cr"}.issubset(found):
        if color_family == "Green":
            messages.append(
                "Pb + Cr on a green binding is an elemental pattern consistent with a lead-chromate and blue mixture often described as chrome green."
            )
        else:
            messages.append(
                "Pb + Cr on a yellow/orange/brown binding is an elemental pattern consistent with chrome yellow or a related lead-chromate variant."
            )

    if "Hg" in found:
        messages.append(
            "Hg is a watchlist signal for mercury-bearing red components; treat it as requiring specialist interpretation because XRF alone does not identify the exact mercury pigment."
        )

    if "Cu" in found and "As" not in found:
        messages.append("Cu without arsenic is not, by itself, sufficient support for emerald green.")
    if "Fe" in found and not ({"As", "Cu"} & found):
        messages.append("Fe may relate to Prussian blue or iron-oxide colourants but is not a stand-alone heavy-metal hazard signal here.")
    if "Zn" in found and not (HAZARD_ELEMENTS & found):
        messages.append("Zn alone is not one of the principal hazardous pigment flags used in this tool.")

    return " ".join(messages) if messages else "The entered XRF pattern is non-specific; specialist interpretation is recommended."


def determine_priority(inputs: ScreeningInput, candidates: Sequence[CandidateAssessment]) -> Tuple[str, str]:
    xrf = set(inputs.xrf_elements)
    green_cloth = (
        inputs.color_family == "Green"
        and "cloth" in inputs.binding_type.lower()
        and inputs.publication_year is not None
        and 1800 <= inputs.publication_year <= 1900
    )

    if {"As", "Cu"}.issubset(xrf):
        return (
            "Urgent isolate + confirm",
            "Use nitrile gloves, limit handling, house the volume separately, and confirm the arsenical copper-green assignment with Raman, FTIR, or XRD if available.",
        )

    if any(
        candidate.profile.key == "emerald_green"
        and candidate.lab_fit_label == LAB_FIT_STRONG_LOCAL
        for candidate in candidates
    ):
        return (
            "High priority for confirmatory testing",
            "LAB sits inside the app's tighter provisional emerald-green band against uploaded local references. Treat this as a strong colour alert and obtain XRF before making a chemistry claim.",
        )

    if {"Pb", "Cr"}.issubset(xrf):
        return (
            "High priority for restricted handling",
            "Lead/chromium pigments are strongly indicated. Keep handling controlled and confirm the exact Pb/Cr system only if the distinction matters for treatment or access.",
        )

    if green_cloth:
        return (
            "High priority for instrumental screening",
            "Current research shows multiple hazardous green systems in 19th-century bindings. Even when colour falls outside the tight band, green cloth should still be triaged early.",
        )

    if (
        inputs.color_family in {"Yellow", "Orange", "Brown", "Red"}
        and inputs.publication_year is not None
        and 1850 <= inputs.publication_year <= 1900
    ):
        return (
            "Moderate priority",
            "Later-century yellow, orange, brown, and red cloth can still justify testing, but the evidence base is less specific than for green cloth.",
        )

    return (
        "Lower immediate priority",
        "Inputs do not strongly support one of the major heavy-metal cloth-binding systems tracked here, though absence of evidence is not proof of safety.",
    )


def determine_handling(inputs: ScreeningInput, candidates: Sequence[CandidateAssessment]) -> Tuple[str, str]:
    xrf = set(inputs.xrf_elements)
    if {"As", "Cu"}.issubset(xrf):
        return (
            "Arsenical handling protocol",
            (
                "Wear nitrile gloves, avoid face contact, bag the book if storage isolation is needed, "
                "and do not use consumer arsenic spot tests. Project guidance warns that emerald green on "
                "bookcloth can be friable and can transfer to surfaces."
            ),
        )

    if any(
        candidate.profile.key == "emerald_green"
        and candidate.lab_fit_label == LAB_FIT_STRONG_LOCAL
        for candidate in candidates
    ):
        return (
            "Precautionary green-screen protocol",
            (
                "Because the binding sits inside the tighter emerald-green ΔE00 band against uploaded local references, "
                "minimise unnecessary handling, avoid dry abrasion, and move to XRF screening before treatment or broad access decisions."
            ),
        )

    if {"Pb", "Cr"}.issubset(xrf):
        return (
            "Lead/chromium handling protocol",
            (
                "Avoid food and face contact while handling, wash hands after use, and minimise abrasive "
                "manipulation. Poison Book Project guidance currently indicates lower offset risk than with emerald green."
            ),
        )

    if "Hg" in xrf:
        return (
            "Mercury watchlist protocol",
            (
                "Keep handling controlled and seek institutional advice for mercury-bearing material, especially "
                "before treatment, sampling, or housing work."
            ),
        )

    return (
        "Standard 19th-century handling protocol",
        (
            "Use normal rare-material precautions: clean hands, no food or drink, and no dry abrasion. "
            "Escalate only after more specific evidence appears."
        ),
    )


def build_sop(inputs: ScreeningInput, outcome: ScreeningOutcome) -> ScenarioSOP:
    xrf = set(inputs.xrf_elements)
    top = outcome.candidates[0]
    nearest = nearest_lab_candidate(outcome.candidates)
    green_cloth = inputs.color_family == "Green" and "cloth" in inputs.binding_type.lower()
    uses_fallback = not top.has_local_reference
    sheltered_step = (
        "Measure sheltered or less-exposed areas if available, then rerun the screen to see whether a less-faded colour point moves closer to a hazardous class."
    )
    local_reference_step = (
        "Build or upload a same-device local reference library for the most plausible class before making repeated colour-based decisions."
    )

    if {"As", "Cu"}.issubset(xrf):
        return ScenarioSOP(
            key="arsenical_copper_green",
            title="SOP: Arsenical Copper Green Screen",
            summary=(
                "As + Cu is present on a green binding. Treat the case as an arsenical copper-green screen that requires controlled handling and, if needed, confirmatory analysis."
            ),
            steps=(
                "Limit handling, isolate the volume from routine circulation, and use nitrile gloves.",
                "Avoid dry cleaning, surface rubbing, or other abrasive manipulation until the handling plan is set.",
                "Record the XRF pattern, colour evidence, and any visible friability in the object file or survey sheet.",
                "Seek Raman, FTIR, or XRD only if compound-level confirmation matters for treatment, exhibition, or access decisions.",
                sheltered_step,
                local_reference_step,
            ),
        )

    if {"Pb", "Cr"}.issubset(xrf) and (inputs.color_family == "Green" or top.profile.key == "chrome_green"):
        return ScenarioSOP(
            key="lead_chromium_green",
            title="SOP: Lead/Chromium Green Screen",
            summary=(
                "Pb + Cr is present on a green binding. Treat this as a lead/chromium green-system case pending any mixture-specific confirmation."
            ),
            steps=(
                "Use controlled handling and avoid abrasive treatment steps until the Pb/Cr risk is logged.",
                "If Fe is also present, note that it can support a chrome-green-plus-blue mixture interpretation, but do not treat that as compound proof.",
                "Record the result as a lead/chromium green screen rather than a specific pigment identification.",
                "Escalate to Raman, FTIR, or XRD only if distinguishing the exact green system will change treatment or access decisions.",
                sheltered_step,
                local_reference_step,
            ),
        )

    if {"Pb", "Cr"}.issubset(xrf):
        return ScenarioSOP(
            key="lead_chromium_yellow",
            title="SOP: Lead/Chromium Yellow-Orange Screen",
            summary=(
                "Pb + Cr is present on a yellow, orange, or brown binding. Treat this as a lead-chromate-family screen pending any variant-specific confirmation."
            ),
            steps=(
                "Use controlled handling, minimise abrasive contact, and keep food and face contact away from the work area.",
                "Record the case as a lead/chromium yellow-orange screen rather than a confirmed chrome-yellow variant.",
                "Use further spectroscopy only if the exact pigment family matters for treatment, access, or sampling decisions.",
                sheltered_step,
                local_reference_step,
            ),
        )

    if "Hg" in xrf:
        return ScenarioSOP(
            key="mercury_watchlist",
            title="SOP: Mercury Watchlist Case",
            summary=(
                "Hg is present on a red component. Treat this as a mercury watchlist case that requires controlled handling and specialist interpretation."
            ),
            steps=(
                "Keep handling controlled and route the item through institutional health-and-safety guidance before treatment or sampling.",
                "Record the result as an Hg watchlist signal rather than a vermilion or cinnabar identification.",
                "Use confirmatory analysis only if the mercury-bearing component will affect treatment, storage, or access decisions.",
                sheltered_step,
                local_reference_step,
            ),
        )

    if green_cloth and top.lab_fit_label == LAB_FIT_STRONG_LOCAL:
        return ScenarioSOP(
            key="strong_local_green",
            title="SOP: Strong Local Green Match Pending XRF",
            summary=(
                "Colour sits close to uploaded local references for a hazardous green class, but no decisive XRF pattern is entered yet. Treat this as a high-priority colour alert, not a pigment identification."
            ),
            steps=(
                "Keep handling controlled and avoid unnecessary dry abrasion while the screen is unresolved.",
                "Prioritise qualitative XRF as the next test before any treatment or broad access decision.",
                "Record the case as a strong local colour match pending elemental confirmation.",
                sheltered_step,
                "Review the local reference record to confirm that the analogue cloth, instrument, and ageing state are comparable.",
            ),
        )

    if green_cloth and not xrf and top.lab_fit_label == LAB_FIT_NONE and nearest.delta_e00 > 8.0:
        summary = (
            f"Historic green-cloth context keeps this book in scope, but the nearest hazard class is still far by colour "
            f"(nearest-reference ΔE00 {nearest.delta_e00:.2f}) and no XRF pattern is entered."
        )
        if uses_fallback:
            summary += " The result is therefore ambiguous and still running on built-in fallback anchors."
        return ScenarioSOP(
            key="ambiguous_green_pending_xrf",
            title="SOP: Ambiguous Green Pending XRF",
            summary=summary,
            steps=(
                "Record the case as ambiguous green or unresolved green, not as emerald green or chrome green.",
                "Use normal rare-material precautions, but avoid dry abrasion or unnecessary handling until the green system is screened instrumentally.",
                "Run qualitative XRF next. Use As + Cu, Pb + Cr, Fe support, or the absence of hazard elements to decide whether the colour ambiguity reflects a hazardous system or an ordinary green colourant.",
                sheltered_step,
                local_reference_step,
                "If XRF is non-specific or negative, downgrade the record from hazard-class language to a contextual green watchlist note.",
            ),
        )

    if top.lab_fit_label in PROVISIONAL_LAB_FIT_LABELS or top.lab_fit_label == LAB_FIT_CLOSE:
        summary = (
            f"The current result is a provisional colour-based flag for {top.profile.label.lower()} with no decisive elemental confirmation yet."
        )
        if uses_fallback:
            summary += " It still depends on built-in fallback anchors."
        return ScenarioSOP(
            key="provisional_colour_flag",
            title="SOP: Provisional Colour Flag",
            summary=summary,
            steps=(
                "Record the result as provisional and do not convert it into a pigment statement.",
                "Use qualitative XRF before treatment, rehousing changes, or access decisions that depend on hazard class.",
                sheltered_step,
                local_reference_step,
                "If repeated screens stay provisional and XRF remains non-specific, classify the item as unresolved rather than forcing a pigment family.",
            ),
        )

    return ScenarioSOP(
        key="lower_priority_review",
        title="SOP: Lower-Priority Review Case",
        summary=(
            "The current inputs do not support a strong hazardous-pigment scenario by colour or XRF alone. Keep the record reviewable, but do not escalate on colour evidence alone."
        ),
        steps=(
            "Use standard rare-material handling and avoid unnecessary intervention.",
            "Record the result as low-confidence or not supported by current screening evidence.",
            "Retest only if a new cue appears, such as a stronger XRF pattern, a better sheltered-area measurement, or a more comparable local reference.",
        ),
    )


def scenario_sop_library() -> Tuple[ScenarioSOP, ...]:
    return (
        ScenarioSOP(
            key="arsenical_copper_green",
            title="Arsenical Copper Green Screen",
            summary="Use when As + Cu appears on green cloth or the case is otherwise being managed as an arsenical green screen.",
            steps=(
                "Limit handling and isolate the volume from routine circulation.",
                "Avoid abrasive cleaning or treatment until the handling plan is set.",
                "Record the XRF pattern and any friability or transfer risk.",
                "Confirm instrumentally only if compound-level identification will change decisions.",
            ),
        ),
        ScenarioSOP(
            key="lead_chromium_green",
            title="Lead/Chromium Green Screen",
            summary="Use when Pb + Cr appears on green cloth and the case is being handled as a lead/chromium green system.",
            steps=(
                "Use controlled handling and avoid abrasive manipulation.",
                "Record the result as a Pb/Cr green-system screen rather than a specific pigment identity.",
                "Note Fe support separately if present.",
                "Confirm further only if the exact green system matters operationally.",
            ),
        ),
        ScenarioSOP(
            key="lead_chromium_yellow",
            title="Lead/Chromium Yellow-Orange Screen",
            summary="Use when Pb + Cr appears on yellow, orange, or brown cloth.",
            steps=(
                "Handle as a Pb/Cr case and minimise abrasive contact.",
                "Record the result as a lead-chromate-family screen.",
                "Use further spectroscopy only if the exact pigment variant matters.",
            ),
        ),
        ScenarioSOP(
            key="mercury_watchlist",
            title="Mercury Watchlist Case",
            summary="Use when Hg appears on a red component or the case otherwise needs specialist mercury review.",
            steps=(
                "Keep handling controlled and route through specialist review.",
                "Record Hg as a watchlist signal rather than a pigment identification.",
                "Confirm further only if treatment or access decisions require it.",
            ),
        ),
        ScenarioSOP(
            key="ambiguous_green_pending_xrf",
            title="Ambiguous Green Pending XRF",
            summary="Use when green cloth is contextually suspicious but the colour fit is poor or split between classes.",
            steps=(
                "Record the case as unresolved green rather than assigning a pigment family.",
                "Run qualitative XRF next.",
                "Measure sheltered areas if available.",
                "Use a same-device local reference library before relying on colour distance again.",
            ),
        ),
        ScenarioSOP(
            key="provisional_colour_flag",
            title="Provisional Colour Flag",
            summary="Use when a class is near by colour but still lacks decisive XRF or local-reference support.",
            steps=(
                "Keep the result provisional.",
                "Do not infer chemistry from colour alone.",
                "Add local references and XRF before escalation.",
            ),
        ),
    )


def evaluate_binding(
    inputs: ScreeningInput, reference_samples: Sequence[ReferenceSample]
) -> ScreeningOutcome:
    candidates = [
        score_emerald_green(inputs, reference_samples),
        score_chrome_green(inputs, reference_samples),
        score_chrome_yellow(inputs, reference_samples),
        score_mercury_red(inputs, reference_samples),
    ]
    ranked = tuple(sorted(candidates, key=lambda candidate: candidate.score, reverse=True))
    priority_label, priority_explanation = determine_priority(inputs, ranked)
    handling_label, handling_advice = determine_handling(inputs, ranked)
    xrf_summary = interpret_xrf(inputs.xrf_elements, inputs.color_family)
    color_note = nearest_lab_class_note(inputs, reference_samples)
    return ScreeningOutcome(
        priority_label=priority_label,
        priority_explanation=priority_explanation,
        handling_label=handling_label,
        handling_advice=handling_advice,
        xrf_summary=xrf_summary,
        candidates=ranked,
        color_note=color_note,
        reference_library_note=reference_library_note(reference_samples),
    )


def plot_color_context(
    observed_lab: LABColor, reference_samples: Sequence[ReferenceSample]
) -> go.Figure:
    fig = go.Figure()
    cluster_models = get_cluster_models(reference_samples)
    plot_labels = {
        "emerald_green": "Emerald-green-like",
        "chrome_green": "Chrome-green-like",
        "chrome_yellow": "Chrome-yellow-like",
        "mercury_red": "Mercury-red-like",
    }
    label_offsets = {
        "emerald_green": (18, 6),
        "chrome_green": (0, 18),
        "chrome_yellow": (0, 16),
        "mercury_red": (-16, 8),
        "observed": (0, -22),
    }
    all_a_values = [observed_lab.a]
    all_b_values = [observed_lab.b]

    for profile in HAZARD_CLASS_PROFILES.values():
        model = cluster_models[profile.cluster_key]
        swatch = model.centroid
        reference_points = model.references
        all_a_values.extend(sample.color.a for sample in reference_points)
        all_a_values.append(model.centroid.a)
        all_b_values.extend(sample.color.b for sample in reference_points)
        all_b_values.append(model.centroid.b)

        fig.add_trace(
            go.Scatter(
                x=[sample.color.a for sample in reference_points],
                y=[sample.color.b for sample in reference_points],
                mode="markers",
                marker=dict(
                    size=10,
                    color=[sample.color.to_hex() for sample in reference_points],
                    opacity=0.65,
                    line=dict(color="#2c241c", width=0.9),
                ),
                name=profile.label,
                showlegend=False,
                cliponaxis=False,
                customdata=[
                    [sample.color.name, sample.source, sample.color.L, sample.color.to_hex()]
                    for sample in reference_points
                ],
                hovertemplate=(
                    "%{customdata[0]}"
                    "<br>Class: "
                    f"{plot_labels.get(profile.key, profile.label)}"
                    "<br>a*: %{x:.1f}"
                    "<br>b*: %{y:.1f}"
                    "<br>L*: %{customdata[2]:.1f}"
                    "<br>sRGB: %{customdata[3]}"
                    "<br>Source: %{customdata[1]}"
                    "<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[model.centroid.a],
                y=[model.centroid.b],
                mode="markers",
                marker=dict(
                    size=17,
                    color=swatch.to_hex(),
                    symbol="x",
                    line=dict(color="#2c241c", width=1.4),
                ),
                name=f"{profile.label} centroid",
                showlegend=False,
                cliponaxis=False,
                hovertemplate=(
                    f"{plot_labels.get(profile.key, profile.label)} centroid"
                    "<br>a*: %{x:.1f}"
                    "<br>b*: %{y:.1f}"
                    f"<br>L*: {model.centroid.L:.1f}"
                    f"<br>Reference count: {len(reference_points)}"
                    f"<br>Mean class ΔE00: {model.mean_distance:.2f}"
                    "<extra></extra>"
                ),
            )
        )
        xshift, yshift = label_offsets.get(profile.key, (0, 16))
        fig.add_annotation(
            x=model.centroid.a,
            y=model.centroid.b,
            text=plot_labels.get(profile.key, profile.label),
            showarrow=False,
            xshift=xshift,
            yshift=yshift,
            font=dict(size=11, color="#5a4f43"),
            bgcolor="rgba(255,249,240,0.9)",
            bordercolor="rgba(120,104,82,0.22)",
            borderwidth=1,
            borderpad=3,
        )

    fig.add_trace(
        go.Scatter(
            x=[observed_lab.a],
            y=[observed_lab.b],
            mode="markers",
            marker=dict(
                size=18,
                color=observed_lab.to_hex(),
                symbol="diamond",
                line=dict(color="#201914", width=1.5),
            ),
            name="Observed swatch",
            showlegend=False,
            cliponaxis=False,
            hovertemplate=(
                "Observed swatch"
                "<br>a*: %{x:.1f}"
                "<br>b*: %{y:.1f}"
                f"<br>L*: {observed_lab.L:.1f}"
                "<extra></extra>"
            ),
        )
    )
    observed_xshift, observed_yshift = label_offsets["observed"]
    fig.add_annotation(
        x=observed_lab.a,
        y=observed_lab.b,
        text="Observed swatch",
        showarrow=False,
        xshift=observed_xshift,
        yshift=observed_yshift,
        font=dict(size=11, color="#5a4f43"),
        bgcolor="rgba(255,249,240,0.95)",
        bordercolor="rgba(120,104,82,0.22)",
        borderwidth=1,
        borderpad=3,
    )

    x_padding = 12
    y_padding = 10

    fig.update_layout(
        title=dict(
            text="Reference Library in a* / b* Space",
            x=0.02,
            y=0.985,
            xanchor="left",
            yanchor="top",
            pad=dict(t=8, b=8),
        ),
        xaxis_title="a* (green to red)",
        yaxis_title="b* (blue to yellow)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,249,240,0.72)",
        margin=dict(l=36, r=28, t=110, b=54),
        height=470,
        hovermode="closest",
        font=dict(color="#5a4f43"),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=1.08,
        xanchor="left",
        yanchor="bottom",
        showarrow=False,
        align="left",
        font=dict(size=11, color="#6b5d4b"),
        text=f"Observed L*: {observed_lab.L:.1f} | circles = reference samples | crosses = class centroids",
    )
    fig.update_xaxes(
        range=[min(all_a_values) - x_padding, max(all_a_values) + x_padding],
        automargin=True,
        showgrid=True,
        gridcolor="rgba(141, 162, 191, 0.18)",
        zeroline=False,
    )
    fig.update_yaxes(
        range=[min(all_b_values) - y_padding, max(all_b_values) + y_padding],
        automargin=True,
        showgrid=True,
        gridcolor="rgba(141, 162, 191, 0.18)",
        zeroline=False,
    )
    fig.add_hline(y=0, line_dash="dot", line_color="#7b6a52")
    fig.add_vline(x=0, line_dash="dot", line_color="#7b6a52")
    return fig


def build_pdf_color_context_plot(
    observed_lab: LABColor, reference_samples: Sequence[ReferenceSample]
) -> Drawing:
    width = 470
    height = 330
    margin_left = 48
    margin_right = 20
    margin_top = 34
    margin_bottom = 44
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    drawing = Drawing(width, height)

    cluster_models = get_cluster_models(reference_samples)
    plot_labels = {
        "emerald_green": "Emerald-green-like",
        "chrome_green": "Chrome-green-like",
        "chrome_yellow": "Chrome-yellow-like",
        "mercury_red": "Mercury-red-like",
    }
    label_offsets = {
        "emerald_green": (-14, 8),
        "chrome_green": (-18, 11),
        "chrome_yellow": (-24, 10),
        "mercury_red": (-10, 10),
        "observed": (-18, -15),
    }

    all_a_values = [observed_lab.a]
    all_b_values = [observed_lab.b]
    for profile in HAZARD_CLASS_PROFILES.values():
        model = cluster_models[profile.cluster_key]
        reference_points = model.references
        all_a_values.extend(sample.color.a for sample in reference_points)
        all_a_values.append(model.centroid.a)
        all_b_values.extend(sample.color.b for sample in reference_points)
        all_b_values.append(model.centroid.b)

    x_min = min(all_a_values) - 12
    x_max = max(all_a_values) + 12
    y_min = min(all_b_values) - 10
    y_max = max(all_b_values) + 10

    def to_x(value: float) -> float:
        if x_max == x_min:
            return margin_left + plot_width / 2
        return margin_left + ((value - x_min) / (x_max - x_min)) * plot_width

    def to_y(value: float) -> float:
        if y_max == y_min:
            return margin_bottom + plot_height / 2
        return margin_bottom + ((value - y_min) / (y_max - y_min)) * plot_height

    background = Rect(
        margin_left,
        margin_bottom,
        plot_width,
        plot_height,
        fillColor=colors.HexColor("#fff9f0"),
        strokeColor=colors.HexColor("#d4c4ad"),
        strokeWidth=0.8,
    )
    drawing.add(background)

    tick_count = 5
    for value in np.linspace(x_min, x_max, tick_count):
        x_coord = to_x(float(value))
        drawing.add(
            Line(
                x_coord,
                margin_bottom,
                x_coord,
                margin_bottom + plot_height,
                strokeColor=colors.HexColor("#d9ddd8"),
                strokeWidth=0.5,
            )
        )
        drawing.add(
            String(
                x_coord - 8,
                margin_bottom - 16,
                f"{value:.0f}",
                fontName="Helvetica",
                fontSize=7,
                fillColor=colors.HexColor("#6b5d4b"),
            )
        )
    for value in np.linspace(y_min, y_max, tick_count):
        y_coord = to_y(float(value))
        drawing.add(
            Line(
                margin_left,
                y_coord,
                margin_left + plot_width,
                y_coord,
                strokeColor=colors.HexColor("#d9ddd8"),
                strokeWidth=0.5,
            )
        )
        drawing.add(
            String(
                margin_left - 26,
                y_coord - 3,
                f"{value:.0f}",
                fontName="Helvetica",
                fontSize=7,
                fillColor=colors.HexColor("#6b5d4b"),
            )
        )

    if x_min <= 0 <= x_max:
        zero_x = to_x(0.0)
        zero_line = Line(
            zero_x,
            margin_bottom,
            zero_x,
            margin_bottom + plot_height,
            strokeColor=colors.HexColor("#7b6a52"),
            strokeWidth=0.8,
        )
        zero_line.strokeDashArray = [2, 2]
        drawing.add(zero_line)
    if y_min <= 0 <= y_max:
        zero_y = to_y(0.0)
        zero_line = Line(
            margin_left,
            zero_y,
            margin_left + plot_width,
            zero_y,
            strokeColor=colors.HexColor("#7b6a52"),
            strokeWidth=0.8,
        )
        zero_line.strokeDashArray = [2, 2]
        drawing.add(zero_line)

    for profile in HAZARD_CLASS_PROFILES.values():
        model = cluster_models[profile.cluster_key]
        for sample in model.references:
            drawing.add(
                Circle(
                    to_x(sample.color.a),
                    to_y(sample.color.b),
                    3.6,
                    fillColor=colors.HexColor(sample.color.to_hex()),
                    strokeColor=colors.HexColor("#2c241c"),
                    strokeWidth=0.5,
                )
            )

        centroid_x = to_x(model.centroid.a)
        centroid_y = to_y(model.centroid.b)
        swatch = colors.HexColor(model.centroid.to_hex())
        drawing.add(Line(centroid_x - 5, centroid_y - 5, centroid_x + 5, centroid_y + 5, strokeColor=swatch, strokeWidth=1.6))
        drawing.add(Line(centroid_x - 5, centroid_y + 5, centroid_x + 5, centroid_y - 5, strokeColor=swatch, strokeWidth=1.6))
        label_dx, label_dy = label_offsets.get(profile.key, (6, 10))
        drawing.add(
            String(
                centroid_x + label_dx,
                centroid_y + label_dy,
                plot_labels.get(profile.key, profile.label),
                fontName="Helvetica",
                fontSize=7,
                fillColor=colors.HexColor("#5a4f43"),
            )
        )

    observed_x = to_x(observed_lab.a)
    observed_y = to_y(observed_lab.b)
    drawing.add(
        Polygon(
            [
                observed_x,
                observed_y + 6,
                observed_x + 6,
                observed_y,
                observed_x,
                observed_y - 6,
                observed_x - 6,
                observed_y,
            ],
            fillColor=colors.HexColor(observed_lab.to_hex()),
            strokeColor=colors.HexColor("#201914"),
            strokeWidth=1.0,
        )
    )
    observed_dx, observed_dy = label_offsets["observed"]
    drawing.add(
        String(
            observed_x + observed_dx,
            observed_y + observed_dy,
            "Observed swatch",
            fontName="Helvetica",
            fontSize=7,
            fillColor=colors.HexColor("#5a4f43"),
        )
    )

    drawing.add(
        String(
            margin_left,
            height - 18,
            "Reference Library in a* / b* Space",
            fontName="Helvetica-Bold",
            fontSize=10,
            fillColor=colors.HexColor("#4e3517"),
        )
    )
    drawing.add(
        String(
            margin_left,
            height - 30,
            f"Observed L*: {observed_lab.L:.1f} | circles = references | crosses = centroids | diamond = observed",
            fontName="Helvetica",
            fontSize=7,
            fillColor=colors.HexColor("#6b5d4b"),
        )
    )
    drawing.add(
        String(
            margin_left + plot_width / 2 - 42,
            12,
            "a* (green to red)",
            fontName="Helvetica",
            fontSize=8,
            fillColor=colors.HexColor("#5a4f43"),
        )
    )
    drawing.add(
        String(
            3,
            margin_bottom + plot_height / 2,
            "b* (blue to yellow)",
            fontName="Helvetica",
            fontSize=8,
            fillColor=colors.HexColor("#5a4f43"),
            angle=90,
        )
    )
    return drawing


@st.cache_data(show_spinner=False)
def parse_marc_records(content: bytes) -> List[MARCTemplateRecord]:
    records: List[MARCTemplateRecord] = []
    reader = MARCReader(io.BytesIO(content), to_unicode=True, force_utf8=True)
    for index, record in enumerate(reader, start=1):
        title = ""
        if hasattr(record, "title") and record.title():
            title = record.title().strip(" /")
        elif record.get_fields("245"):
            field = record.get_fields("245")[0]
            title = " ".join(field.get_subfields("a", "b")).strip(" /")

        year = None
        for tag in ("264", "260"):
            for field in record.get_fields(tag):
                for value in field.get_subfields("c"):
                    year = parse_optional_year(value)
                    if year is not None:
                        break
                if year is not None:
                    break
            if year is not None:
                break

        if year is None and record.get_fields("008"):
            year = parse_optional_year(record["008"].data[7:11])

        oclc = ""
        for field in record.get_fields("035"):
            for value in field.get_subfields("a"):
                if "OCoLC" in value or value.strip():
                    oclc = value.strip()
                    break
            if oclc:
                break

        color_terms = extract_marc_color_terms(record)
        records.append(
            MARCTemplateRecord(
                record_id=f"record-{index}",
                title=title or f"Untitled record {index}",
                year=year,
                oclc=oclc,
                color_terms=color_terms,
            )
        )
    return records


def records_to_template(records: Sequence[MARCTemplateRecord]) -> pd.DataFrame:
    rows = []
    for record in records:
        primary_color_name = record.color_terms[0] if record.color_terms else ""
        named_color = lookup_named_color(primary_color_name) if primary_color_name else None
        rows.append(
            {
                "record_id": record.record_id,
                "title": record.title,
                "year": record.year,
                "oclc": record.oclc,
                "marc_color_terms": "; ".join(record.color_terms),
                "color_name": primary_color_name,
                "region": "Unknown",
                "binding_type": "Original cloth-case binding",
                "binding_color": infer_color_family_from_name(primary_color_name) if primary_color_name else "",
                "vividness": "Unknown",
                "fading_evidence": "Unknown",
                "spine_browned": "",
                "stamped_decoration": "",
                "xrf_elements": "",
                "swatch_hex": named_color.to_hex() if named_color else "",
                "lab_l": named_color.L if named_color else "",
                "lab_a": named_color.a if named_color else "",
                "lab_b": named_color.b if named_color else "",
                "lab_replicates": "",
            }
        )
    return pd.DataFrame(rows)


def example_batch_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "record_id": "demo-1",
                "title": "The Court Album",
                "year": 1853,
                "oclc": "",
                "marc_color_terms": "Vivid green",
                "color_name": "Vivid green",
                "region": "United Kingdom / Ireland",
                "binding_type": "Original cloth-case binding",
                "binding_color": "Green",
                "vividness": "Vivid/bright",
                "fading_evidence": "Possible fading/browning",
                "spine_browned": "yes",
                "stamped_decoration": "yes",
                "xrf_elements": "As,Cu",
                "swatch_hex": "#1f8b63",
                "lab_l": "",
                "lab_a": "",
                "lab_b": "",
                "lab_replicates": "55.0,-40.8,27.1|54.6,-39.9,28.2|55.4,-40.1,27.6",
            },
            {
                "record_id": "demo-2",
                "title": "The Fern Garden",
                "year": 1872,
                "oclc": "",
                "marc_color_terms": "Olive green",
                "color_name": "Olive green",
                "region": "United Kingdom / Ireland",
                "binding_type": "Original cloth-case binding",
                "binding_color": "Green",
                "vividness": "Muted/olive",
                "fading_evidence": "No obvious fading",
                "spine_browned": "no",
                "stamped_decoration": "yes",
                "xrf_elements": "Pb,Cr",
                "swatch_hex": "#69794a",
                "lab_l": "",
                "lab_a": "",
                "lab_b": "",
                "lab_replicates": "47.4,-15.9,23.0|48.0,-16.8,24.1|47.3,-16.2,22.7",
            },
        ]
    )


def row_to_screening_input(row: pd.Series) -> ScreeningInput:
    catalogued_color_name = (
        str(
            row.get("color_name")
            or row.get("catalogued_color_name")
            or row.get("marc_color_term")
            or ""
        ).strip()
    )
    if not catalogued_color_name:
        marc_color_terms = str(row.get("marc_color_terms", "")).strip()
        if marc_color_terms:
            catalogued_color_name = marc_color_terms.split(";")[0].strip()

    named_color = lookup_named_color(catalogued_color_name) if catalogued_color_name else None

    swatch_hex = str(row.get("swatch_hex", "")).strip()
    if not swatch_hex:
        swatch_hex = named_color.to_hex() if named_color else DEFAULT_BATCH_SWATCH_HEX
    if not swatch_hex.startswith("#"):
        swatch_hex = f"#{swatch_hex}"

    manual_L = pd.to_numeric(row.get("lab_l"), errors="coerce")
    manual_a = pd.to_numeric(row.get("lab_a"), errors="coerce")
    manual_b = pd.to_numeric(row.get("lab_b"), errors="coerce")
    use_manual_lab = not np.isnan([manual_L, manual_a, manual_b]).any()
    replicate_measurements = parse_replicate_measurements_field(
        row.get("lab_replicates"),
        str(row.get("title", "Observed binding")).strip() or "Observed binding",
    )
    measurement_summary = summarize_measurements(
        replicate_measurements,
        str(row.get("title", "Observed binding")).strip() or "Observed binding",
    )

    if measurement_summary is not None:
        observed_lab = measurement_summary.mean_lab
    else:
        observed_lab = make_observed_lab(
            name=str(row.get("title", "Observed binding")),
            swatch_hex=swatch_hex,
            manual_L=float(manual_L) if use_manual_lab else None,
            manual_a=float(manual_a) if use_manual_lab else None,
            manual_b=float(manual_b) if use_manual_lab else None,
            named_color=named_color,
        )

    color_family = str(row.get("binding_color", "")).strip()
    if not color_family:
        if catalogued_color_name:
            color_family = infer_color_family_from_name(catalogued_color_name)
        if color_family == "Unknown" or not color_family:
            inferred_name, _ = infer_named_color_from_lab(observed_lab)
            color_family = infer_color_family_from_name(inferred_name.name)

    return ScreeningInput(
        title=str(row.get("title", "")).strip() or "Untitled binding",
        publication_year=parse_optional_year(row.get("year")),
        region=str(row.get("region", "Unknown")).strip() or "Unknown",
        binding_type=str(row.get("binding_type", "Cloth binding, date or originality uncertain")).strip(),
        color_family=color_family or "Unknown",
        vividness=str(row.get("vividness", "Unknown")).strip() or "Unknown",
        spine_browned=to_bool_or_none(row.get("spine_browned")),
        stamped_decoration=to_bool_or_none(row.get("stamped_decoration")),
        xrf_elements=parse_xrf_elements(row.get("xrf_elements")),
        observed_lab=observed_lab,
        measurement_summary=measurement_summary,
        fading_evidence=str(row.get("fading_evidence", "Unknown")).strip() or "Unknown",
    )


def batch_screen(
    df: pd.DataFrame, reference_samples: Sequence[ReferenceSample]
) -> pd.DataFrame:
    required = {"title"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Batch CSV is missing required columns: {', '.join(sorted(missing))}")

    output_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        inputs = row_to_screening_input(row)
        outcome = evaluate_binding(inputs, reference_samples)
        sop = build_sop(inputs, outcome)
        top = outcome.candidates[0]
        nearest = nearest_lab_candidate(outcome.candidates)
        inferred_name, inferred_distance = infer_named_color_from_lab(inputs.observed_lab)
        top_closeness_label, _, top_closeness_note = colour_closeness_signal(top)
        output_rows.append(
            {
                "record_id": row.get("record_id", ""),
                "title": inputs.title,
                "year": inputs.publication_year,
                "color_name": row.get("color_name", row.get("catalogued_color_name", "")),
                "binding_color": inputs.color_family,
                "inferred_color_name": inferred_name.name,
                "inferred_color_distance_delta_e00": round(inferred_distance, 2),
                "xrf_elements": ", ".join(inputs.xrf_elements),
                "priority": outcome.priority_label,
                "sop_scenario": sop.title,
                "sop_summary": sop.summary,
                "top_screening_class": top.profile.label,
                "top_lab_class": top.profile.label,
                "top_class_delta_e00_to_reference": round(top.delta_e00, 2),
                "top_class_raw_delta_e00_to_reference": round(top.raw_delta_e00, 2),
                "top_class_fading_adjustment_delta_e00": round(top.fading_adjustment, 2),
                "top_class_nearest_reference_name": top.nearest_reference_name,
                "top_class_reference_basis": reference_basis_label(top.has_local_reference),
                "top_class_closeness_signal": top_closeness_label,
                "top_class_closeness_note": top_closeness_note,
                "nearest_lab_class": nearest.profile.label,
                "nearest_lab_delta_e00_to_reference": round(nearest.delta_e00, 2),
                "nearest_lab_raw_delta_e00_to_reference": round(nearest.raw_delta_e00, 2),
                "nearest_lab_fading_adjustment_delta_e00": round(nearest.fading_adjustment, 2),
                "nearest_lab_reference_name": nearest.nearest_reference_name,
                "nearest_lab_reference_basis": reference_basis_label(nearest.has_local_reference),
                "colour_screen_band": top.lab_fit_label,
                "class_score": top.score,
                "evidence_level": top.confidence,
                "handling": outcome.handling_label,
                "xrf_summary": outcome.xrf_summary,
                "measurement_count": inputs.measurement_summary.count if inputs.measurement_summary else 1,
                "measurement_mean_delta_e00": round(inputs.measurement_summary.mean_delta_e00, 2)
                if inputs.measurement_summary
                else "",
                "measurement_max_delta_e00": round(inputs.measurement_summary.max_delta_e00, 2)
                if inputs.measurement_summary
                else "",
                "reasoning": " | ".join(top.evidence[:3]),
            }
        )
    return pd.DataFrame(output_rows)


def render_priority_card(label: str, explanation: str) -> None:
    css_class = "priority-low"
    chip_label = "Lower priority"
    if label.startswith("Urgent"):
        css_class = "priority-urgent"
        chip_label = "Urgent"
    elif label.startswith("High"):
        css_class = "priority-high"
        chip_label = "High priority"
    elif label.startswith("Moderate"):
        css_class = "priority-moderate"
        chip_label = "Moderate priority"

    st.markdown(
        f"""
        <div class="priority-card {css_class}">
            <div class="priority-card-header">
                <div class="smallcaps">Testing Priority</div>
                <div class="priority-chip">{escape(chip_label)}</div>
            </div>
            <h3>{escape(label)}</h3>
            <p>{escape(explanation)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_metric_card(
    label: str,
    value: str,
    note: Optional[str] = None,
    *,
    value_is_html: bool = False,
    extra_class: str = "",
) -> str:
    value_html = value if value_is_html else escape(value)
    note_html = f'<div class="metric-card-note">{escape(note)}</div>' if note else ""
    class_name = "metric-card"
    if extra_class:
        class_name = f"{class_name} {extra_class}"
    return (
        f'<div class="{class_name}">'
        f'<div class="metric-card-label">{escape(label)}</div>'
        f'<div class="metric-card-value">{value_html}</div>'
        f"{note_html}"
        "</div>"
    )


def render_metric_grid(cards: Sequence[str], compact: bool = False) -> None:
    for start in range(0, len(cards), 2):
        row_cards = cards[start:start + 2]
        columns = st.columns(len(row_cards), gap="small")
        for column, card in zip(columns, row_cards):
            wrapper_start = '<div class="metric-grid-compact">' if compact else ""
            wrapper_end = "</div>" if compact else ""
            column.markdown(f"{wrapper_start}{card}{wrapper_end}", unsafe_allow_html=True)


def render_support_note(paragraphs: Sequence[str], title: Optional[str] = None) -> None:
    title_html = f'<div class="smallcaps">{escape(title)}</div>' if title else ""
    paragraphs_html = "".join(f"<p>{escape(paragraph)}</p>" for paragraph in paragraphs if paragraph)
    if not paragraphs_html:
        return
    st.markdown(
        f'<div class="support-note">{title_html}{paragraphs_html}</div>',
        unsafe_allow_html=True,
    )


def render_sop_card(sop: ScenarioSOP, *, title: str = "Scenario SOP") -> None:
    st.markdown(f"### {title}")
    st.info(sop.summary)
    st.markdown(f"**{sop.title}**")
    st.markdown("\n".join(f"{index}. {step}" for index, step in enumerate(sop.steps, start=1)))


def render_summary_metrics(outcome: ScreeningOutcome) -> None:
    top = outcome.candidates[0]
    nearest = nearest_lab_candidate(outcome.candidates)
    closeness_label, closeness_css_class, closeness_note = colour_closeness_signal(top)
    cards = [
        build_metric_card(
            "Top screening class",
            top.profile.label,
            "Overall rank using colour, book context, and any XRF evidence.",
        ),
        build_metric_card(
            "Nearest LAB class",
            nearest.profile.label,
            f"Pure colour distance against {reference_basis_label(nearest.has_local_reference).lower()}.",
        ),
        build_metric_card(
            "Colour-screen band",
            top.lab_fit_label,
            (
                f"Nearest reference: {top.nearest_reference_name} at ΔE00 "
                f"{format_delta_e00_value(top.delta_e00, top.raw_delta_e00, top.fading_adjustment)}."
            ),
        ),
        build_metric_card(
            "Closeness cue",
            f'<span class="signal-pill {closeness_css_class}">{escape(closeness_label)}</span>',
            closeness_note,
            value_is_html=True,
            extra_class="metric-card-signal",
        ),
    ]
    render_metric_grid(cards)
    notes = [
        "Traffic-light closeness is a colour-similarity cue only: green means close to uploaded local references, "
        "amber means provisional or loose similarity, and red means not close by colour alone."
    ]
    if nearest.profile.key != top.profile.key:
        notes.append(
            "Nearest LAB class is based only on colour distance to the reference library. "
            "Top screening class also includes year, binding context, visual observations, and any XRF evidence."
        )
    render_support_note(notes, title="How To Read This")


def render_measurement_summary(measurement_summary: Optional[MeasurementSummary]) -> None:
    if measurement_summary is None:
        return
    metrics = [
        build_metric_card("Measurements", str(measurement_summary.count)),
        build_metric_card("Mean ΔE00", f"{measurement_summary.mean_delta_e00:.2f}"),
        build_metric_card("Max ΔE00", f"{measurement_summary.max_delta_e00:.2f}"),
        build_metric_card(
            "Mean LAB",
            (
                f"L* {measurement_summary.mean_lab.L:.1f} | "
                f"a* {measurement_summary.mean_lab.a:.1f} | "
                f"b* {measurement_summary.mean_lab.b:.1f}"
            ),
        ),
    ]
    render_metric_grid(metrics, compact=True)
    render_support_note(
        [
            "Within-set spread helps show whether your measurements are tighter than the decision band. "
            "The app downgrades colour bands when the within-set spread crosses the active threshold."
        ],
        title="Replicate Spread",
    )


def render_candidate(
    assessment: CandidateAssessment, reference_samples: Sequence[ReferenceSample]
) -> None:
    swatch = get_reference_swatches(reference_samples)[assessment.profile.cluster_key]
    closeness_label, closeness_css_class, closeness_note = colour_closeness_signal(assessment)
    evidence_html = "".join(f"<li>{item}</li>" for item in assessment.evidence[:4]) or "<li>No specific evidence entered.</li>"
    st.markdown(
        f"""
        <div class="candidate-card">
            <div class="smallcaps">Hazard Colour Class</div>
            <h4>{assessment.profile.label} <span style="font-size:0.88rem; color:#7b6141;">({assessment.score}/99)</span></h4>
            <div class="candidate-meta">
                {assessment.profile.class_basis} • {assessment.profile.risk_label} • {assessment.confidence}
            </div>
            <div style="display:flex; gap:0.9rem; align-items:flex-start;">
                <div style="width:56px; height:56px; border-radius:12px; background:{swatch.to_hex()}; border:1px solid rgba(32,25,20,0.22);"></div>
                <div style="flex:1;">
                    <p style="margin:0 0 0.45rem 0;">{assessment.summary}</p>
                    <p style="margin:0 0 0.45rem 0;"><strong>Colour screen:</strong> {assessment.lab_fit_label}</p>
                    <p style="margin:0 0 0.45rem 0;"><strong>Closeness signal:</strong> <span class="signal-pill {closeness_css_class}">{closeness_label}</span> {closeness_note}</p>
                    <p style="margin:0 0 0.45rem 0;"><strong>Reference basis:</strong> {reference_basis_label(assessment.has_local_reference)}</p>
                    <p style="margin:0 0 0.45rem 0;"><strong>Nearest reference:</strong> {assessment.nearest_reference_name} ({assessment.reference_source}; {assessment.reference_count} reference points in class)</p>
                    <p style="margin:0 0 0.45rem 0;"><strong>LAB distance:</strong> primary ΔE00 {format_delta_e00_value(assessment.delta_e00, assessment.raw_delta_e00, assessment.fading_adjustment)}; secondary ΔE76 {assessment.delta_e76:.2f}</p>
                    <p style="margin:0 0 0.45rem 0;"><strong>Research support:</strong> bookcloth use {assessment.profile.use_evidence_confidence}; colour data {assessment.profile.color_data_confidence}</p>
                    <p style="margin:0 0 0.45rem 0;"><strong>Score basis:</strong> the score combines LAB distance with date, binding context, visual observations, and any XRF evidence.</p>
                    <ul style="margin:0 0 0.45rem 1rem; padding:0;">{evidence_html}</ul>
                    <p style="margin:0;"><strong>Next step:</strong> {assessment.next_step}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_report_sections(
    inputs: ScreeningInput, outcome: ScreeningOutcome
) -> List[Tuple[str, List[str]]]:
    top = outcome.candidates[0]
    nearest = nearest_lab_candidate(outcome.candidates)
    sop = build_sop(inputs, outcome)
    closeness_label, _, closeness_note = colour_closeness_signal(top)
    measurement_lines = []
    if inputs.measurement_summary is not None:
        measurement_lines.extend(
            [
                f"Repeated measurements averaged: {inputs.measurement_summary.count}",
                f"Within-set mean ΔE00: {inputs.measurement_summary.mean_delta_e00:.2f}",
                f"Within-set max ΔE00: {inputs.measurement_summary.max_delta_e00:.2f}",
            ]
        )
    return [
        (
            "Binding",
            [
                f"Title: {inputs.title}",
                f"Year: {inputs.publication_year or 'Unknown'}",
                f"Region: {inputs.region}",
                f"Binding type: {inputs.binding_type}",
                f"Observed family: {inputs.color_family}",
                f"Vividness: {inputs.vividness}",
                f"Fading evidence: {inputs.fading_evidence}",
                f"XRF elements: {', '.join(inputs.xrf_elements) if inputs.xrf_elements else 'None entered'}",
            ]
            + measurement_lines,
        ),
        (
            "Assessment",
            [
                f"Priority: {outcome.priority_label}",
                f"Top screening class: {top.profile.label} ({top.score}/99, {top.confidence})",
                (
                    f"Nearest LAB class: {nearest.profile.label} "
                    f"(nearest-reference ΔE00 "
                    f"{format_delta_e00_value(nearest.delta_e00, nearest.raw_delta_e00, nearest.fading_adjustment)})"
                ),
                f"Top-class colour screen band: {top.lab_fit_label}",
                f"Top-class closeness signal: {closeness_label}",
                f"Top-class reference basis: {reference_basis_label(top.has_local_reference)}",
                f"Top-class nearest reference: {top.nearest_reference_name}",
                (
                    "Top-class primary colour distance: ΔE00 "
                    f"{format_delta_e00_value(top.delta_e00, top.raw_delta_e00, top.fading_adjustment)}"
                ),
                f"Top-class secondary colour distance: ΔE76 {top.delta_e76:.2f}",
                (
                    "Top-class research support: "
                    f"bookcloth use {top.profile.use_evidence_confidence} "
                    f"; colour data {top.profile.color_data_confidence}"
                ),
                "Score basis: top screening score combines LAB distance with date, binding context, visual observations, and any XRF evidence.",
                f"Handling: {outcome.handling_label}",
            ],
        ),
        (
            "Interpretation",
            [
                outcome.priority_explanation,
                outcome.xrf_summary,
                outcome.color_note,
                outcome.reference_library_note,
                f"Closeness note: {closeness_note}",
                top.profile.strongest_date_signal,
                top.profile.exposure_risk_note,
            ],
        ),
        (
            "Scenario SOP",
            [sop.title, sop.summary] + [f"Step {index}: {step}" for index, step in enumerate(sop.steps, start=1)],
        ),
        ("Top Screening Class Evidence", list(top.evidence[:5])),
        ("Caveat", [top.profile.caveat]),
    ]


def create_screening_report_pdf(
    inputs: ScreeningInput,
    outcome: ScreeningOutcome,
    reference_samples: Sequence[ReferenceSample],
) -> bytes:
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=16 * mm,
        title=APP_TITLE,
        author="Codex",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#201914"),
        spaceAfter=5 * mm,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#59452b"),
        spaceAfter=5 * mm,
    )
    caption_style = ParagraphStyle(
        "ReportCaption",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.2,
        leading=10.5,
        textColor=colors.HexColor("#6b5d4b"),
        spaceAfter=4 * mm,
    )
    section_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        textColor=colors.HexColor("#4e3517"),
        spaceBefore=2 * mm,
        spaceAfter=2 * mm,
    )
    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=12.5,
        textColor=colors.HexColor("#201914"),
    )

    story: List[Any] = [
        Paragraph(escape(APP_TITLE), title_style),
        Paragraph(
            escape(
                "LAB-based screening report for historical hazard colour classes. "
                "This document supports triage and safer handling; it does not identify pigments from LAB alone."
            ),
            subtitle_style,
        ),
        Paragraph("Colour Context Plot", section_style),
        Paragraph(
            escape(
                "Static version of the reference-library plot used in the app. It shows a* / b* position only and should be read alongside the written assessment."
            ),
            caption_style,
        ),
        build_pdf_color_context_plot(inputs.observed_lab, reference_samples),
        Spacer(1, 4 * mm),
    ]

    for heading, bullets in build_report_sections(inputs, outcome):
        story.append(Paragraph(escape(heading), section_style))
        list_items = [
            ListItem(Paragraph(escape(bullet), bullet_style), leftIndent=0)
            for bullet in bullets
        ]
        story.append(
            ListFlowable(
                list_items,
                bulletType="bullet",
                start="circle",
                leftIndent=10,
                bulletFontName="Helvetica",
                bulletFontSize=8,
            )
        )
        story.append(Spacer(1, 2.5 * mm))

    doc.build(story)
    return buffer.getvalue()


def render_single_binding_tab(reference_samples: Sequence[ReferenceSample]) -> None:
    left, right = st.columns((1.05, 1.2), gap="large")

    with left:
        st.subheader("Single Binding Screen")
        st.markdown(
            """
            <div class="note-card">
                Enter the binding's visible features first. If you have XRF, add the elemental hits as
                qualitative signals rather than concentrations. If you have a catalogued ISCC-NBS or Getty-style
                colour term, the app can use the bundled lookup to seed LAB and infer a broad colour family.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(reference_library_note(reference_samples))
        title = st.text_input("Title or shelfmark", value="Observed binding")
        publication_year = st.number_input(
            "Publication year",
            min_value=1700,
            max_value=1999,
            value=1855,
            step=1,
        )
        region = st.selectbox("Imprint region", REGIONS, index=1)
        binding_type = st.selectbox("Binding type", BINDING_TYPES, index=0)
        catalogued_color_name = st.text_input(
            "Optional catalogued colour name",
            value="",
            help="Exact names from the bundled ISCC-NBS lookup are matched directly before swatch-based inference is used.",
        )
        named_color = lookup_named_color(catalogued_color_name) if catalogued_color_name else None
        if catalogued_color_name:
            if named_color:
                st.caption(
                    f"Matched lookup term: {named_color.name} | {named_color.to_hex()} | "
                    f"L* {named_color.L:.1f}, a* {named_color.a:.1f}, b* {named_color.b:.1f}"
                )
            else:
                st.caption("No exact lookup match found. The app will fall back to the swatch or measured LAB.")
        color_family = st.selectbox("Observed cloth colour family", COLOR_FAMILIES, index=0)
        vividness = st.selectbox("Colour character", VIVIDNESS_OPTIONS, index=0)
        spine_browned = st.selectbox(
            "Is the spine or board edge noticeably browned relative to the boards?",
            ["Unknown", "Yes", "No"],
            index=0,
        )
        fading_evidence = st.selectbox(
            "Visible fading or browning relative to sheltered areas?",
            FADING_EVIDENCE_OPTIONS,
            index=0,
            help=(
                "Conservative cue only. When fading is flagged and the observed hue stays near the class "
                "while chroma has dropped, the app may apply a small capped fading allowance."
            ),
        )
        stamped_decoration = st.selectbox(
            "Stamped or gilt decoration present?",
            ["Unknown", "Yes", "No"],
            index=1,
        )
        xrf_elements = st.multiselect(
            "Qualitative XRF elements observed",
            ELEMENT_OPTIONS,
            help="Use element hits such as As, Cu, Cr, Pb, or Hg. The app intentionally does not use ppm values.",
        )
        swatch_hex = st.color_picker(
            "Optional screen swatch of the cloth",
            value="#1f8b63" if color_family == "Green" else "#b9722f",
            help="Useful as a visual aid only. Screen colour is never treated as proof of pigment identity.",
        )
        replicate_measurements: Tuple[LABColor, ...] = tuple()
        measurement_summary: Optional[MeasurementSummary] = None
        manual_L: Optional[float] = None
        manual_a: Optional[float] = None
        manual_b: Optional[float] = None
        with st.expander("Optional measured LAB values"):
            use_manual_lab = st.checkbox("Use measured LAB instead of the screen swatch", value=False)
            if use_manual_lab:
                manual_L = st.number_input("L*", min_value=0.0, max_value=100.0, value=55.0, step=0.1)
                manual_a = st.number_input("a*", min_value=-128.0, max_value=127.0, value=-28.0, step=0.1)
                manual_b = st.number_input("b*", min_value=-128.0, max_value=127.0, value=19.0, step=0.1)
            replicate_text = st.text_area(
                "Repeated LAB measurements (optional)",
                value="",
                height=110,
                placeholder="One measurement per line, for example:\n55.0, -40.8, 27.1\n54.6, -39.9, 28.2\n55.4, -40.1, 27.6",
                help="If entered, the app averages these measurements, reports within-set ΔE00 spread, and downgrades colour-band calls when the spread crosses the active threshold.",
            )
            if replicate_text.strip():
                try:
                    replicate_measurements = parse_replicate_measurements_text(replicate_text, title or "Observed binding")
                    measurement_summary = summarize_measurements(
                        replicate_measurements,
                        title or "Observed binding",
                    )
                except ValueError as exc:
                    st.error(f"Could not parse repeated LAB measurements: {exc}")
                    return

    spine_bool = {"Yes": True, "No": False}.get(spine_browned)
    decor_bool = {"Yes": True, "No": False}.get(stamped_decoration)

    observed_lab = (
        measurement_summary.mean_lab
        if measurement_summary is not None
        else make_observed_lab(
            name=title,
            swatch_hex=swatch_hex,
            manual_L=manual_L,
            manual_a=manual_a,
            manual_b=manual_b,
            named_color=named_color if "named_color" in locals() else None,
        )
    )
    effective_color_family = color_family
    if effective_color_family == "Unknown":
        inferred_name, _ = infer_named_color_from_lab(observed_lab)
        effective_color_family = infer_color_family_from_name(inferred_name.name)

    inputs = ScreeningInput(
        title=title,
        publication_year=int(publication_year) if publication_year else None,
        region=region,
        binding_type=binding_type,
        color_family=effective_color_family,
        vividness=vividness,
        spine_browned=spine_bool,
        stamped_decoration=decor_bool,
        xrf_elements=tuple(xrf_elements),
        observed_lab=observed_lab,
        measurement_summary=measurement_summary,
        fading_evidence=fading_evidence,
    )
    outcome = evaluate_binding(inputs, reference_samples)
    sop = build_sop(inputs, outcome)

    with right:
        render_priority_card(outcome.priority_label, outcome.priority_explanation)
        render_summary_metrics(outcome)
        render_measurement_summary(measurement_summary)
        render_support_note([outcome.reference_library_note], title="Reference Basis")

        st.markdown("### Ranked Screening Classes")
        for assessment in outcome.candidates:
            render_candidate(assessment, reference_samples)

        st.markdown("### XRF Reading")
        st.info(outcome.xrf_summary)
        st.markdown("### Handling Guidance")
        st.warning(outcome.handling_advice)
        render_sop_card(sop)
        st.caption(outcome.color_note)
        st.plotly_chart(plot_color_context(observed_lab, reference_samples), use_container_width=True)
        st.caption(
            "This plot shows LAB distance only. Ranked screening classes also include year, binding context, "
            "visual observations, and any XRF evidence. Replace built-in fallback anchors with same-device local "
            "references whenever possible, and treat colour-distance calls as provisional when replicate spread is large."
        )

        report_pdf = create_screening_report_pdf(inputs, outcome, reference_samples)
        st.download_button(
            "Download screening report",
            data=report_pdf,
            file_name="cloth_binding_hazard_screen.pdf",
            mime="application/pdf",
        )


def render_batch_tab(reference_samples: Sequence[ReferenceSample]) -> None:
    st.subheader("Batch Triage")
    st.markdown(
        """
        <div class="note-card">
            Use this when you have a spreadsheet of observed bindings, or when you want a starter
            template generated from a MARC file before visual review. The app uses the active
            reference library for nearest-reference screening and the bundled ISCC-NBS palette
            for exact colour-name lookup and nearest-name inference.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(reference_library_note(reference_samples))

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### Observation CSV")
        st.download_button(
            "Download example batch template",
            data=example_batch_template().to_csv(index=False),
            file_name="hazardous_pigment_batch_template.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download local reference-library template",
            data=example_reference_library_template().to_csv(index=False),
            file_name="hazardous_pigment_reference_library_template.csv",
            mime="text/csv",
        )
        st.caption(
            "Optional columns: `lab_replicates` accepts pipe-separated measurements such as "
            "`55.0,-40.8,27.1|54.6,-39.9,28.2|55.4,-40.1,27.6`, and `fading_evidence` accepts "
            "`Unknown`, `Possible fading/browning`, or `No obvious fading`."
        )
        batch_file = st.file_uploader(
            "Upload a completed observation CSV",
            type="csv",
            key="batch_csv",
        )
        if batch_file is not None:
            try:
                batch_df = pd.read_csv(batch_file)
                results_df = batch_screen(batch_df, reference_samples)
                st.success(f"Processed {len(results_df)} binding rows.")
                st.dataframe(results_df, use_container_width=True, hide_index=True)
                st.download_button(
                    "Download batch results",
                    data=results_df.to_csv(index=False),
                    file_name="hazardous_pigment_batch_results.csv",
                    mime="text/csv",
                )
            except Exception as exc:
                st.error(f"Could not process batch CSV: {exc}")

    with col2:
        st.markdown("### MARC to Review Template")
        marc_file = st.file_uploader(
            "Upload a MARC file to generate a starter review sheet",
            type=["mrc", "marc"],
            key="marc_upload",
        )
        st.caption(
            "The starter sheet now prefers colour terms in 655$b, then falls back to binding-related MARC notes. "
            "Matching terms pre-fill colour name, broad family, swatch hex, and LAB values."
        )
        if marc_file is not None:
            try:
                records = parse_marc_records(marc_file.read())
                if not records:
                    st.warning("No MARC records were found in the uploaded file.")
                else:
                    template_df = records_to_template(records)
                    st.success(f"Extracted {len(template_df)} MARC records.")
                    st.dataframe(template_df.head(25), use_container_width=True, hide_index=True)
                    st.download_button(
                        "Download MARC starter template",
                        data=template_df.to_csv(index=False),
                        file_name="marc_cloth_binding_review_template.csv",
                        mime="text/csv",
                    )
            except Exception as exc:
                st.error(f"Could not parse MARC file: {exc}")


def render_reference_card(
    profile: HazardClassProfile, reference_samples: Sequence[ReferenceSample]
) -> None:
    cluster_model = get_cluster_models(reference_samples)[profile.cluster_key]
    local_count, fallback_count = class_reference_counts(profile.cluster_key, reference_samples)
    swatch = cluster_model.centroid
    reference_basis = (
        f"{local_count} uploaded local point{'s' if local_count != 1 else ''}; "
        f"{fallback_count} built-in fallback anchor{'s' if fallback_count != 1 else ''}"
    )
    spread_note = (
        "Class spread is calculated from the uploaded local points active for this class."
        if local_count
        else (
            "Class spread is calculated from the active built-in fallback anchors and should not be read as "
            "validated bookcloth variance."
        )
    )
    st.markdown(
        f"""
        <div class="candidate-card">
            <div class="smallcaps">LAB Hazard Class</div>
            <h4>{profile.label}</h4>
            <div class="candidate-meta">{profile.class_basis} • {profile.risk_label}</div>
            <div style="display:flex; gap:0.85rem;">
                <div style="width:54px; height:54px; border-radius:12px; background:{swatch.to_hex()}; border:1px solid rgba(32,25,20,0.22);"></div>
                <div>
                    <p style="margin:0 0 0.35rem 0;">{profile.description}</p>
                    <p style="margin:0 0 0.25rem 0;"><strong>Elemental screen:</strong> {profile.xrf_signature}</p>
                    <p style="margin:0 0 0.25rem 0;"><strong>Research support:</strong> bookcloth use {profile.use_evidence_confidence}; colour data {profile.color_data_confidence}</p>
                    <p style="margin:0 0 0.25rem 0;"><strong>Date signal:</strong> {profile.strongest_date_signal}</p>
                    <p style="margin:0 0 0.25rem 0;"><strong>Reference basis:</strong> {reference_basis}.</p>
                    <p style="margin:0 0 0.25rem 0;"><strong>Fallback evidence:</strong> {built_in_reference_detail(profile.cluster_key)}</p>
                    <p style="margin:0 0 0.25rem 0;"><strong>Class spread:</strong> mean ΔE00 to centroid {cluster_model.mean_distance:.2f}; max class ΔE00 {cluster_model.max_distance:.2f}</p>
                    <p style="margin:0 0 0.25rem 0;"><strong>Spread note:</strong> {spread_note}</p>
                    <p style="margin:0 0 0.25rem 0;"><strong>Handling note:</strong> {profile.handling_note}</p>
                    <p style="margin:0 0 0.25rem 0;"><strong>Exposure note:</strong> {profile.exposure_risk_note}</p>
                    <p style="margin:0;"><strong>Caveat:</strong> {profile.caveat}</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_reference_tab(reference_samples: Sequence[ReferenceSample]) -> None:
    st.subheader("Evidence, Methods, and Boundaries")
    st.markdown(
        """
        <div class="note-card">
            The app is tuned for original 19th-century cloth-case bindings. It becomes less reliable
            when the covering material is paper, a later rebinding, or a mixed-material structure with
            only a small coloured component.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="note-card">
            This build does not ship an empirical validation dataset for the LAB thresholds. The `ΔE00`
            bands below are current working defaults anchored to the literature and are intentionally
            downgraded when only built-in fallback anchors are available or when replicate spread crosses
            the active band. A separate fading-aware allowance may reduce the raw nearest-reference
            `ΔE00`, but only when hue stays near the class and the observed chroma has dropped.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="note-card">
            The report behind this build separates intrinsic toxicity from likely handling exposure. A
            pigment can be intrinsically hazardous yet still present lower routine-transfer risk when the
            colourant remains well adhered, while friable surface pigment increases practical exposure risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Current Working Rules")
    st.markdown(
        """
        1. Treat green cloth as the highest-priority visual class for instrumental screening.
        2. Use nearest-reference `ΔE00` as the primary colour-distance metric; keep `ΔE76` only as a secondary comparison value.
        3. If fading or browning is flagged and chroma has dropped while hue stays near the class, apply only a small capped fading allowance and report both raw and adjusted `ΔE00`.
        4. Treat `ΔE00 <= 2.0` as a high-priority match only when uploaded local references are available and replicate spread stays below that band.
        5. Treat `ΔE00 <= 5.0` as a confirmatory-testing flag, not as chemistry proof, and keep fallback-only classes provisional.
        6. Read XRF qualitatively as elemental screening: `As + Cu`, `Pb + Cr`, and `Hg` are elemental patterns, not compound confirmation.
        7. Build and maintain a same-device local reference library whenever possible.
        """
    )

    st.caption(reference_library_note(reference_samples))

    st.markdown("### LAB Hazard Classes")
    for profile in HAZARD_CLASS_PROFILES.values():
        render_reference_card(profile, reference_samples)

    st.markdown("### Why This App Avoids Overclaiming")
    st.markdown(
        """
        - Gil et al. showed that emerald-green coverings have a distinct vis-NIR signature, but they also stress that colour alone is not a good indicator of a specific pigment.
        - Vermeulen et al. showed that arsenical hazards extend beyond intact green covers to degraded products and transfer to neighbouring books.
        - Poison Book Project guidance indicates that emerald green can be friable on cloth, while current observations suggest lower hand-transfer risk for chrome yellow and chrome green.
        - Otero et al. reported Lab* coordinates from 76,23,71 to 85,30,89 for reconstructed 19th-century chrome yellows, so this build now uses literature-derived fallback anchors for the chrome-yellow-like class rather than generic swatches.
        - Bookcloth-specific use evidence is stronger than the colourimetry evidence for several classes. Emerald green and chrome green are historically well documented on cloth, but open cloth-specific LAB datasets remain sparse.
        - Chrome-green descriptions in the literature often refer to lead chromate mixed with Prussian blue, yet open mixture-ratio-to-LAB mappings for cloth were not located; this app therefore keeps chrome-green anchors explicitly proxy-based.
        - Turner documented lead and mercury in historical books more broadly, so this app keeps a conservative mercury watchlist without overstating cloth-specific certainty.
        - This repo does not ship a validation dataset for the LAB thresholds, so the `ΔE00` bands should be read as working defaults rather than universal decision boundaries.
        - The fading-aware allowance is intentionally capped and only applies when hue remains near the reference while chroma drops; it is a conservative heuristic, not a published ageing model.
        - Repeated measurements are not just displayed; the app downgrades colour bands when within-set spread crosses the active threshold.
        - The emerald-green-like, chrome-green-like, and mercury-red-like built-ins still rely on ISCC-NBS hue proxies; the more defensible workflow is a same-device local reference library measured on confirmed or institutionally trusted analogues.
        """
    )

    st.markdown("### Scenario SOP Library")
    for sop in scenario_sop_library():
        render_sop_card(sop, title=sop.title)

    st.markdown('### Sources')
    links = "".join(
        f'<li><a href="{url}" target="_blank">{label}</a></li>'
        for label, url in SOURCE_LINKS
    )
    st.markdown(f'<div class="source-list"><ul>{links}</ul></div>', unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_css()
    with st.sidebar:
        st.markdown("### Reference Library")
        st.caption(
            "Upload a same-device local reference library to replace the built-in fallback anchors. "
            "Fallback-only classes remain provisional because the built-ins are not same-device confirmed "
            "bookcloth measurements."
        )
        st.download_button(
            "Download reference template",
            data=example_reference_library_template().to_csv(index=False),
            file_name="hazardous_pigment_reference_library_template.csv",
            mime="text/csv",
            use_container_width=True,
        )
        reference_file = st.file_uploader(
            "Upload local reference library CSV",
            type="csv",
            key="reference_library_csv",
        )

    try:
        reference_samples = resolve_reference_samples(
            reference_file.getvalue() if reference_file is not None else None
        )
    except Exception as exc:
        st.error(f"Could not load the reference library: {exc}")
        reference_samples = get_default_reference_samples()

    render_hero()
    st.warning(
        "LAB screening only. The app assigns hazard colour classes to support safer handling and testing; it does not replace XRF, Raman, FTIR, or professional conservation judgement."
    )

    tabs = st.tabs(["Single Binding Screen", "Batch Triage", "Evidence & Handling"])
    with tabs[0]:
        render_single_binding_tab(reference_samples)
    with tabs[1]:
        render_batch_tab(reference_samples)
    with tabs[2]:
        render_reference_tab(reference_samples)


if __name__ == "__main__":
    main()
