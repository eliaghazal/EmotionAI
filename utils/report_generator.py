"""PDF session report generator using ReportLab."""

import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image as RLImage, PageTemplate,
    Paragraph, Spacer, Table, TableStyle, HRFlowable,
)

logger = logging.getLogger(__name__)

EMOTION_LABELS: List[str] = [
    "Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"
]
EMOTION_COLORS_HEX: Dict[str, str] = {
    "Angry": "#FF4444", "Disgust": "#9B59B6", "Fear": "#E67E22",
    "Happy": "#F1C40F", "Sad": "#3498DB", "Surprise": "#1ABC9C",
    "Neutral": "#95A5A6",
}
EVAL_JSON = Path(__file__).parent.parent / "models" / "saved" / "evaluation_results.json"


def _hex_to_rl(hex_color: str) -> colors.HexColor:
    return colors.HexColor(hex_color)


def _buf_from_fig(fig: plt.Figure) -> io.BytesIO:
    """Render matplotlib figure to an in-memory PNG buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf


def _distribution_figure(distribution: Dict[str, float]) -> io.BytesIO:
    labels = list(distribution.keys())
    vals = list(distribution.values())
    emo_colors = [EMOTION_COLORS_HEX.get(l, "#AAAAAA") for l in labels]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.barh(labels, vals, color=emo_colors, edgecolor="white")
    ax.set_xlabel("Percentage (%)")
    ax.set_title("Emotion Distribution")
    ax.set_xlim(0, max(vals) * 1.15 if vals else 100)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    return _buf_from_fig(fig)


def _timeline_figure(timeline: List[tuple]) -> io.BytesIO:
    """Simple timeline showing emotion over time."""
    if not timeline:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No timeline data", ha="center", va="center")
        return _buf_from_fig(fig)

    times, emotions = zip(*timeline)
    emo_idx = [EMOTION_LABELS.index(e) if e in EMOTION_LABELS else -1 for e in emotions]

    fig, ax = plt.subplots(figsize=(7, 2.5))
    sc = ax.scatter(times, emo_idx, c=[i / 6.0 for i in emo_idx],
                    cmap="tab10", s=15, alpha=0.7)
    ax.set_yticks(range(7))
    ax.set_yticklabels(EMOTION_LABELS, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title("Emotion Timeline")
    plt.tight_layout()
    return _buf_from_fig(fig)


def _person_emotion_figure(distribution: Dict[str, float], name: str) -> io.BytesIO:
    """Compact horizontal bar chart for one person's emotion distribution."""
    labels = [e for e in EMOTION_LABELS if distribution.get(e, 0) > 0]
    vals = [distribution.get(e, 0) for e in labels]
    emo_colors = [EMOTION_COLORS_HEX.get(e, "#AAAAAA") for e in labels]
    if not labels:
        labels = ["No data"]; vals = [100]; emo_colors = ["#AAAAAA"]

    fig, ax = plt.subplots(figsize=(4.5, max(1.5, len(labels) * 0.45)))
    ax.barh(labels, vals, color=emo_colors, edgecolor="white", height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f"{v:.0f}%", va="center", fontsize=7)
    ax.set_xlim(0, max(vals) * 1.25 if vals else 100)
    ax.set_xlabel("% of frames", fontsize=7)
    ax.tick_params(axis="both", labelsize=7)
    ax.set_title(name, fontsize=8, fontweight="bold")
    plt.tight_layout()
    return _buf_from_fig(fig)


def generate_session_pdf(
    output_path: str,
    session_stats: Dict,
    timeline: List[tuple],
    model_name: str = "Custom CNN",
    top_predictions: Optional[List[Dict]] = None,
    person_stats: Optional[Dict] = None,
    face_images_dir: Optional[str] = None,
) -> None:
    """Generate a PDF session report.

    Args:
        output_path: Destination file path (*.pdf).
        session_stats: Dict with keys: dominant_emotion, distribution_percent,
                       avg_confidence, total_faces_detected, duration_seconds.
        timeline: List of (relative_seconds, emotion_label) tuples.
        model_name: Name of the model used during the session.
        top_predictions: Optional list of top confident predictions with keys
                         face_crop (np.ndarray), emotion, confidence.
    """
    doc = BaseDocTemplate(output_path, pagesize=A4)
    frame = Frame(doc.leftMargin, doc.bottomMargin,
                  doc.width, doc.height, id="main")
    template = PageTemplate(id="main_page", frames=[frame])
    doc.addPageTemplates([template])

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title", parent=styles["Title"],
        textColor=_hex_to_rl("#2C3E50"), fontSize=22, spaceAfter=6
    )
    heading_style = ParagraphStyle(
        "Heading", parent=styles["Heading2"],
        textColor=_hex_to_rl("#3498DB"), spaceAfter=4
    )
    body_style = styles["Normal"]

    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph("EmotionAI Session Report", title_style))
    story.append(HRFlowable(width="100%", thickness=2, color=_hex_to_rl("#3498DB")))
    story.append(Spacer(1, 0.3 * cm))

    date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    duration = session_stats.get("duration_seconds", 0)
    duration_str = f"{int(duration // 60)}m {int(duration % 60)}s"
    total_faces = session_stats.get("total_faces_detected", 0)
    dom_emotion = session_stats.get("dominant_emotion", "N/A")

    meta_data = [
        ["Date", date_str, "Model Used", model_name],
        ["Duration", duration_str, "Total Faces Detected", str(total_faces)],
        ["Dominant Emotion", dom_emotion, "Avg Confidence",
         f"{session_stats.get('avg_confidence', 0) * 100:.1f}%"],
    ]
    meta_table = Table(meta_data, colWidths=[3.5 * cm, 5.5 * cm, 4 * cm, 4 * cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), _hex_to_rl("#EBF5FB")),
        ("BACKGROUND", (2, 0), (2, -1), _hex_to_rl("#EBF5FB")),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.5 * cm))

    # ── Distribution ────────────────────────────────────────────────────────
    story.append(Paragraph("Emotion Distribution", heading_style))
    dist = session_stats.get("distribution_percent", {})
    if dist:
        buf = _distribution_figure(dist)
        story.append(RLImage(buf, width=10 * cm, height=8 * cm))
    story.append(Spacer(1, 0.4 * cm))

    # ── Timeline ────────────────────────────────────────────────────────────
    story.append(Paragraph("Emotion Timeline", heading_style))
    tl_buf = _timeline_figure(timeline)
    story.append(RLImage(tl_buf, width=14 * cm, height=5 * cm))
    story.append(Spacer(1, 0.4 * cm))

    # ── Model performance ───────────────────────────────────────────────────
    if EVAL_JSON.exists():
        story.append(Paragraph("Model Performance Summary", heading_style))
        try:
            with open(EVAL_JSON) as f:
                eval_data = json.load(f)
            rows = [["Model", "Accuracy", "Macro F1"]]
            for mname, mdata in eval_data.items():
                rows.append([
                    mname,
                    f"{mdata.get('accuracy', 0) * 100:.2f}%",
                    f"{mdata.get('macro_f1', 0):.4f}",
                ])
            perf_table = Table(rows, colWidths=[8 * cm, 4 * cm, 4 * cm])
            perf_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), _hex_to_rl("#2C3E50")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, _hex_to_rl("#EBF5FB")]),
            ]))
            story.append(perf_table)
        except Exception as exc:
            logger.warning("Could not load evaluation results: %s", exc)

    # ── Per-person breakdown ─────────────────────────────────────────────────
    if person_stats:
        story.append(Paragraph("Per-Person Emotion Breakdown", heading_style))
        face_dir = Path(face_images_dir) if face_images_dir else None

        for person_name, pdata in sorted(person_stats.items()):
            if person_name == "Unknown" and len(person_stats) > 1:
                continue   # skip the catch-all bucket when named people exist
            dominant = pdata.get("dominant", "—")
            total_f = pdata.get("total_frames", 0)
            dist = pdata.get("distribution", {})

            # Sub-heading
            story.append(Paragraph(
                f"{person_name}  —  Dominant: {dominant}  |  Frames: {total_f}",
                ParagraphStyle("person_head", parent=styles["Normal"],
                               fontName="Helvetica-Bold", fontSize=10,
                               textColor=_hex_to_rl("#2C3E50"), spaceAfter=3)
            ))

            # Face photo (left) + emotion bars (right) side by side
            row_items = []

            face_img_path = face_dir / f"{person_name}.jpg" if face_dir else None
            if face_img_path and face_img_path.exists():
                try:
                    row_items.append(RLImage(str(face_img_path),
                                             width=3.2 * cm, height=3.2 * cm))
                except Exception:
                    row_items.append(Spacer(3.2 * cm, 3.2 * cm))
            else:
                row_items.append(Spacer(3.2 * cm, 0.1 * cm))

            chart_buf = _person_emotion_figure(dist, person_name)
            row_items.append(RLImage(chart_buf, width=11 * cm, height=3.2 * cm))

            person_table = Table([row_items], colWidths=[3.5 * cm, 12 * cm])
            person_table.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(person_table)
            story.append(Spacer(1, 0.3 * cm))

    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.grey))
    story.append(Paragraph(
        "Generated by EmotionAI — Face Detection & Emotion Recognition System",
        ParagraphStyle("footer", fontSize=8, textColor=colors.grey)
    ))

    doc.build(story)
    logger.info("Session PDF saved to %s", output_path)
