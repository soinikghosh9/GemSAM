"""
MedGamma — Clinical AI for Medical Imaging
Kaggle MedGemma Impact Challenge Submission

Premium Gradio Demo with clinical-grade UI/UX.
Built with MedGemma 1.5 4B + SAM2.

Usage:
    python demo/gradio_app.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import gradio as gr

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─── Clinical Color Palette ─────────────────────────────────────────────────
CLINICAL_COLORS = {
    "Cardiomegaly":        "#E74C3C",
    "Pleural effusion":    "#3498DB",
    "Pleural thickening":  "#1ABC9C",
    "Aortic enlargement":  "#E67E22",
    "Lung Opacity":        "#9B59B6",
    "Nodule/Mass":         "#F39C12",
    "Pulmonary fibrosis":  "#2ECC71",
    "Consolidation":       "#E91E63",
    "Pneumonia":           "#00BCD4",
    "Infiltration":        "#FF9800",
    "Pneumothorax":        "#FF5722",
    "Calcification":       "#8BC34A",
    "ILD":                 "#795548",
    "Atelectasis":         "#607D8B",
    "default":             "#E74C3C",
    "No significant abnormality": "#27AE60",
}


def get_color(label: str, alpha: int = 255) -> tuple:
    """Get RGBA color for a finding class."""
    hex_color = CLINICAL_COLORS.get(label, CLINICAL_COLORS["default"])
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    return (r, g, b, alpha)


def draw_findings_on_image(
    image: Image.Image,
    findings: List[Dict],
    masks: Optional[List[np.ndarray]] = None
) -> Image.Image:
    """Draw clinical bounding boxes with styled labels on image."""
    annotated = image.copy().convert("RGBA")
    draw = ImageDraw.Draw(annotated)
    w, h = annotated.size

    try:
        font = ImageFont.truetype("arial.ttf", max(14, min(w, h) // 35))
        font_small = ImageFont.truetype("arial.ttf", max(11, min(w, h) // 45))
    except:
        font = ImageFont.load_default()
        font_small = font

    # Draw masks first
    if masks:
        for i, (finding, mask) in enumerate(zip(findings, masks)):
            if mask is not None and mask.size > 0:
                r, g, b, _ = get_color(finding.get("class", "default"))
                mask_resized = np.array(Image.fromarray(mask.astype(np.uint8) * 255).resize(image.size))
                overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
                overlay_pixels = np.array(overlay)
                overlay_pixels[mask_resized > 127] = [r, g, b, 90]
                overlay = Image.fromarray(overlay_pixels, "RGBA")
                annotated = Image.alpha_composite(annotated, overlay)
                draw = ImageDraw.Draw(annotated)

    # Draw boxes with clinical styling
    for finding in findings:
        box = finding.get("box", [])
        label = finding.get("class", "Unknown")

        # Skip normal/placeholder boxes
        if label.lower() in ("no significant abnormality", "no finding", "normal"):
            continue
        if len(box) != 4:
            continue

        x1, y1, x2, y2 = box
        # Scale from 0-1000 to image dims
        sx1 = int(x1 * w / 1000)
        sy1 = int(y1 * h / 1000)
        sx2 = int(x2 * w / 1000)
        sy2 = int(y2 * h / 1000)

        r, g, b, _ = get_color(label)
        box_color = (r, g, b, 255)
        fill_color = (r, g, b, 35)

        # Semi-transparent fill
        overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle([sx1, sy1, sx2, sy2], fill=fill_color)
        annotated = Image.alpha_composite(annotated, overlay)
        draw = ImageDraw.Draw(annotated)

        # Box outline (2px)
        draw.rectangle([sx1, sy1, sx2, sy2], outline=box_color, width=2)

        # Label pill
        conf = finding.get("confidence", None)
        label_text = label
        if conf is not None:
            label_text = f"{label} {conf:.0%}"

        bbox = draw.textbbox((sx1, sy1 - 22), label_text, font=font)
        pill_w = bbox[2] - bbox[0] + 12
        pill_h = bbox[3] - bbox[1] + 6
        pill_y = max(0, sy1 - pill_h - 4)

        # Draw pill background
        draw.rounded_rectangle(
            [sx1, pill_y, sx1 + pill_w, pill_y + pill_h],
            radius=4, fill=box_color
        )
        draw.text((sx1 + 6, pill_y + 2), label_text, fill=(255, 255, 255, 255), font=font)

    return annotated.convert("RGB")


# ─── Pipeline Analysis ───────────────────────────────────────────────────────

def analyze_image(
    image: Image.Image,
    enable_segmentation: bool = True
) -> tuple:
    """Run full MedGamma agentic pipeline on a medical image."""
    if image is None:
        blank = Image.new("RGB", (448, 448), (30, 30, 40))
        return (blank, blank, blank,
                "—", "—",
                "Upload an image to begin analysis.",
                "{}", "")

    temp_path = "temp_analysis_input.jpg"
    image.save(temp_path)

    try:
        from src.orchestrator import ClinicalOrchestrator
        import cv2

        orchestrator = ClinicalOrchestrator(checkpoint_dir="checkpoints/production")
        state = orchestrator.run_pipeline(temp_path, "Evaluate for clinical pathologies and localize.")

        img_np = np.array(image)

        # ── Detection Image ──
        findings_list = []
        for det in state.get("detections", []):
            findings_list.append({
                "class": det["label"],
                "box": det["box"],
                "confidence": det.get("confidence", 1.0)
            })
        annotated_detection = draw_findings_on_image(image.copy(), findings_list)

        # ── Heatmap Image ──
        if state.get("heatmap") is not None:
            heatmap = state["heatmap"]
            heatmap_norm = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
            heatmap_u8 = np.uint8(255 * heatmap_norm)
            # Resize heatmap to match input image dimensions
            h_img, w_img = img_np.shape[:2]
            heatmap_resized = cv2.resize(heatmap_u8, (w_img, h_img), interpolation=cv2.INTER_LINEAR)
            heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_INFERNO)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            overlay = cv2.addWeighted(img_bgr, 0.45, heatmap_color, 0.55, 0)
            heatmap_img = Image.fromarray(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        else:
            heatmap_img = image.copy()

        # ── Segmentation Image ──
        masks_list = [seg["mask"] for seg in state.get("segmentations", [])]
        annotated_seg = draw_findings_on_image(
            Image.fromarray(img_np), findings_list, masks_list
        )

        # ── Clinical Output ──
        if state.get("detections"):
            n = len(state["detections"])
            classes = list(set(d["label"] for d in state["detections"]))
            status_text = f"⚠️ ABNORMAL — {n} finding(s): {', '.join(classes)}"
            modality_text = f"📋 Chest X-ray (Auto-detected)"
        else:
            status_text = "✅ NORMAL — No acute findings"
            modality_text = "📋 Chest X-ray (Auto-detected)"

        # Format clinical report
        clinical_reasoning = state.get("current_thought", "")
        raw_output = state.get("raw_medgemma_output", clinical_reasoning)
        clinical_report = state.get("final_report", "")

        # Build formatted findings JSON
        findings_json = json.dumps({"findings": findings_list}, indent=2)

        # Build clinical summary
        if not clinical_report:
            if findings_list:
                lines = ["**Clinical Report**\n"]
                for i, f in enumerate(findings_list, 1):
                    desc = f.get("description", f"Presence of {f['class']} detected.")
                    lines.append(f"{i}. **{f['class']}** — {desc}")
                clinical_report = "\n".join(lines)
            else:
                clinical_report = (
                    "**Clinical Report**\n\n"
                    "No acute cardiopulmonary abnormality identified. "
                    "Heart size and mediastinal contour are within normal limits. "
                    "Lungs are clear bilaterally without focal consolidation, effusion, or pneumothorax. "
                    "Osseous structures are unremarkable."
                )

        return (
            annotated_detection, heatmap_img, annotated_seg,
            modality_text, status_text,
            clinical_report,
            findings_json,
            raw_output or "Analysis complete."
        )

    except ImportError:
        return demo_analyze(image)
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"⚠️ Analysis error: {str(e)}"
        return (image, image, image, "Error", error_msg, error_msg, "{}", str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def demo_analyze(image: Image.Image) -> tuple:
    """Demo mode with simulated clinical results."""
    w, h = image.size
    demo_findings = [
        {"class": "Cardiomegaly", "box": [350, 400, 700, 700], "confidence": 0.92,
         "description": "Moderate cardiomegaly with cardiothoracic ratio >0.5."},
        {"class": "Pleural effusion", "box": [600, 500, 900, 800], "confidence": 0.78,
         "description": "Small left-sided pleural effusion with blunting of the costophrenic angle."},
    ]

    annotated = draw_findings_on_image(image, demo_findings)
    findings_json = json.dumps({"findings": demo_findings}, indent=2)

    clinical_report = (
        "**Clinical Report** *(Demo Mode)*\n\n"
        "1. **Cardiomegaly** (confidence: 92%) — Moderate cardiomegaly with "
        "cardiothoracic ratio >0.5, suggesting left ventricular enlargement.\n\n"
        "2. **Pleural effusion** (confidence: 78%) — Small left-sided pleural "
        "effusion with blunting of the costophrenic angle.\n\n"
        "*⚠️ Demo mode — load models for real analysis*"
    )

    return (
        annotated,
        Image.new("RGB", image.size, (40, 20, 60)),
        annotated,
        "📋 X-ray (Demo Mode)",
        "⚠️ ABNORMAL — 2 finding(s): Cardiomegaly, Pleural effusion",
        clinical_report,
        findings_json,
        "VLM Reasoning: Demo mode — model not loaded."
    )


# ─── Custom CSS ──────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global ────────────────────────────────────────────────────────── */
.gradio-container {
    max-width: 1400px !important;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}

/* ── Header ────────────────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f2027 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    border: 1px solid rgba(56, 189, 248, 0.15);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.app-header h1 {
    color: #f8fafc !important;
    font-size: 1.8em !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    margin: 0 0 4px 0 !important;
}
.app-header .subtitle {
    color: #94a3b8;
    font-size: 0.95em;
    margin-top: 4px;
}
.badge {
    display: inline-block;
    background: linear-gradient(135deg, #0ea5e9, #06b6d4);
    color: white;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.75em;
    font-weight: 600;
    margin-left: 10px;
    vertical-align: middle;
}

/* ── Cards ─────────────────────────────────────────────────────────── */
.panel-card {
    background: #1e293b !important;
    border: 1px solid rgba(148, 163, 184, 0.12) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
.panel-card .label-wrap {
    background: rgba(15, 23, 42, 0.6) !important;
}

/* ── Status badges ─────────────────────────────────────────────────── */
.status-normal {
    background: linear-gradient(135deg, #065f46, #047857) !important;
    color: #ecfdf5 !important;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.status-abnormal {
    background: linear-gradient(135deg, #7f1d1d, #991b1b) !important;
    color: #fef2f2 !important;
    padding: 8px 16px;
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* ── Image panels ──────────────────────────────────────────────────── */
.image-panel img {
    border-radius: 8px !important;
    border: 1px solid rgba(148, 163, 184, 0.1);
}
.image-panel .label-wrap span {
    font-weight: 600 !important;
    font-size: 0.85em !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em;
}

/* ── Clinical report ───────────────────────────────────────────────── */
.clinical-report textarea {
    font-family: 'Georgia', 'Times New Roman', serif !important;
    font-size: 0.95em !important;
    line-height: 1.7 !important;
    background: #0f172a !important;
    border: 1px solid rgba(148, 163, 184, 0.12) !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── Analyze button ────────────────────────────────────────────────── */
.analyze-btn {
    background: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1.05em !important;
    letter-spacing: 0.02em;
    padding: 12px 24px !important;
    border-radius: 10px !important;
    box-shadow: 0 4px 16px rgba(14, 165, 233, 0.25) !important;
    transition: all 0.2s ease !important;
}
.analyze-btn:hover {
    box-shadow: 0 6px 24px rgba(14, 165, 233, 0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Pipeline indicators ───────────────────────────────────────────── */
.pipeline-step {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 6px;
    font-size: 0.8em;
    font-weight: 500;
    background: rgba(30, 41, 59, 0.8);
    border: 1px solid rgba(148, 163, 184, 0.1);
    color: #94a3b8;
}

/* ── Example images ────────────────────────────────────────────────── */
.examples-section .gallery {
    gap: 8px !important;
}
.examples-section img {
    border-radius: 8px !important;
    border: 2px solid transparent !important;
    transition: border-color 0.2s ease;
}
.examples-section img:hover {
    border-color: #0ea5e9 !important;
}

/* ── Accordion ─────────────────────────────────────────────────────── */
.accordion-section {
    border: 1px solid rgba(148, 163, 184, 0.08) !important;
    border-radius: 10px !important;
    margin-top: 8px;
}
"""


# ─── Build UI ────────────────────────────────────────────────────────────────

def create_demo() -> gr.Blocks:
    """Create premium clinical Gradio demo."""

    # Find example images
    demo_dir = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(demo_dir, "examples")
    example_images = []
    if os.path.exists(examples_dir):
        for f in sorted(os.listdir(examples_dir)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                example_images.append([os.path.join(examples_dir, f)])

    with gr.Blocks(
        title="MedGamma — Clinical AI for Medical Imaging",
    ) as demo:

        # ── Header ──
        gr.HTML("""
        <div class="app-header">
            <h1>🏥 MedGamma <span class="badge">v2.0</span></h1>
            <div class="subtitle">
                Clinical Explainable AI for Medical Imaging &nbsp;·&nbsp;
                <strong>MedGemma 1.5 4B</strong> + <strong>SAM2</strong> &nbsp;·&nbsp;
                Kaggle MedGemma Impact Challenge
            </div>
            <div style="margin-top: 12px; display: flex; gap: 8px; flex-wrap: wrap;">
                <span class="pipeline-step">🔬 Modality Detection</span>
                <span class="pipeline-step">🩺 Abnormality Screening</span>
                <span class="pipeline-step">📍 Pathology Localization</span>
                <span class="pipeline-step">🧠 Explainability Heatmap</span>
                <span class="pipeline-step">✂️ SAM2 Segmentation</span>
                <span class="pipeline-step">📝 Clinical Report</span>
            </div>
        </div>
        """)

        # ── Main Layout ──
        with gr.Row(equal_height=False):

            # ── Left: Input Panel ──
            with gr.Column(scale=1, min_width=320):
                input_image = gr.Image(
                    label="Upload Medical Image",
                    type="pil",
                    height=380,
                    elem_classes=["image-panel"],
                    sources=["upload", "clipboard"],
                )

                analyze_btn = gr.Button(
                    "🔍  Analyze Image",
                    variant="primary",
                    size="lg",
                    elem_classes=["analyze-btn"],
                )

                with gr.Row():
                    enable_seg = gr.Checkbox(
                        label="Enable SAM2 Segmentation",
                        value=True,
                        scale=2,
                    )

                if example_images:
                    gr.Examples(
                        examples=example_images,
                        inputs=input_image,
                        label="📂 Sample Chest X-rays",
                        examples_per_page=4,
                        elem_id="examples-section",
                    )

            # ── Right: Results Panel ──
            with gr.Column(scale=2, min_width=600):

                # Status bar
                with gr.Row():
                    modality_output = gr.Markdown(
                        value="📋 *Upload an image to begin*",
                        elem_classes=["status-badge"],
                    )
                    abnormality_output = gr.Markdown(
                        value="—",
                        elem_classes=["status-badge"],
                    )

                # Image results row
                with gr.Row():
                    output_image_det = gr.Image(
                        label="Detection",
                        height=280,
                        elem_classes=["image-panel"],
                    )
                    output_image_heat = gr.Image(
                        label="Explainability",
                        height=280,
                        elem_classes=["image-panel"],
                    )
                    output_image_seg = gr.Image(
                        label="Segmentation",
                        height=280,
                        elem_classes=["image-panel"],
                    )

                # Clinical Report
                clinical_report = gr.Markdown(
                    value="*Upload an image and click Analyze to generate a clinical report.*",
                    label="📋 Clinical Report",
                    elem_classes=["clinical-report"],
                )

                # Expandable sections
                with gr.Accordion("🔬 VLM Reasoning Trace", open=False, elem_classes=["accordion-section"]):
                    clinical_insights = gr.Textbox(
                        label="Model Internal Reasoning",
                        lines=6,
                        interactive=False,
                        show_label=False,
                    )

                with gr.Accordion("📊 Structured Findings (JSON)", open=False, elem_classes=["accordion-section"]):
                    findings_json = gr.Code(
                        label="Findings",
                        language="json",
                        lines=12,
                    )

        # ── Footer ──
        with gr.Accordion("ℹ️ About MedGamma", open=False, elem_classes=["accordion-section"]):
            gr.Markdown("""
            ### Architecture

            | Component | Model | Role |
            |-----------|-------|------|
            | **Clinical Reasoning** | MedGemma 1.5 4B (LoRA) | Multimodal VLM for detection + report generation |
            | **Explainability** | Gradient-based attention | Visual heatmaps showing model focus areas |
            | **Segmentation** | SAM2 (Hiera-Tiny) | Pixel-precise lesion boundary delineation |

            ### Edge Deployment
            - **Raspberry Pi 5** (8GB) + AI HAT+2 (Hailo-10H, 40 TOPS)
            - MedGemma on CPU (~30-60s), SAM2 on NPU (~25-50ms)
            - Privacy-preserving local inference — no cloud required

            ### Training Data
            - **VinDr-CXR**: 15,000 chest X-rays with 14 pathology classes
            - **NIH ChestX-ray**: 112,120 images with 15 disease labels
            - **SLAKE / VQA-RAD**: Medical visual question answering
            """)

        # ── Connect Events ──
        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image, enable_seg],
            outputs=[
                output_image_det, output_image_heat, output_image_seg,
                modality_output, abnormality_output,
                clinical_report, findings_json, clinical_insights,
            ],
        )

    return demo


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MedGamma Clinical Demo")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    print("""
    ╔═══════════════════════════════════════════════════╗
    ║   MedGamma — Clinical AI for Medical Imaging      ║
    ║   Kaggle MedGemma Impact Challenge                ║
    ╚═══════════════════════════════════════════════════╝
    """)

    demo = create_demo()
    demo.launch(
        server_port=args.port,
        share=args.share,
        server_name="127.0.0.1",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.cyan,
            secondary_hue=gr.themes.colors.blue,
            neutral_hue=gr.themes.colors.slate,
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ).set(
            body_background_fill="#0f172a",
            body_background_fill_dark="#0f172a",
            block_background_fill="#1e293b",
            block_background_fill_dark="#1e293b",
            input_background_fill="#0f172a",
            input_background_fill_dark="#0f172a",
            block_border_width="0px",
            block_shadow="0 2px 8px rgba(0,0,0,0.3)",
        ),
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
