from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

from app.core.config import settings


DISCLAIMER = (
    "Research prototype only. This report supports demonstration and academic "
    "evaluation and is not a clinically validated standalone diagnosis."
)


class ReportService:
    REGION_MAP = {
        "FP1": "Frontal",
        "FP2": "Frontal",
        "F3": "Frontal",
        "F4": "Frontal",
        "F7": "Frontal",
        "F8": "Frontal",
        "FZ": "Frontal",
        "C3": "Central",
        "C4": "Central",
        "CZ": "Central",
        "P3": "Parietal",
        "P4": "Parietal",
        "PZ": "Parietal",
        "T3": "Temporal",
        "T4": "Temporal",
        "T5": "Temporal",
        "T6": "Temporal",
        "O1": "Occipital",
        "O2": "Occipital",
        # bipolar channels
        "C3-P3": "Central-Parietal",
        "C4-P4": "Central-Parietal",
        "F3-C3": "Frontal-Central",
        "F4-C4": "Frontal-Central",
        "FP1-F3": "Frontal",
        "FP2-F4": "Frontal",
        "F7-T3": "Frontal-Temporal",
        "F8-T4": "Frontal-Temporal",
        "T3-T5": "Temporal",
        "T4-T6": "Temporal",
        "T5-O1": "Temporal-Occipital",
        "T6-O2": "Temporal-Occipital",
        "FZ-CZ": "Frontal-Central",
        "CZ-PZ": "Central-Parietal",
    }

    @staticmethod
    def build_report_data(
        patient_code: str,
        original_filename: str,
        inference_summary: Dict[str, Any],
        explain_summary: str,
        recording_date: Optional[str] = None,
        duration_minutes: Optional[float] = None,
    ) -> Dict[str, Any]:
        probabilities = inference_summary.get("probabilities", []) or []
        top_channels_raw = inference_summary.get("top_channels", []) or []
        seizure_ranges = inference_summary.get("seizure_ranges", []) or []
        threshold = float(inference_summary.get("threshold", 0.5))

        prediction = inference_summary.get("overall_decision", "Unknown")
        confidence = float(inference_summary.get("max_probability", 0.0))
        mean_probability = float(inference_summary.get("mean_probability", 0.0))
        seizure_duration_min = ReportService._estimate_seizure_duration_minutes(
            seizure_ranges=seizure_ranges,
            inference_summary=inference_summary,
        )

        confidence_label = ReportService._confidence_label(confidence)
        top_channels = ReportService._normalize_top_channels(top_channels_raw)
        dominant_region = ReportService._dominant_region(top_channels)
        findings = ReportService._build_clinical_findings(
            prediction=prediction,
            confidence=confidence,
            confidence_label=confidence_label,
            dominant_region=dominant_region,
            seizure_duration_min=seizure_duration_min,
            explain_summary=explain_summary,
        )
        recommendations = ReportService._build_recommendations(
            prediction=prediction,
            confidence=confidence,
            seizure_duration_min=seizure_duration_min,
            dominant_region=dominant_region,
        )

        return {
            "patient_id": patient_code,
            "eeg_file": original_filename,
            "recording_date": recording_date or "N/A",
            "duration_minutes": duration_minutes,
            "prediction": prediction,
            "confidence_score": confidence,
            "confidence_label": confidence_label,
            "threshold": threshold,
            "mean_probability": mean_probability,
            "num_windows": int(inference_summary.get("num_windows", 0)),
            "num_seizure_windows": int(inference_summary.get("num_seizure_windows", 0)),
            "estimated_seizure_duration_minutes": seizure_duration_min,
            "probability_timeline": probabilities,
            "seizure_ranges": seizure_ranges,
            "top_channels": top_channels,
            "dominant_region": dominant_region,
            "clinical_findings": findings,
            "recommendations": recommendations,
            "explanation_summary": explain_summary,
            "disclaimer": DISCLAIMER,
        }

    @staticmethod
    def generate_pdf(
        patient_code: str,
        original_filename: str,
        inference_summary: Dict[str, Any],
        explain_summary: str,
        report_name: str,
        recording_date: Optional[str] = None,
        duration_minutes: Optional[float] = None,
    ) -> Path:
        report = ReportService.build_report_data(
            patient_code=patient_code,
            original_filename=original_filename,
            inference_summary=inference_summary,
            explain_summary=explain_summary,
            recording_date=recording_date,
            duration_minutes=duration_minutes,
        )

        out_path = settings.reports_dir / report_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        c = canvas.Canvas(str(out_path), pagesize=A4)
        width, height = A4

        margin_x = 18 * mm
        top_margin = 18 * mm
        bottom_margin = 18 * mm
        usable_width = width - (2 * margin_x)
        y = height - top_margin

        def ensure_space(current_y: float, needed: float) -> float:
            if current_y - needed < bottom_margin:
                c.showPage()
                return height - top_margin
            return current_y

        y = ReportService._draw_header(c, margin_x, y, usable_width, report)

        y = ensure_space(y, 90)
        y -= 8
        y = ReportService._draw_section_title(c, margin_x, y, "Patient & Recording Details")
        y = ReportService._draw_key_value_block(
            c,
            margin_x,
            y,
            usable_width,
            [
                ("Patient ID", str(report["patient_id"])),
                ("EEG File", str(report["eeg_file"])),
                ("Recording Date", str(report["recording_date"])),
                (
                    "Duration",
                    f'{report["duration_minutes"]:.1f} minutes'
                    if isinstance(report["duration_minutes"], (int, float))
                    else "N/A",
                ),
            ],
        )

        y = ensure_space(y, 130)
        y -= 4
        y = ReportService._draw_section_title(c, margin_x, y, "Detection Summary")
        y = ReportService._draw_key_value_block(
            c,
            margin_x,
            y,
            usable_width,
            [
                ("Prediction", str(report["prediction"])),
                (
                    "Confidence",
                    f'{report["confidence_score"] * 100:.1f}% ({report["confidence_label"]})',
                ),
                ("Threshold", f'{report["threshold"]:.2f}'),
                ("Mean Probability", f'{report["mean_probability"] * 100:.1f}%'),
                ("Windows Analysed", str(report["num_windows"])),
                ("Seizure Windows", str(report["num_seizure_windows"])),
                (
                    "Estimated Seizure Duration",
                    f'{report["estimated_seizure_duration_minutes"]:.1f} minutes',
                ),
                (
                    "Dominant Region",
                    report["dominant_region"] or "Not clearly localized",
                ),
            ],
        )

        y = ensure_space(y, 90)
        y -= 4
        y = ReportService._draw_section_title(c, margin_x, y, "Clinical Findings")
        y = ReportService._draw_paragraph(
            c,
            report["clinical_findings"],
            margin_x,
            y,
            usable_width,
            line_height=14,
        )

        # charts
        y = ensure_space(y, 220)
        y -= 6
        y = ReportService._draw_section_title(c, margin_x, y, "Seizure Probability Timeline")
        y = ReportService._draw_probability_chart(
            c,
            report["probability_timeline"],
            margin_x,
            y,
            usable_width,
            85 * mm,
            threshold=report["threshold"],
            duration_minutes=report["duration_minutes"],
        )

        y = ensure_space(y, 220)
        y -= 6
        y = ReportService._draw_section_title(c, margin_x, y, "Top Channel Contribution")
        y = ReportService._draw_channel_bar_chart(
            c,
            report["top_channels"],
            margin_x,
            y,
            usable_width,
            75 * mm,
        )

        y = ensure_space(y, 100)
        y -= 6
        y = ReportService._draw_section_title(c, margin_x, y, "Top Channel Details")
        if report["top_channels"]:
            channel_lines = [
                f'{item["channel"]}: {item["score"] * 100:.1f}% ({item["region"]})'
                for item in report["top_channels"][:6]
            ]
            y = ReportService._draw_bullets(c, channel_lines, margin_x, y, usable_width)
        else:
            y = ReportService._draw_paragraph(
                c,
                "No channel contribution information available.",
                margin_x,
                y,
                usable_width,
            )

        y = ensure_space(y, 120)
        y -= 4
        y = ReportService._draw_section_title(c, margin_x, y, "Recommendations")
        y = ReportService._draw_bullets(
            c,
            report["recommendations"],
            margin_x,
            y,
            usable_width,
        )

        y = ensure_space(y, 80)
        y -= 4
        y = ReportService._draw_section_title(c, margin_x, y, "Explainability Summary")
        y = ReportService._draw_paragraph(
            c,
            report["explanation_summary"],
            margin_x,
            y,
            usable_width,
            line_height=14,
        )

        y = ensure_space(y, 30)
        y -= 10
        c.setStrokeColor(colors.HexColor("#334155"))
        c.line(margin_x, y, width - margin_x, y)
        y -= 12

        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#94a3b8"))
        for line in ReportService._wrap_text(DISCLAIMER, "Helvetica", 8, usable_width):
            c.drawString(margin_x, y, line)
            y -= 10

        c.save()
        return out_path
    
    @staticmethod
    def generate_csv(
        patient_code: str,
        original_filename: str,
        inference_summary: Dict[str, Any],
        explain_summary: str,
        report_name: str,
        recording_date: Optional[str] = None,
        duration_minutes: Optional[float] = None,
    ) -> Path:
        report = ReportService.build_report_data(
            patient_code=patient_code,
            original_filename=original_filename,
            inference_summary=inference_summary,
            explain_summary=explain_summary,
            recording_date=recording_date,
            duration_minutes=duration_minutes,
        )

        out_path = settings.reports_dir / report_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["field", "value"])
            writer.writerow(["patient_id", report["patient_id"]])
            writer.writerow(["eeg_file", report["eeg_file"]])
            writer.writerow(["recording_date", report["recording_date"]])
            writer.writerow(["duration_minutes", report["duration_minutes"]])
            writer.writerow(["prediction", report["prediction"]])
            writer.writerow(["confidence_score", report["confidence_score"]])
            writer.writerow(["confidence_label", report["confidence_label"]])
            writer.writerow(["threshold", report["threshold"]])
            writer.writerow(["mean_probability", report["mean_probability"]])
            writer.writerow(["num_windows", report["num_windows"]])
            writer.writerow(["num_seizure_windows", report["num_seizure_windows"]])
            writer.writerow(
                ["estimated_seizure_duration_minutes", report["estimated_seizure_duration_minutes"]]
            )
            writer.writerow(["dominant_region", report["dominant_region"]])
            writer.writerow(["clinical_findings", report["clinical_findings"]])
            writer.writerow(["explanation_summary", report["explanation_summary"]])
            writer.writerow(["recommendations", " | ".join(report["recommendations"])])
            writer.writerow(["disclaimer", report["disclaimer"]])

        return out_path

    @staticmethod
    def generate_json(
        patient_code: str,
        original_filename: str,
        inference_summary: Dict[str, Any],
        explain_summary: str,
        report_name: str,
        recording_date: Optional[str] = None,
        duration_minutes: Optional[float] = None,
    ) -> Path:
        report = ReportService.build_report_data(
            patient_code=patient_code,
            original_filename=original_filename,
            inference_summary=inference_summary,
            explain_summary=explain_summary,
            recording_date=recording_date,
            duration_minutes=duration_minutes,
        )

        out_path = settings.reports_dir / report_name
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return out_path

    @staticmethod
    def _confidence_label(confidence: float) -> str:
        if confidence >= 0.90:
            return "High confidence"
        if confidence >= 0.70:
            return "Moderate confidence"
        return "Low confidence"

    @staticmethod
    def _normalize_top_channels(top_channels_raw: Any) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []

        for item in top_channels_raw:
            if isinstance(item, dict):
                channel = str(item.get("channel", "")).upper()
                score = float(item.get("score", 0.0))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                channel = str(item[0]).upper()
                score = float(item[1])
            else:
                continue

            normalized.append(
                {
                    "channel": channel,
                    "score": score,
                    "region": ReportService._resolve_region(channel),
                }
            )

        normalized.sort(key=lambda x: x["score"], reverse=True)
        return normalized

    @staticmethod
    def _resolve_region(channel: str) -> str:
        if channel in ReportService.REGION_MAP:
            return ReportService.REGION_MAP[channel]

        if "-" in channel:
            parts = [p.strip().upper() for p in channel.split("-") if p.strip()]
            part_regions = []
            for part in parts:
                if part in ReportService.REGION_MAP:
                    region = ReportService.REGION_MAP[part]
                    if region not in part_regions:
                        part_regions.append(region)
            if part_regions:
                return "-".join(part_regions)

        return "Unknown"

    @staticmethod
    def _dominant_region(top_channels: List[Dict[str, Any]]) -> Optional[str]:
        if not top_channels:
            return None

        region_scores: Dict[str, float] = {}
        for item in top_channels[:5]:
            region = item["region"]
            region_scores[region] = region_scores.get(region, 0.0) + float(item["score"])

        if not region_scores:
            return None

        return max(region_scores.items(), key=lambda x: x[1])[0]

    @staticmethod
    def _estimate_seizure_duration_minutes(
        seizure_ranges: List[Any],
        inference_summary: Dict[str, Any],
    ) -> float:
        if seizure_ranges:
            total = 0.0
            for rng in seizure_ranges:
                if isinstance(rng, dict):
                    start_min = float(rng.get("start_min", 0.0))
                    end_min = float(rng.get("end_min", start_min))
                    total += max(0.0, end_min - start_min)
                elif isinstance(rng, (list, tuple)) and len(rng) >= 2:
                    start_min = float(rng[0])
                    end_min = float(rng[1])
                    total += max(0.0, end_min - start_min)
            if total > 0:
                return round(total, 2)

        num_seizure_windows = int(inference_summary.get("num_seizure_windows", 0))
        win_sec = float(inference_summary.get("window_sec", 10))
        step_sec = float(inference_summary.get("step_sec", win_sec))
        duration_min = (num_seizure_windows * step_sec) / 60.0
        return round(duration_min, 2)

    @staticmethod
    def _build_clinical_findings(
        prediction: str,
        confidence: float,
        confidence_label: str,
        dominant_region: Optional[str],
        seizure_duration_min: float,
        explain_summary: str,
    ) -> str:
        prediction_lower = prediction.lower()
        review_needed = "review needed" in prediction_lower
        seizure_detected = "seizure" in prediction_lower and (
            "detected" in prediction_lower or "activity" in prediction_lower
        )

        if seizure_detected and not review_needed:
            region_text = (
                f" The strongest explainability signals were observed in the {dominant_region.lower()} region."
                if dominant_region
                else ""
            )
            duration_text = (
                f" Estimated abnormal activity duration was approximately {seizure_duration_min:.1f} minutes."
                if seizure_duration_min > 0
                else ""
            )
            return (
                f"The analysis detected seizure-like activity with {confidence_label.lower()} "
                f"({confidence * 100:.1f}%).{duration_text}{region_text} "
                f"{explain_summary}"
            )

        if review_needed:
            region_text = (
                f" The strongest explainability signals were observed in the {dominant_region.lower()} region."
                if dominant_region
                else ""
            )
            duration_text = (
                f" Estimated flagged activity duration was approximately {seizure_duration_min:.1f} minutes."
                if seizure_duration_min > 0
                else ""
            )
            return (
                f"The analysis flagged possible seizure-like activity for review with {confidence_label.lower()} "
                f"({confidence * 100:.1f}%).{duration_text}{region_text} "
                f"{explain_summary}"
            )

        return (
            f"The analysis did not identify seizure activity above the decision threshold. "
            f"Model confidence for seizure presence remained below the alert threshold, "
            f"with a maximum probability of {confidence * 100:.1f}%. {explain_summary}"
        )

    @staticmethod
    def _build_recommendations(
        prediction: str,
        confidence: float,
        seizure_duration_min: float,
        dominant_region: Optional[str],
    ) -> List[str]:
        recs: List[str] = []

        prediction_lower = prediction.lower()
        review_needed = "review needed" in prediction_lower
        seizure_detected = "seizure" in prediction_lower and (
            "detected" in prediction_lower or "activity" in prediction_lower
        )

        if seizure_detected and not review_needed:
            recs.append("Immediate clinical review is recommended.")
            if confidence >= 0.90:
                recs.append("High-confidence detection supports urgent correlation with EEG waveform review.")
            else:
                recs.append("Correlate the flagged segments with clinical context and waveform inspection.")
            if seizure_duration_min >= 10:
                recs.append("Consider prolonged seizure burden assessment and continued close monitoring.")
            if dominant_region:
                recs.append(
                    f"Review activity in the {dominant_region.lower()} region for focal involvement."
                )
        elif review_needed:
            recs.append("Clinical review is recommended before interpreting this result as a definitive seizure event.")
            recs.append("Correlate the flagged segments with waveform inspection and the broader patient context.")
            if dominant_region:
                recs.append(
                    f"Review activity in the {dominant_region.lower()} region for possible focal involvement."
                )
        else:
            recs.append("Continue routine EEG monitoring if clinically indicated.")
            recs.append("Correlate findings with clinical observations and specialist review.")

        recs.append("This output should be interpreted by a qualified healthcare professional.")
        return recs

    @staticmethod
    def _draw_header(
        c: canvas.Canvas,
        x: float,
        y: float,
        width: float,
        report: Dict[str, Any],
    ) -> float:
        c.setFillColor(colors.HexColor("#0f172a"))
        c.roundRect(x, y - 32, width, 32, 6, fill=1, stroke=0)

        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 15)
        c.drawString(x + 10, y - 12, "NeuroXAI Seizure Detection Report")

        prediction = str(report["prediction"])
        badge_text = prediction
        badge_width = stringWidth(badge_text, "Helvetica-Bold", 9) + 14
        badge_x = x + width - badge_width - 10

        prediction_lower = prediction.lower()

        if "review needed" in prediction_lower:
            badge_color = colors.HexColor("#b45309")
        elif "seizure" in prediction_lower and ("detected" in prediction_lower or "activity" in prediction_lower):
            badge_color = colors.HexColor("#b91c1c")
        else:
            badge_color = colors.HexColor("#065f46")
        c.setFillColor(badge_color)
        c.roundRect(badge_x, y - 24, badge_width, 14, 4, fill=1, stroke=0)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 9)
        c.drawString(badge_x + 7, y - 19, badge_text)

        return y - 42

    @staticmethod
    def _draw_section_title(
        c: canvas.Canvas,
        x: float,
        y: float,
        title: str,
    ) -> float:
        c.setFillColor(colors.HexColor("#1d4ed8"))
        c.setFont("Helvetica-Bold", 11)
        c.drawString(x, y, title)
        return y - 10

    @staticmethod
    def _draw_key_value_block(
        c: canvas.Canvas,
        x: float,
        y: float,
        width: float,
        rows: List[Tuple[str, str]],
    ) -> float:
        row_height = 16
        block_height = (len(rows) * row_height) + 10

        c.setFillColor(colors.HexColor("#f8fafc"))
        c.setStrokeColor(colors.HexColor("#cbd5e1"))
        c.roundRect(x, y - block_height + 4, width, block_height, 5, fill=1, stroke=1)

        current_y = y - 10
        for label, value in rows:
            c.setFont("Helvetica-Bold", 9)
            c.setFillColor(colors.HexColor("#334155"))
            c.drawString(x + 8, current_y, f"{label}:")
            c.setFont("Helvetica", 9)
            c.setFillColor(colors.HexColor("#0f172a"))
            c.drawString(x + 110, current_y, str(value)[:90])
            current_y -= row_height

        return y - block_height - 4

    @staticmethod
    def _draw_paragraph(
        c: canvas.Canvas,
        text: str,
        x: float,
        y: float,
        width: float,
        line_height: int = 12,
    ) -> float:
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.HexColor("#0f172a"))
        lines = ReportService._wrap_text(text, "Helvetica", 9, width)
        for line in lines:
            c.drawString(x, y, line)
            y -= line_height
        return y

    @staticmethod
    def _draw_bullets(
        c: canvas.Canvas,
        items: List[str],
        x: float,
        y: float,
        width: float,
    ) -> float:
        c.setFont("Helvetica", 9)
        c.setFillColor(colors.HexColor("#0f172a"))

        for item in items:
            wrapped = ReportService._wrap_text(item, "Helvetica", 9, width - 12)
            if not wrapped:
                continue
            c.drawString(x, y, u"\u2022")
            c.drawString(x + 10, y, wrapped[0])
            y -= 12
            for line in wrapped[1:]:
                c.drawString(x + 10, y, line)
                y -= 12

        return y

    @staticmethod
    def _draw_probability_chart(
        c: canvas.Canvas,
        probabilities: List[float],
        x: float,
        y: float,
        width: float,
        height: float,
        threshold: float = 0.5,
        duration_minutes: Optional[float] = None,
    ) -> float:
        chart_bottom = y - height
        c.setFillColor(colors.HexColor("#ffffff"))
        c.setStrokeColor(colors.HexColor("#cbd5e1"))
        c.roundRect(x, chart_bottom, width, height, 5, fill=1, stroke=1)

        if not probabilities:
            c.setFillColor(colors.HexColor("#64748b"))
            c.setFont("Helvetica", 9)
            c.drawCentredString(x + width / 2, chart_bottom + height / 2, "No probability data available")
            return chart_bottom - 8

        inner_left = x + 12
        inner_right = x + width - 12
        inner_bottom = chart_bottom + 18
        inner_top = y - 12

        plot_w = inner_right - inner_left
        plot_h = inner_top - inner_bottom

        # axes
        c.setStrokeColor(colors.HexColor("#94a3b8"))
        c.line(inner_left, inner_bottom, inner_left, inner_top)
        c.line(inner_left, inner_bottom, inner_right, inner_bottom)

        # threshold line
        th_y = inner_bottom + max(0.0, min(1.0, threshold)) * plot_h
        c.setStrokeColor(colors.HexColor("#f59e0b"))
        c.setDash(3, 2)
        c.line(inner_left, th_y, inner_right, th_y)
        c.setDash()
        c.setFillColor(colors.HexColor("#b45309"))
        c.setFont("Helvetica", 7)
        c.drawRightString(inner_right, th_y + 3, f"Threshold {threshold:.2f}")

        # grid / y labels
        for frac, label in [(0.0, "0"), (0.25, "25"), (0.5, "50"), (0.75, "75"), (1.0, "100")]:
            gy = inner_bottom + frac * plot_h
            c.setStrokeColor(colors.HexColor("#e2e8f0"))
            c.line(inner_left, gy, inner_right, gy)
            c.setFillColor(colors.HexColor("#64748b"))
            c.setFont("Helvetica", 7)
            c.drawRightString(inner_left - 3, gy - 2, label)

        # line
        vals = [max(0.0, min(1.0, float(v))) for v in probabilities]
        n = len(vals)
        if n == 1:
            px = inner_left
            py = inner_bottom + vals[0] * plot_h
            c.setFillColor(colors.HexColor("#ef4444"))
            c.circle(px, py, 1.5, fill=1, stroke=0)
        else:
            c.setStrokeColor(colors.HexColor("#ef4444"))
            c.setLineWidth(1.3)
            for i in range(n - 1):
                x1 = inner_left + (i / (n - 1)) * plot_w
                y1 = inner_bottom + vals[i] * plot_h
                x2 = inner_left + ((i + 1) / (n - 1)) * plot_w
                y2 = inner_bottom + vals[i + 1] * plot_h
                c.line(x1, y1, x2, y2)

        # x labels
        c.setFillColor(colors.HexColor("#64748b"))
        c.setFont("Helvetica", 7)
        if duration_minutes and duration_minutes > 0 and n > 1:
            for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
                tx = inner_left + frac * plot_w
                tlabel = f"{duration_minutes * frac:.1f}"
                c.drawCentredString(tx, inner_bottom - 10, tlabel)
            c.drawCentredString((inner_left + inner_right) / 2, chart_bottom + 4, "Time (min)")
        else:
            c.drawCentredString((inner_left + inner_right) / 2, chart_bottom + 4, "Window index")

        return chart_bottom - 8

    @staticmethod
    def _draw_channel_bar_chart(
        c: canvas.Canvas,
        top_channels: List[Dict[str, Any]],
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> float:
        chart_bottom = y - height
        c.setFillColor(colors.HexColor("#ffffff"))
        c.setStrokeColor(colors.HexColor("#cbd5e1"))
        c.roundRect(x, chart_bottom, width, height, 5, fill=1, stroke=1)

        if not top_channels:
            c.setFillColor(colors.HexColor("#64748b"))
            c.setFont("Helvetica", 9)
            c.drawCentredString(x + width / 2, chart_bottom + height / 2, "No channel contribution data available")
            return chart_bottom - 8

        inner_left = x + 24
        inner_right = x + width - 12
        inner_bottom = chart_bottom + 22
        inner_top = y - 12

        plot_w = inner_right - inner_left
        plot_h = inner_top - inner_bottom

        c.setStrokeColor(colors.HexColor("#94a3b8"))
        c.line(inner_left, inner_bottom, inner_left, inner_top)
        c.line(inner_left, inner_bottom, inner_right, inner_bottom)

        items = top_channels[:6]
        n = len(items)
        gap = 10
        bar_w = max(10, (plot_w - gap * (n + 1)) / max(n, 1))

        for frac, label in [(0.0, "0"), (0.25, "25"), (0.5, "50"), (0.75, "75"), (1.0, "100")]:
            gy = inner_bottom + frac * plot_h
            c.setStrokeColor(colors.HexColor("#e2e8f0"))
            c.line(inner_left, gy, inner_right, gy)
            c.setFillColor(colors.HexColor("#64748b"))
            c.setFont("Helvetica", 7)
            c.drawRightString(inner_left - 4, gy - 2, label)

        for i, item in enumerate(items):
            score = max(0.0, min(1.0, float(item.get("score", 0.0))))
            bh = score * plot_h
            bx = inner_left + gap + i * (bar_w + gap)
            by = inner_bottom

            c.setFillColor(colors.HexColor("#3b82f6"))
            c.rect(bx, by, bar_w, bh, fill=1, stroke=0)

            c.setFillColor(colors.HexColor("#334155"))
            c.setFont("Helvetica", 7)
            c.drawCentredString(bx + bar_w / 2, by - 10, str(item.get("channel", ""))[:10])
            c.drawCentredString(bx + bar_w / 2, by + bh + 3, f"{score * 100:.0f}")

        return chart_bottom - 8

    @staticmethod
    def _wrap_text(
        text: str,
        font_name: str,
        font_size: int,
        max_width: float,
    ) -> List[str]:
        words = str(text).split()
        if not words:
            return []

        lines: List[str] = []
        current = words[0]

        for word in words[1:]:
            trial = f"{current} {word}"
            if stringWidth(trial, font_name, font_size) <= max_width:
                current = trial
            else:
                lines.append(current)
                current = word

        lines.append(current)
        return lines