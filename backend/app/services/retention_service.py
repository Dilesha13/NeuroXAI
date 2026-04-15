from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.db.models.explanation import Explanation
from app.db.models.inference_result import InferenceResult
from app.db.models.report import Report



class RetentionService:
    @staticmethod
    def _parse_retention(value: str) -> Optional[timedelta]:
        raw = (value or "").strip().lower()

        if raw == "5 minutes":
            return timedelta(minutes=5)
        if raw == "1 day":
            return timedelta(days=1)
        if raw == "3 months":
            return timedelta(days=90)
        if raw == "6 months":
            return timedelta(days=180)
        if raw == "1 year":
            return timedelta(days=365)
        if raw == "forever":
            return None

        return timedelta(days=90)

    @staticmethod
    def cleanup_old_analysis_data(db: Session, retention_value: str) -> dict:
        delta = RetentionService._parse_retention(retention_value)
        if delta is None:
            return {
                "retention_value": retention_value,
                "deleted_inference_results": 0,
                "deleted_explanations": 0,
                "deleted_reports": 0,
                "deleted_files": 0,
                "matched_inference_ids": [],
                "message": "Retention is set to Forever. No cleanup performed.",
            }

        cutoff = datetime.now(timezone.utc) - delta

        all_inferences = db.query(InferenceResult).all()

        old_inferences = []
        for inf in all_inferences:
            created = inf.created_at
            if created is None:
                continue

            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)

            if created < cutoff:
                old_inferences.append(inf)

        inference_ids = [item.id for item in old_inferences]

        print("retention_value =", retention_value)
        print("cutoff =", cutoff)
        print("all_inference_ids =", [item.id for item in all_inferences])
        print("matched_inference_ids =", inference_ids)

        if not inference_ids:
            return {
                "retention_value": retention_value,
                "deleted_inference_results": 0,
                "deleted_explanations": 0,
                "deleted_reports": 0,
                "deleted_files": 0,
                "matched_inference_ids": [],
                "message": "No old analysis data found for cleanup.",
            }

        explanations = (
            db.query(Explanation)
            .filter(Explanation.inference_result_id.in_(inference_ids))
            .all()
        )

        reports = (
            db.query(Report)
            .filter(Report.inference_result_id.in_(inference_ids))
            .all()
        )

        print("matched_explanation_ids =", [item.id for item in explanations])
        print("matched_report_ids =", [item.id for item in reports])

        deleted_files = 0

        for exp in explanations:
            for maybe_path in [
                exp.saliency_path,
                exp.temporal_attention_path,
                exp.gat_attention_path,
            ]:
                if maybe_path:
                    p = Path(maybe_path)
                    if p.exists() and p.is_file():
                        try:
                            p.unlink()
                            deleted_files += 1
                        except Exception as e:
                            print("failed deleting explanation artifact:", maybe_path, e)

        for rep in reports:
            if rep.report_path:
                p = Path(rep.report_path)
                if p.exists() and p.is_file():
                    try:
                        p.unlink()
                        deleted_files += 1
                    except Exception as e:
                        print("failed deleting report file:", rep.report_path, e)

        deleted_explanations = len(explanations)
        deleted_reports = len(reports)
        deleted_inference_results = len(old_inferences)

        for exp in explanations:
            db.delete(exp)

        for rep in reports:
            db.delete(rep)

        db.flush()

        for inf in old_inferences:
            db.delete(inf)

        db.commit()

        return {
            "retention_value": retention_value,
            "deleted_inference_results": deleted_inference_results,
            "deleted_explanations": deleted_explanations,
            "deleted_reports": deleted_reports,
            "deleted_files": deleted_files,
            "matched_inference_ids": inference_ids,
            "message": "Cleanup completed successfully.",
        }