# NeuroXAI Backend

FastAPI backend for a final-year project prototype on explainable neonatal EEG seizure detection.

## Features
- EDF upload
- EEG preprocessing aligned with training
- MST-GAT checkpoint loading
- Window-level inference
- Basic explainability outputs (saliency, temporal attention, GAT attention summary)
- Patient, EEG record, inference result, and report storage
- PDF report generation

## Important
This is a research prototype / clinical decision-support demo. It is **not** a clinically validated standalone diagnostic system.

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

## API docs
- Swagger UI: `/docs`
- OpenAPI JSON: `/openapi.json`

## Notes
- The checkpoint file is included under `assets/checkpoints/`.
- The MST-GAT architecture and graph definition were extracted from the supplied training notebook.
- Inference uses a fixed stride of 10 seconds for deployment. Training used label-aware asymmetric stepping, which is not directly available at inference time.
