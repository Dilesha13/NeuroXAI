# NeuroXAI

> **NeuroXAI: An Explainable Multi-Scale Spatio-Temporal Graph Attention Network for Neonatal Seizure Detection Using EEG Signals**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![React](https://img.shields.io/badge/React-TypeScript-61DAFB)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# Overview

NeuroXAI is an AI-powered web application developed for automated neonatal seizure detection using Electroencephalography (EEG) recordings.

The project proposes a novel **Multi-Scale Spatio-Temporal Graph Attention Network (MST-GAT)** that combines temporal feature extraction at multiple time scales with graph-based spatial learning to accurately detect seizure events while providing interpretable explanations.

Unlike conventional black-box deep learning models, NeuroXAI integrates attention-based explainability to highlight important EEG channels, temporal segments, and graph connections that contribute to model predictions, improving clinician trust and decision support.

The system also provides a complete web application where clinicians can upload EEG recordings, manage patients, perform AI analysis, visualize explanations, and generate diagnostic reports.

---

# Key Features

- Automated neonatal seizure detection from EEG recordings
- Multi-Scale Temporal CNN architecture
- Graph Attention Network (GAT) for spatial EEG modelling
- Attention-based Explainable AI (XAI)
- EEG upload and preprocessing
- Patient management
- AI prediction dashboard
- Explainability visualization
- PDF report generation
- Authentication and user management
- Modern responsive web interface

---

# Research Motivation

Neonatal seizures are neurological emergencies that often occur without obvious clinical symptoms, making manual diagnosis difficult. Continuous EEG monitoring is considered the gold standard for seizure detection, but manual interpretation is time-consuming, requires specialist expertise, and is prone to variability between clinicians.

Although many deep learning models have achieved high detection accuracy, they often suffer from three major limitations:

- Lack of clinical interpretability
- Limited multi-scale temporal modelling
- Poor modelling of spatial relationships between EEG channels

NeuroXAI addresses these limitations through an explainable multi-scale spatio-temporal deep learning framework.

---

# Proposed Architecture

The proposed MST-GAT architecture consists of:

1. EEG preprocessing
2. Multi-scale temporal feature extraction
3. Graph Attention Network (GAT)
4. Temporal Attention
5. Feature Fusion
6. Binary seizure classification
7. Explainability generation

```
EEG Signal
     │
     ▼
Preprocessing
     │
     ▼
────────────────────────────
 Multi-Scale CNN Branches
────────────────────────────
 Short-term
 Medium-term
 Long-term
────────────────────────────
     │
     ▼
Graph Attention Network
     │
     ▼
Temporal Attention
     │
     ▼
Feature Fusion
     │
     ▼
Seizure Classification
     │
     ▼
Explainability Outputs
```

---

# Technology Stack

## AI / Machine Learning

- Python
- PyTorch
- PyTorch Geometric
- MNE-Python
- NumPy
- Pandas
- SciPy
- Scikit-learn
- Matplotlib

## Backend

- FastAPI
- SQLAlchemy
- PostgreSQL
- Pydantic

## Frontend

- React
- TypeScript
- Tailwind CSS
- Axios
- React Router

## Development Tools

- Git
- GitHub
- Docker
- Jupyter Notebook
- Google Colab
- RunPod

---

# Dataset

This project uses the **Helsinki Neonatal EEG Dataset**.

The dataset contains continuous neonatal EEG recordings with expert seizure annotations.

Due to licensing restrictions, the dataset is **not included** in this repository.

Please obtain the dataset from the official providers before reproducing the experiments.

---

# Data Preprocessing

The preprocessing pipeline includes:

- EDF file loading
- Channel normalization
- Bipolar montage generation
- Band-pass filtering (0.5–30 Hz)
- Resampling
- Sliding window segmentation
- Window labelling
- Tensor generation

---

# Explainable AI

Unlike traditional black-box models, NeuroXAI provides interpretable predictions through:

- Temporal attention visualization
- EEG channel importance
- Graph attention visualization
- Prediction confidence

These explanations help clinicians understand why the model predicts seizure activity.

---

# Web Application Features

The web application provides:

- User authentication
- Dashboard
- Patient management
- EEG upload
- AI analysis
- Explainability page
- Report generation
- Settings management

---

# Project Structure

```
NeuroXAI
│
├── backend
│   ├── api
│   ├── models
│   ├── preprocessing
│   ├── inference
│   ├── explainability
│   ├── reports
│   ├── services
│   └── main.py
│
├── frontend
│   ├── src
│   ├── components
│   ├── pages
│   ├── services
│   └── assets
│
├── notebooks
│   ├── preprocessing.ipynb
│   └── training.ipynb
│
├── trained_model
│
├── docs
│
├── README.md
└── requirements.txt
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/NeuroXAI.git

cd NeuroXAI
```

---

## Backend

Create virtual environment

```bash
python -m venv venv
```

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run FastAPI

```bash
uvicorn main:app --reload
```

---

## Frontend

```bash
cd frontend

npm install

npm run dev
```

---

# Running the Application

Start backend

```bash
uvicorn main:app --reload
```

Start frontend

```bash
npm run dev
```

Open

```
http://localhost:5173
```

---

# Model Evaluation

The proposed MST-GAT model was evaluated using subject-wise validation and standard classification metrics.

Evaluation metrics include:

- AUC-ROC
- Sensitivity
- Specificity
- Precision
- F1-score
- Confusion Matrix

Detailed experimental results are available in the thesis.

---

# Screenshots

Add screenshots here.

Example:

```
docs/
    dashboard.png
    upload.png
    analysis.png
    explainability.png
    reports.png
```

---

# Demonstration Videos

## Evaluation Video

[NeuroXAI Evaluation vedio](https://youtu.be/AZNuNvjdJN0)

---

## Presentation Video

[NeuroXAI Presentation vedio](https://youtu.be/QedwIsAsMQk)

---

# Thesis

The complete dissertation is available here.

[NeuroXAI Thesis](https://drive.google.com/file/d/1NESt6WLwQADytsDkKKnhfOUL7Z-ciTx0/view?usp=drive_link)

---

# Future Improvements

- Cross-dataset validation
- Real-time EEG streaming
- Cloud deployment
- Mobile application
- Clinical validation with hospitals
- Multi-class seizure classification
- Additional explainability techniques (SHAP, LIME)

---

# Citation

If you use this work, please cite:

```bibtex
@thesis{kariyawasam2026neuroxai,
  author = {Tharushi Dilesha Kariyawasam},
  title = {NeuroXAI: An Explainable Multi-Scale Spatio-Temporal Graph Attention Network for Neonatal Seizure Detection Using EEG Signals},
  school = {University of Westminster},
  year = {2026}
}
```

---

# Author

**Tharushi Dilesha Kariyawasam**

BEng (Hons) Software Engineering

University of Westminster

GitHub: https://github.com/Dilesha13

LinkedIn: https://www.linkedin.com/in/tharushi-dilesha-a3973b257/

---

# License

This project is licensed under the MIT License.

---

# Acknowledgements

- University of Westminster
- Informatics Institute of Technology (IIT)
- Helsinki Neonatal EEG Dataset contributors
- PyTorch
- FastAPI
- React
- MNE-Python
