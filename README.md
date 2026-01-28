
# 🧠 Neonatal Seizure Detection – NeuroXAI Prototype

This project presents a **working prototype** for **EEG-based neonatal seizure detection** using a deep learning model with **explainable AI (XAI)**.
It demonstrates an **end-to-end pipeline** from raw EEG input to seizure probability prediction and visual explanation.

⚠️ **This is a baseline research prototype**, developed as part of an undergraduate Final Year Project (FYP). It is **not intended for clinical use**.

---

## 📌 Key Features

* 📂 Load **raw neonatal EEG recordings** (EDF format)
* ⚙️ On-the-fly EEG preprocessing
* 🧩 Window-based seizure prediction using a trained CNN
* 📈 Seizure probability timeline visualization
* 🧠 Explainability via **gradient-based saliency maps**
* 🖥️ Interactive **Streamlit web interface**

---

## 🧪 Dataset

This project uses the **Helsinki Neonatal EEG Dataset**, which includes:

* `eeg1.edf` – `eeg79.edf` (raw EEG recordings)
* `annotations_2017.mat` (expert seizure annotations)
* Clinical metadata and ethics approval documents

📄 **Note:**
Due to ethical and licensing restrictions, the dataset is **not included in this repository**.

---

## 📁 Project Structure

```
NeuroXAI/
│ README.md
│ requirements.txt
│ app.py
│
├── data/                     # NOT pushed to GitHub
│   └── raw/
│       ├── eeg/              # eeg1.edf ... eeg79.edf
│       └── annotations/
│           └── annotations_2017.mat
│
├── neuroxai/
│   ├── data/
│   │   ├── load_edf.py
│   │   ├── load_annotations.py
│   │   ├── segment.py
│   │   └── splits.py
│   ├── models/
│   │   └── cnn1d.py
│   ├── preprocessing/
│   │   └── filters.py
│   └── explainability/
│       └── saliency.py
│
├── scripts/
│   ├── run_preprocess.py
│   ├── run_train.py
│   ├── run_eval.py
│   └── run_explain.py
│
└── outputs/
    └── runs/
        └── prototype_cnn/
            ├── best_model.pt
            ├── metrics_test.json
            └── confusion_matrix_test.txt
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/NeuroXAI.git
cd NeuroXAI
```

### 2️⃣ Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Model Training (Offline – Optional)

Model training is **not required to run the prototype UI**, as a trained model (`best_model.pt`) is already provided.

If retraining is needed:

```bash
python -m scripts.run_preprocess
python -m scripts.run_train
```

This will:

* preprocess EEG data
* train the CNN model
* save the best model as `best_model.pt`

---

## 🖥️ Running the Prototype (Streamlit UI)

### 1️⃣ Ensure required files exist

```
data/raw/eeg/*.edf
data/raw/annotations/annotations_2017.mat
outputs/runs/prototype_cnn/best_model.pt
```

### 2️⃣ Launch the app

```bash
streamlit run app.py
```

### 3️⃣ Open browser

```
http://localhost:8501
```

---

## 🧠 How the Prototype Works

### 🔹 Inference Pipeline (Real-World Style)

1. User selects an EEG recording
2. EEG is preprocessed **on-the-fly**
3. Signal is segmented into overlapping windows
4. Each window is passed through the trained CNN
5. Seizure probability is computed per window
6. Results are visualized as:

   * Probability timeline
   * Saliency heatmap (XAI)

👉 **No preprocessed dataset is required at inference time**, mimicking real clinical deployment.

---

## 📈 Model Performance (Baseline)

| Metric      | Value |
| ----------- | ----- |
| AUC         | 0.57  |
| Sensitivity | 0.38  |
| Specificity | 0.78  |

⚠️ These results represent a **baseline model**. Improving performance is part of future work.

---

## 🧠 Explainability (XAI)

Gradient-based **saliency maps** are used to visualize:

* which EEG channels
* and which time samples

most influenced the model’s seizure prediction.

This supports **model transparency and interpretability**, which are critical in healthcare AI.

---

## 🚧 Limitations

* Baseline CNN architecture
* Moderate sensitivity
* No cross-hospital validation
* Not clinically certified

---

## 🔮 Future Work

* Improved architectures (GNNs, attention models)
* Better class-imbalance handling
* Advanced explainability (Grad-CAM, SHAP)
* Cross-dataset validation
* Clinical collaboration
