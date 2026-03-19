# 🏥 MediScope — Patient Risk & Treatment Intelligence System

MediScope is an end-to-end Machine Learning pipeline that analyzes patient healthcare data to predict disease risk, estimate treatment cost, and segment patients into meaningful clusters for better decision-making.

It combines multiple ML techniques into a single structured workflow, making it a practical and real-world oriented project.


# 🚀 Problem Statement

Healthcare systems often deal with large patient datasets but lack quick, actionable insights such as:

* Who is at **high disease risk?**
* What is the **expected treatment cost?**
* How can patients be **grouped for better care planning?**

MediScope addresses these problems using Machine Learning models and automated reporting.

---

# ⚙️ Features

* Full ML Pipeline (Load → Preprocess → Train → Evaluate → Report)
* Regression Models (Linear, Ridge, Lasso) for cost prediction
* Classification Models (Logistic, Decision Tree, SVM) for disease risk
* Clustering (K-Means) for patient segmentation
* PCA for dimensionality reduction & visualization
* Confusion Matrix & ROC Curve evaluation
* Automated plots saved in structured folders
* CLI-based execution (modular commands)
* Patient-specific insight generation via CLI

---

# 🧠 ML Models Used

### Regression

* Linear Regression
* Ridge Regression
* Lasso Regression

### Classification

* Logistic Regression
* Decision Tree
* Support Vector Machine (SVM)

### Unsupervised Learning

* K-Means Clustering
* PCA (Dimensionality Reduction)

---

# 📂 Project Structure

```
mediscope-ml
│
├── config
│   └── config.yaml
│
├── data
│   └── patients.csv
│
├── src
│   ├── loader.py
│   ├── preprocessor.py
│   ├── regression.py
│   ├── classification.py
│   ├── clustering.py
│   ├── dimensionality.py
│   ├── evaluation.py
│   ├── visualization.py
│   ├── report.py
│   └── logger.py
│
├── outputs
│   ├── plots
│   ├── models
│   └── reports
│
├── notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Modeling.ipynb
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🔄 Pipeline Flow

```
Patient Data
   ↓
Preprocessing
   ↓
Regression + Classification Models
   ↓
Clustering (K-Means)
   ↓
PCA Visualization
   ↓
Evaluation Metrics
   ↓
Reports & Patient Insights
```

---

# 💻 How to Run (CLI)

### 1️⃣ Train Models

```bash
python main.py --train
```

---

### 2️⃣ Evaluate Models

```bash
python main.py --evaluate
```

---

### 3️⃣ Generate Plots

```bash
python main.py --plots
```

---

### 4️⃣ Generate Report

```bash
python main.py --report
```

---

### 5️⃣ Get Insight for Specific Patient

```bash
python main.py --report --patient 234
```

👉 Replace `234` with any valid row index from dataset.

---

### 🔥 Run Full Pipeline

```bash
python main.py --train --evaluate --plots --report
```

---

# 📊 Outputs

### Models

Saved in:

```
outputs/models/
```

### Plots

```
outputs/plots/
├── regression
├── classification
├── clustering
└── pca
```

### Reports

```
outputs/reports/
├── report.json
├── report.txt
```

---

# 🧍 Sample Patient Insight

```
Patient Insight
----------------------
Age: 54
BMI: 31.2
Smoker: yes

Risk Level: HIGH

Key Findings:
• High BMI detected
• Smoking risk present
• High glucose level

Recommendation:
- Improve diet
- Increase physical activity
- Regular health checkups
```

---

# 📓 Notebooks

* **01_EDA.ipynb** → Data analysis, distributions, correlations
* **02_Modeling.ipynb** → Model experimentation

---

# 🎯 Key Learnings

* End-to-end ML pipeline design
* Model comparison & evaluation
* Feature preprocessing (encoding, scaling)
* Real-world problem structuring
* CLI-based ML system design

---

# 🌟 Future Improvements

* Streamlit dashboard for interactive UI
* Hyperparameter tuning (GridSearchCV)
* Feature importance explanations (SHAP)
* Real medical dataset integration
* API deployment (Flask/FastAPI)

---

# 🧾 Author Note

This project is part of a Machine Learning roadmap and focuses on building a **practical, modular, and real-world ML system** rather than just isolated models.


