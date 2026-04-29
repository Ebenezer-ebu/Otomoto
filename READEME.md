# 🚀 Otomoto Marketing Segmentation Optimization (ANN Churn Prediction)

## 📌 Overview

This project focuses on building and optimizing an **Artificial Neural Network (ANN)** to predict **customer churn** using telecom customer data as a proxy for Otomoto’s marketing segmentation use case.

The goal is to identify customers likely to churn so that targeted retention strategies can be applied, ultimately improving marketing ROI and reducing revenue loss.

---

## 🎯 Objectives

- Build a baseline ANN model for churn prediction
- Experiment with multiple optimization algorithms:
  - SGD with Momentum
  - RMSprop
  - Adam (with tuned learning rate)
- Compare performance using key classification metrics
- Identify the best model for business use (not just accuracy)

---

## 🧠 Problem Statement

Otomoto, an automotive marketplace, struggles with effectively segmenting customers for targeted marketing campaigns. Poor segmentation leads to inefficient spending and missed retention opportunities.

This project addresses that by:
- Predicting churn probability
- Enabling data-driven customer segmentation
- Supporting smarter marketing decisions

---

## 📊 Dataset

- **Name:** Telco Customer Churn Dataset  
- **Source:** IBM Sample Dataset (Kaggle)  
- **Samples:** 7,043 customers  
- **Features:** 21 columns (19 features after preprocessing)  
- **Target Variable:** `Churn` (Yes/No)

### Key Features:
- Customer tenure
- Contract type (monthly, yearly)
- Monthly & total charges
- Internet & subscription services

---

## ⚙️ Tech Stack

- Python 3.x
- TensorFlow / Keras (v2.13.0)
- NumPy / Pandas
- Scikit-learn
- Matplotlib / Seaborn

---

## 🏗️ Model Architecture

Baseline ANN structure:
```
Input Layer (19 features)
↓
Dense Layer (64 neurons, ReLU)
↓
Batch Normalization
↓
Dropout (30%)
↓
Dense Layer (32 neurons, ReLU)
↓
Batch Normalization
↓
Dropout (30%)
↓
Output Layer (1 neuron, Sigmoid)
```

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

---

## 🧪 Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|------|--------|----------|--------|----------|------|
| Baseline (Adam) | 79.42% | 0.64 | 0.52 | 0.5748 | 0.8385 |
| SGD + Momentum | 79.49% | 0.64 | 0.51 | 0.5693 | 0.8396 |
| RMSprop | 79.28% | 0.63 | **0.53** | **0.5780** | 0.8356 |
| Adam (LR=0.0005) | 79.35% | 0.63 | 0.53 | 0.5776 | 0.8377 |

---

## 🏆 Best Model

**RMSprop** was selected as the best model based on:

- Highest **F1-Score**
- Best **Recall** (critical for churn detection)

> 📌 Insight: While accuracy differences are minimal, RMSprop performs better at identifying actual churners — which is more valuable for business impact.

---

## 💼 Business Impact

- ~**53.5% of churners** correctly identified
- Enables:
  - Targeted retention campaigns
  - Better marketing spend allocation
- Estimated savings:
  - ~$53K annually (based on churn prevention assumptions)

---

## 📂 Project Structure
```
.
├── data/
│ └── telco_churn.csv
├── notebooks/
│ └── model_experiments.ipynb
├── src/
│ ├── preprocessing.py
│ ├── model.py
│ ├── train.py
│ └── evaluate.py
├── outputs/
│ ├── otomoto_optimization_results.png
│ └── best_otomoto_model.h5
├── requirements.txt
└── README.md
```

---

## ⚡ How to Run the Project

### 1. Clone the Repository

```
git clone git@github.com:Ebenezer-ebu/Otomoto.git
cd otomoto
```
### 2. Create a Virtual Environment
```
python -m venv venv
```
### Activate it:
Mac/Linux
```
source venv/bin/activate
```
Windows
```
venv\Scripts\activate
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```

### 4. Add Dataset

Place the dataset at the root:
```
teleconnect.csv
```

### 5. Run Training
```
python src/train.py
```

### 6. Evaluate Model
```
python src/evaluate.py
```

### 7. View Outputs
Generated files:
- 📊 otomoto_optimization_results.png → model comparison visualization
- 🤖 best_otomoto_model.h5 → saved trained model

## 🔍 Key Learnings
- Optimizer choice had minimal impact on accuracy
- Recall and F1-score are more important than accuracy for churn prediction
- Slight improvements can still have significant business value
- Feature engineering likely has more impact than optimizer tuning

## 🚧 Future Improvements
- Hyperparameter tuning (GridSearch / RandomSearch)
- Handle class imbalance (SMOTE / class weights)
- Add explainability (SHAP / LIME)
- Experiment with deeper neural networks
- Deploy as an API (Flask / FastAPI)

## 📚 References
- Géron, A. — Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow
- Kingma & Ba — Adam Optimization Algorithm
- TensorFlow Documentation

👤 Author
Ebenezer Ifezulike (Eben)
Frontend & Full Stack Developer | Machine Learning Enthusiast

## ⭐ Final Note
This project demonstrates how machine learning can be applied to solve real-world business problems — not just by improving metrics, but by driving actionable insights and measurable impact.

