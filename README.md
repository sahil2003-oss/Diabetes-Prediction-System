# Diabetes Prediction System

A supervised machine learning classification system built for early-stage clinical screening of diabetes risk based on historical medical parameters.

---

## Overview

This project analyzes clinical metrics such as insulin levels, glucose concentration, body mass index (BMI), and blood pressure to reliably predict patient diabetes status.

---

## Key Features

* **Clinical Classification**: Designed a supervised ML pipeline achieving **88% overall accuracy**.
* **Model Optimization**: Evaluated and fine-tuned multiple algorithms (Logistic Regression, Support Vector Machines, Random Forest) with an emphasis on minimizing false negatives for reliable medical screening.
* **Statistical Analysis**: Conducted feature-relationship mapping using Seaborn to study correlations between vital diagnostics (glucose, insulin, BMI, blood pressure).
* **Interactive Deployment**: Deployed as a live web app via Streamlit.

---

## Tech Stack

* **Language**: Python
* **Libraries & Visualization**: scikit-learn, Pandas, NumPy, Seaborn
* **Deployment**: Streamlit Cloud

---

## Getting Started

### Prerequisites

* Python 3.x

### Installation & Setup

1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/sahil2003-oss/diabetes-prediction-system.git](https://github.com/sahil2003-oss/diabetes-prediction-system.git)
   cd diabetes-prediction-system

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Run the Streamlit App:**:
   ```bash
   streamlit run app.py
