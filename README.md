# Salary Predictor

A machine learning-powered web app that predicts Data Science salary ranges based on your profile — built with XGBoost, Flask, and a custom dark-themed UI.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-3.x-black?logo=flask)
![XGBoost](https://img.shields.io/badge/XGBoost-ML-orange)
![TailwindCSS](https://img.shields.io/badge/TailwindCSS-UI-38bdf8?logo=tailwindcss)

---

## Live Demo

> Coming soon — deploying on Render

---

## What It Does

Instead of predicting a single salary number (which is unreliable), this app predicts a **salary range** using three models:

| Model | Percentile | Meaning |
|---|---|---|
| Conservative | 25th | Lower-end realistic estimate |
| Most Likely | 50th | Median salary prediction |
| Optimistic | 75th | Higher-end realistic estimate |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | XGBoost (Quantile Regression) |
| Backend | Flask (Python) |
| Frontend | HTML + TailwindCSS + Vanilla JS |
| Data | 70,000+ US Data Science salaries (2020–2025) |
| Deployment | Render |

---

## Project Structure

```
salary-predictor/
│
├── model/                  ← Trained ML models (generated locally)
│   ├── model_low.pkl
│   ├── model_mid.pkl
│   ├── model_high.pkl
│   ├── encoders.pkl
│   └── features.pkl
│
├── static/
│   └── logo.png            ← App logo
│
├── templates/
│   └── index.html          ← Frontend UI
│
├── app.py                  ← Flask backend & API
├── train_model.py          ← ML model training script
├── explore.py              ← Dataset exploration
├── requirements.txt        ← Python dependencies
└── README.md
```

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/salary-predictor.git
cd salary-predictor
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download the DS Salaries dataset from Kaggle and place it as `ds_salaries.csv` in the root folder:
https://www.kaggle.com/datasets/samithsachidanandan/the-global-ai-ml-data-science-salary-for-2025

### 5. Train the model
```bash
python train_model.py
```
This generates the `.pkl` model files inside the `model/` folder.

### 6. Run the app
```bash
python app.py
```

Open your browser and go to http://127.0.0.1:5000

---

## Input Features

| Feature | Description |
|---|---|
| Job Title | Your data science role |
| Experience Level | EN (Entry) / MI (Mid) / SE (Senior) / EX (Executive) |
| Employment Type | Full-time, Part-time, Contract, Freelance |
| Company Size | Small / Medium / Large |
| Remote Ratio | On-site (0%) / Hybrid (50%) / Remote (100%) |
| Work Year | 2020–2025 |

---

## Model Performance

- **Dataset**: 70,000+ US-based data science salaries
- **Salary Range**: $70,000 – $294,000
- **Range Coverage**: ~49% of actual salaries fall within the predicted range
- **Average Range Width**: ~$70,000

A 49% coverage at the 25th–75th percentile range is mathematically expected and aligns with how industry tools like Glassdoor and LinkedIn Salary report ranges.

---

## Screenshots

> Screenshots of your app here after deployment!

---

## Acknowledgements

- Dataset: Kaggle - Global AI/ML/Data Science Salary 2025
- UI Design: Generated with Google Stitch, customized manually
- ML: XGBoost Quantile Regression

---

## License

This project is open source and available under the MIT License.
