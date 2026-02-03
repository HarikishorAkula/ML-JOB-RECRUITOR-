# 🚀 ML Job Recruiter using FastAPI

This project is a Machine Learning based Job Recruitment Predictor built using:

- Python
- FastAPI
- Scikit-learn
- HTML/CSS
- Logistic Regression

## 📌 Features

- Predicts whether a candidate will be Shortlisted or Rejected
- Shows confidence score
- Beautiful UI
- REST API using FastAPI

## 🧠 ML Model

Trained using Logistic Regression on recruitment dataset.

Features used:

- Skills Match Score
- Project Count
- Resume Length
- Github Activity
- Education Level
- Experience Level

## ▶️ How to Run

```bash
pip install fastapi uvicorn scikit-learn numpy joblib
uvicorn main:app --reload
