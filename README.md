# HealthCare Bot 🩺🤖

HealthCare Bot is an AI-based web application that provides **preliminary medical guidance** by analyzing user-entered symptoms and predicting possible diseases.  
The system uses **Natural Language Processing (NLP)** and **Machine Learning** trained on **CSV datasets** to ensure simple, explainable, and safe predictions.

⚠️ This application is intended for **educational and academic purposes only** and is **not a replacement for professional medical advice**.

---

## 🚀 Features

- 🔹 Symptom input in natural language  
- 🔹 NLP-based symptom extraction  
- 🔹 Machine Learning disease prediction  
- 🔹 Medical rule engine to avoid unsafe diagnosis  
- 🔹 Severity level detection (Low / Medium / High)  
- 🔹 Confidence explanation for predictions  
- 🔹 Top-3 disease predictions with probabilities  
- 🔹 Nearby hospital suggestion when diagnosis is uncertain  
- 🔹 Simple and clean web interface  

---

## 🧠 How the System Works

1. User enters symptoms (e.g., *headache, fever, vomiting*)
2. NLP maps input text to known symptoms
3. A medical rule engine checks if symptoms are sufficient
4. If sufficient:
   - ML model predicts **Top-3 diseases**
   - Shows **severity** and **confidence**
   - Displays **precautions and natural remedies**
5. If insufficient:
   - System avoids unsafe prediction
   - Suggests **nearby hospitals**

---

## 📊 Dataset (CSV-Based)

The system is trained using structured CSV files:

### 1️⃣ Training Dataset
`Training_cleaned.csv`
- Binary symptom columns (0 / 1)
- Disease label (`prognosis`)
- Carefully designed to avoid false predictions

### 2️⃣ Disease Description
`symptom_Description_cleaned.csv`
- Disease name
- Detailed description

### 3️⃣ Precautions
`symptom_precaution_cleaned.csv`
- Four precautions per disease

### 4️⃣ Natural Remedies
`natural_cures_cleaned.csv`
- Supportive care suggestions

All datasets are **cleaned, normalized, and medically structured** to improve accuracy.

---


