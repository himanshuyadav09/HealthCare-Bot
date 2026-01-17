from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import os
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer, util
from difflib import get_close_matches

# --------------------------------------------------
# APP INIT
# --------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------
# MEDICAL CONFIGURATION
# --------------------------------------------------
PRIMARY_SYMPTOMS = {
    "vomiting", "diarrhea", "rash", "chest_pain", "shortness_of_breath",
    "wheezing", "night_sweats", "weight_loss", "bloody_stool",
    "loss_of_taste", "loss_of_smell", "stiff_neck", "confusion",
    "blurred_vision", "light_sensitivity"
}

SECONDARY_SYMPTOMS = {
    "headache", "fever", "fatigue", "body_ache", "nausea",
    "loss_of_appetite", "runny_nose", "nasal_congestion",
    "dry_cough", "dizziness"
}

# --------------------------------------------------
# LOAD NLP MODEL
# --------------------------------------------------
nlp_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAINING_PATH = os.path.join(BASE_DIR, "Data", "Training_cleaned.csv")
DESC_PATH = os.path.join(BASE_DIR, "MasterData", "symptom_Description_cleaned.csv")
PRECAUTION_PATH = os.path.join(BASE_DIR, "MasterData", "symptom_precaution_cleaned.csv")
CURE_PATH = os.path.join(BASE_DIR, "MasterData", "natural_cures_cleaned.csv")

# --------------------------------------------------
# LOAD TRAINING DATA
# --------------------------------------------------
df = pd.read_csv(TRAINING_PATH)
df.columns = df.columns.str.strip().str.lower()

X = df.iloc[:, :-1]
y = df["prognosis"].str.strip().str.title()

le = LabelEncoder()
y_encoded = le.fit_transform(y)

clf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)
clf.fit(X, y_encoded)

known_symptoms = list(X.columns)
symptom_embeddings = nlp_model.encode(known_symptoms, convert_to_tensor=True)

# --------------------------------------------------
# LOAD MASTER DATA (SAFE)
# --------------------------------------------------
desc_df = pd.read_csv(DESC_PATH)
desc_df.columns = desc_df.columns.str.strip().str.lower()
desc_df["disease"] = desc_df["disease"].str.strip().str.title()
desc_dict = desc_df.set_index("disease")["description"].to_dict()

prec_df = pd.read_csv(PRECAUTION_PATH)
prec_df.columns = prec_df.columns.str.strip().str.lower()
prec_df["disease"] = prec_df["disease"].str.strip().str.title()
precaution_dict = prec_df.set_index("disease").iloc[:, :4].T.to_dict("list")

cure_df = pd.read_csv(CURE_PATH)
cure_df.columns = cure_df.columns.str.strip().str.lower()
cure_df["disease"] = cure_df["disease"].str.strip().str.title()
cure_dict = cure_df.set_index("disease")["natural_cure"].to_dict()

# --------------------------------------------------
# NLP SYMPTOM EXTRACTION
# --------------------------------------------------
def extract_symptoms(text):
    emb = nlp_model.encode(text, convert_to_tensor=True)
    scores = util.cos_sim(emb, symptom_embeddings)[0]
    k = min(5, scores.shape[0])

    top = torch.topk(scores, k)
    return [
        known_symptoms[i]
        for i, s in zip(top.indices, top.values)
        if s.item() > 0.45
    ]

def auto_correct(symptoms):
    corrected = set()
    for s in symptoms:
        corrected.update(get_close_matches(s, known_symptoms, cutoff=0.7))
    return list(corrected)

# --------------------------------------------------
# MEDICAL RULE ENGINE
# --------------------------------------------------
def medical_gate(symptoms):
    primary = [s for s in symptoms if s in PRIMARY_SYMPTOMS]

    if len(symptoms) < 2:
        return False, "Insufficient symptoms provided."

    if len(primary) == 0:
        return False, "Only common symptoms detected. Please add more specific symptoms."

    return True, ""

def severity_level(symptoms):
    primary_count = sum(1 for s in symptoms if s in PRIMARY_SYMPTOMS)

    if primary_count >= 2:
        return "High"
    if primary_count == 1:
        return "Medium"
    return "Low"

def confidence_text(prob):
    if prob >= 0.7:
        return "High confidence (clear symptom pattern)"
    if prob >= 0.4:
        return "Medium confidence (partial overlap)"
    return "Low confidence (insufficient data)"

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["symptoms"]

    # Optional location data (already supported by your HTML)
    latitude = request.form.get("latitude")
    longitude = request.form.get("longitude")

    symptoms = auto_correct(extract_symptoms(text))
    allowed, reason = medical_gate(symptoms)

    # ---------- CASE 1: UNKNOWN / INSUFFICIENT ----------
    if not allowed:
        # If location exists → hospital finder page
        if latitude and longitude:
            return render_template(
                "no_diagnosis.html",
                latitude=latitude,
                longitude=longitude
            )

        return render_template(
            "result.html",
            disease="No Diagnosis",
            description=reason,
            precautions=[],
            cure="N/A"
        )

    # ---------- CASE 2: ML PREDICTION ----------
    input_vector = np.zeros(len(known_symptoms))
    for i, s in enumerate(known_symptoms):
        if s in symptoms:
            input_vector[i] = 1

    probs = clf.predict_proba([input_vector])[0]
    top3_idx = np.argsort(probs)[-3:][::-1]

    top3 = [(le.inverse_transform([i])[0], probs[i]) for i in top3_idx]
    best_disease, best_prob = top3[0]

    explanation = (
        f"Severity: {severity_level(symptoms)} | "
        f"{confidence_text(best_prob)} | "
        f"Other possible conditions: "
        + ", ".join(f"{d} ({p:.0%})" for d, p in top3[1:])
    )

    return render_template(
        "result.html",
        disease=best_disease,
        description=desc_dict.get(best_disease, "") + "\n\n" + explanation,
        precautions=precaution_dict.get(best_disease, []),
        cure=cure_dict.get(best_disease, "")
    )

# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
