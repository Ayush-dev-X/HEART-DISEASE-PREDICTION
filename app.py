# ============================================================
# CARDIOSENSE - FLASK BACKEND
# Supports clinical-only and clinical + genetic assessments
# ============================================================

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request


# ------------------------------------------------------------
# 1. APP AND MODEL LOADING
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

CLINICAL_MODEL_PATH = BASE_DIR / "clinical_model.pkl"
GENETIC_MODEL_PATH = BASE_DIR / "genetic_model.pkl"

for model_path in (CLINICAL_MODEL_PATH, GENETIC_MODEL_PATH):
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Required model file not found: {model_path}\n"
            "Place clinical_model.pkl and genetic_model.pkl beside app.py."
        )

clinical_model = joblib.load(CLINICAL_MODEL_PATH)
genetic_model = joblib.load(GENETIC_MODEL_PATH)


# These names must match the raw DataFrame columns used to train
# the pipelines in Heart.ipynb.
CLINICAL_FEATURES = [
    "Age",
    "Sex",
    "ChestPainType",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "RestingECG",
    "MaxHR",
    "ExerciseAngina",
    "Oldpeak",
    "ST_Slope",
]

GENETIC_FEATURES = [
    "FamilyHistory",
    "rs10757274_Genotype",
    "APOE_Genotype",
    "Lpa_Level",
    "FH_Mutation",
    "Genetic_Risk_Score",
]


# ------------------------------------------------------------
# 2. PERSONALIZED GUIDANCE
# ------------------------------------------------------------

def generate_guidance(
    resting_bp,
    cholesterol,
    fasting_bs,
    exercise_angina,
    chest_pain,
    oldpeak,
    max_hr,
    risk_percentage,
):
    suggestions = []
    precautions = []

    # This wording describes the model output, not a diagnosis.
    if risk_percentage < 30:
        suggestions.append(
            "The model returned a relatively lower score for this assessment. "
            "Continue routine health checks and healthy habits."
        )
    elif risk_percentage < 60:
        suggestions.append(
            "Some assessment factors may deserve closer monitoring. "
            "Discuss your cardiovascular risk factors with a healthcare professional."
        )
    else:
        suggestions.append(
            "The model returned a higher score for this assessment. "
            "Arrange a discussion with a qualified healthcare professional."
        )

    if resting_bp >= 140:
        suggestions.append(
            "This resting blood-pressure reading is elevated; consider rechecking "
            "it correctly and discussing repeated high readings with a clinician."
        )
        precautions.append(
            "Do not change prescribed medication based solely on this app's output."
        )
    elif resting_bp >= 130:
        suggestions.append(
            "This resting blood-pressure reading may warrant regular monitoring."
        )

    if cholesterol >= 240:
        suggestions.append(
            "This cholesterol value is high; discuss the result and overall lipid "
            "profile with a healthcare professional."
        )
    elif cholesterol >= 200:
        suggestions.append(
            "Discuss this cholesterol value in the context of a full lipid profile "
            "and your other risk factors."
        )

    if fasting_bs == 1:
        suggestions.append(
            "The fasting blood-sugar indicator is set to the elevated category. "
            "A clinician can advise whether confirmatory glucose testing is needed."
        )

    if exercise_angina == "Y":
        precautions.append(
            "Discuss exercise-related chest discomfort or symptoms with a clinician. "
            "Do not deliberately exercise to reproduce symptoms."
        )

    if chest_pain == "ASY":
        suggestions.append(
            "The selected chest-pain category should be interpreted in the context "
            "of the person's symptoms and clinical assessment."
        )
    elif chest_pain in ("ATA", "NAP", "TA"):
        suggestions.append(
            "If chest discomfort is new, recurring, or worsening, seek clinical advice."
        )

    if oldpeak >= 2:
        suggestions.append(
            "Discuss this exercise-related ECG measurement with a healthcare professional."
        )

    if max_hr < 100:
        suggestions.append(
            "Interpret the recorded maximum heart rate in the context of age, effort, "
            "medications, and the circumstances of the test."
        )

    precautions.extend([
        "This application is an educational machine-learning demonstration, not a diagnosis.",
        "For new, severe, or persistent chest pain, severe breathlessness, fainting, or other "
        "possible emergency symptoms, seek urgent medical care.",
        "Healthy eating, avoiding tobacco, appropriate physical activity, adequate sleep, "
        "and routine medical care can support cardiovascular health.",
    ])

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(suggestions)), list(dict.fromkeys(precautions))


# ------------------------------------------------------------
# 3. PAGE ROUTES
# ------------------------------------------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/insights")
def insights():
    return render_template("insights.html")


# ------------------------------------------------------------
# 4. PREDICTION ROUTE
# ------------------------------------------------------------

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    try:
        # "clinical" is the safe default for existing forms that do not
        # yet include the assessment_mode field.
        assessment_mode = (
            request.form.get("assessment_mode")
            or request.form.get("mode")
            or "clinical"
        ).strip().lower()

        if assessment_mode in ("genetic", "clinical_genetic", "clinical+genetic"):
            assessment_mode = "genetic"
            selected_model = genetic_model
            model_used = "Clinical + Genetic Logistic Regression"
        else:
            assessment_mode = "clinical"
            selected_model = clinical_model
            model_used = "Clinical Logistic Regression"

        # Read the existing clinical form field names.
        age = float(request.form["age"])
        sex = request.form["sex"].strip().upper()
        chest_pain = request.form["cp"].strip().upper()
        resting_bp = float(request.form["trestbps"])
        cholesterol = float(request.form["chol"])
        fasting_bs = int(request.form["fbs"])
        resting_ecg = request.form["restecg"].strip().upper()
        max_hr = float(request.form["thalach"])
        exercise_angina = request.form["exang"].strip().upper()
        oldpeak = float(request.form["oldpeak"])
        st_slope = request.form["slope"].strip().upper()

        # Validate the categorical values expected by the training dataset.
        allowed_values = {
            "sex": {"M", "F"},
            "cp": {"ATA", "NAP", "TA", "ASY"},
            "restecg": {"NORMAL", "ST", "LVH"},
            "exang": {"Y", "N"},
            "slope": {"UP", "FLAT", "DOWN"},
        }
        submitted_values = {
            "sex": sex,
            "cp": chest_pain,
            "restecg": resting_ecg,
            "exang": exercise_angina,
            "slope": st_slope,
        }
        for field, value in submitted_values.items():
            if value not in allowed_values[field]:
                raise ValueError(f"Invalid value submitted for {field}.")

        if fasting_bs not in (0, 1):
            raise ValueError("Fasting blood sugar must be 0 or 1.")

        if age <= 0 or resting_bp <= 0 or cholesterol < 0 or max_hr <= 0:
            raise ValueError("Please check the age, blood pressure, cholesterol, and heart-rate values.")

        if oldpeak < 0:
            raise ValueError("Oldpeak cannot be negative in this form.")

        # Create the raw clinical DataFrame. The saved Pipeline handles
        # imputation, scaling, and one-hot encoding.
        clinical_input = {
            "Age": age,
            "Sex": sex,
            "ChestPainType": chest_pain,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "RestingECG": resting_ecg.title() if resting_ecg != "LVH" else "LVH",
            "MaxHR": max_hr,
            "ExerciseAngina": exercise_angina,
            "Oldpeak": oldpeak,
            "ST_Slope": st_slope.title(),
        }

        input_record = dict(clinical_input)

        # Genetic fields are only required for genetic assessment.
        family_history = None
        rs10757274_genotype = None
        apoe_genotype = None
        lpa_level = None
        fh_mutation = None
        genetic_risk_score = None

        if assessment_mode == "genetic":
            family_history = int(request.form["family_history"])
            rs10757274_genotype = request.form["rs10757274_genotype"].strip()
            apoe_genotype = request.form["apoe_genotype"].strip()
            lpa_level = float(request.form["lpa_level"])
            fh_mutation = int(request.form["fh_mutation"])
            genetic_risk_score = float(request.form["genetic_risk_score"])

            if family_history not in (0, 1):
                raise ValueError("Family history must be 0 or 1.")
            if fh_mutation not in (0, 1):
                raise ValueError("FH mutation must be 0 or 1.")
            if not rs10757274_genotype or not apoe_genotype:
                raise ValueError("Please select both genotype values.")
            if lpa_level < 0:
                raise ValueError("Lipoprotein(a) level cannot be negative.")

            input_record.update({
                "FamilyHistory": family_history,
                "rs10757274_Genotype": rs10757274_genotype,
                "APOE_Genotype": apoe_genotype,
                "Lpa_Level": lpa_level,
                "FH_Mutation": fh_mutation,
                "Genetic_Risk_Score": genetic_risk_score,
            })

        # Explicitly arrange the raw columns in the training order.
        required_features = (
            CLINICAL_FEATURES
            if assessment_mode == "clinical"
            else CLINICAL_FEATURES + GENETIC_FEATURES
        )
        input_data = pd.DataFrame([input_record]).reindex(
            columns=required_features
        )

        # A pipeline predicts directly from the raw feature DataFrame.
        prediction = int(selected_model.predict(input_data)[0])

        # Use the probability associated with class 1, rather than assuming
        # that the second probability column always represents class 1.
        risk_percentage = None
        if hasattr(selected_model, "predict_proba"):
            probabilities = selected_model.predict_proba(input_data)[0]
            classes = list(selected_model.classes_)
            if 1 in classes:
                class_index = classes.index(1)
                risk_percentage = round(float(probabilities[class_index]) * 100, 1)

        if risk_percentage is None:
            risk_percentage = 100.0 if prediction == 1 else 0.0

        if risk_percentage < 30:
            risk_level = "Low Risk"
        elif risk_percentage < 60:
            risk_level = "Moderate Risk"
        else:
            risk_level = "High Risk"

        prediction_label = (
            "Higher likelihood detected"
            if prediction == 1
            else "Lower likelihood detected"
        )

        suggestions, precautions = generate_guidance(
            resting_bp=resting_bp,
            cholesterol=cholesterol,
            fasting_bs=fasting_bs,
            exercise_angina=exercise_angina,
            chest_pain=chest_pain,
            oldpeak=oldpeak,
            max_hr=max_hr,
            risk_percentage=risk_percentage,
        )

        return render_template(
            "result.html",
            prediction=prediction,
            prediction_label=prediction_label,
            risk_percentage=risk_percentage,
            risk_level=risk_level,
            model_used=model_used,
            assessment_mode=assessment_mode,
            age=age,
            sex=sex,
            chest_pain=chest_pain,
            resting_bp=resting_bp,
            cholesterol=cholesterol,
            fasting_bs=fasting_bs,
            resting_ecg=resting_ecg,
            max_hr=max_hr,
            exercise_angina=exercise_angina,
            oldpeak=oldpeak,
            st_slope=st_slope,
            family_history=family_history,
            rs10757274_genotype=rs10757274_genotype,
            apoe_genotype=apoe_genotype,
            lpa_level=lpa_level,
            fh_mutation=fh_mutation,
            genetic_risk_score=genetic_risk_score,
            suggestions=suggestions,
            precautions=precautions,
        )

    except (KeyError, TypeError, ValueError) as exc:
        return render_template(
            "predict.html",
            error=f"Please check the submitted values. {exc}",
        )
    except Exception:
        app.logger.exception("CardioSense prediction failed")
        return render_template(
            "predict.html",
            error=(
                "The prediction could not be completed. Check that the form values "
                "match the dataset categories and that both saved model files are current."
            ),
        )


# ------------------------------------------------------------
# 5. RUN LOCALLY
# ------------------------------------------------------------

if __name__ == "__main__":
    print("CardioSense is starting...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(host="127.0.0.1", port=5000, debug=False)
