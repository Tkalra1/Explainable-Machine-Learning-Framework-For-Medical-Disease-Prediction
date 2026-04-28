import streamlit as st
import numpy as np
import pickle
import shap
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(page_title="Explainable Machine Learning Framework for Medical Disease Prediction", layout="wide")

st.title(" Explainable Machine Learning Framework for Medical Disease Prediction")
st.markdown("-by Taneeshq Kalra")


heart_model = pickle.load(open("model.pkl", "rb"))
diabetes_model = pickle.load(open("diabetes_model.pkl", "rb"))


disease = st.selectbox("Select Disease", ["Heart Disease", "Diabetes"])

st.divider()


col1, col2 = st.columns(2)

if disease == "Heart Disease":

    with col1:
        age = st.slider("Age", 1, 100, 30)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.slider("Chest Pain Type", 0, 3, 1)
        trestbps = st.slider("Resting BP", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 400, 200)

    with col2:
        fbs = st.selectbox("Fasting Blood Sugar>120", ["No", "Yes"])
        restecg = st.slider("ECG", 0, 2, 1)
        thalach = st.slider("Max Heart Rate", 60, 220, 150)
        exang = st.selectbox("Exercise Angina", ["No", "Yes"])
        oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)

    slope = st.slider("Slope", 0, 2, 1)
    ca = st.slider("Major Vessels", 0, 4, 0)
    thal = st.slider("Thal", 0, 3, 1)

    sex = 1 if sex == "Male" else 0
    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                            restecg, thalach, exang, oldpeak,
                            slope, ca, thal]])

    model_used = heart_model

elif disease == "Diabetes":

    with col1:
        pregnancies = st.slider("Pregnancies", 0, 20, 1)
        glucose = st.slider("Glucose", 50, 200, 120)
        bp = st.slider("Blood Pressure", 40, 140, 70)
        skin = st.slider("Skin Thickness", 0, 100, 20)

    with col2:
        insulin = st.slider("Insulin", 0, 300, 80)
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
        age = st.slider("Age", 1, 100, 30)

    input_data = np.array([[pregnancies, glucose, bp, skin,
                            insulin, bmi, dpf, age]])

    model_used = diabetes_model


if st.button(" Predict"):

    result = model_used.predict(input_data)[0]
    prob = model_used.predict_proba(input_data)[0][1]

    st.divider()
    st.subheader(" Prediction Result")

    # Risk Meter
    st.progress(int(prob * 100))

    if prob < 0.3:
        st.success(f" Low Risk ({prob*100:.2f}%)")
    elif prob < 0.7:
        st.warning(f" Moderate Risk ({prob*100:.2f}%)")
    else:
        st.error(f" High Risk ({prob*100:.2f}%)")

   
    # SHAP Explainability
    
    st.subheader(" Explainable AI Insights")

    try:
        if disease == "Heart Disease":
            data = pd.read_csv("heart.csv")
            X = data.drop("target", axis=1)
        else:
            data = pd.read_csv("diabetes.csv")
            X = data.drop("Outcome", axis=1)

        explainer = shap.TreeExplainer(model_used)

        
        shap_values_single = explainer.shap_values(input_data)

       
        if isinstance(shap_values_single, list):
            contributions = shap_values_single[1] if len(shap_values_single) > 1 else shap_values_single[0]
        else:
            contributions = shap_values_single

        contributions = np.array(contributions)

      
        if contributions.ndim > 1:
            contributions = contributions[0]

        contributions = contributions.flatten()

       
        contributions = contributions[:len(X.columns)]
        feature_names = X.columns[:len(contributions)]

        df_contrib = pd.DataFrame({
            "Feature": feature_names,
            "Contribution": contributions
        }).sort_values(by="Contribution", ascending=False)

        st.markdown(" Feature Contributions")
        st.dataframe(df_contrib)

        
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_values_plot = shap_values

       
        if len(shap_values_plot.shape) > 2:
            shap_values_plot = shap_values_plot[:, :, 0]

        shap.summary_plot(shap_values_plot, X, show=False)

        fig = plt.gcf()
        st.pyplot(fig)
        plt.clf()

    except Exception as e:
        st.warning(f"Explainability error: {e}")

    
    # REPORT
    
    report = f"""
     Health Report

    Disease: {disease}
    Prediction: {'High Risk' if result == 1 else 'Low Risk'}
    Probability: {prob*100:.2f}%
    """

    st.download_button(" Download Report", report, file_name="report.txt")