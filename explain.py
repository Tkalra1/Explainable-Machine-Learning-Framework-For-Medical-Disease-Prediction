import shap
import pickle
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Load dataset
data = pd.read_csv("dataset/heart.csv")
X = data.drop("target", axis=1)

# SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Plot
shap.summary_plot(shap_values, X)