# iris_streamlit_app.py

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="🌸 Iris Flower Classification",
    layout="centered",
    page_icon="🌼"
)

st.title("🌸 Iris Flower Classification App")
st.markdown("""
Predict the **species of an Iris flower** using interactive sliders.  
Choose between **KNN, SVM, Decision Tree, Random Forest** models.
""")

# -------------------- LOAD MODELS & SCALER --------------------
folder_path = r"E:\100 Projects\2\iris-streamlit"

# Load scaler
scaler = pickle.load(open(f"{folder_path}\\scaler.pkl", "rb"))

# Load models
models = {
    "KNN": pickle.load(open(f"{folder_path}\\knn_model.pkl", "rb")),
    "SVM": pickle.load(open(f"{folder_path}\\svm_model.pkl", "rb")),
    "Decision Tree": pickle.load(open(f"{folder_path}\\decision_tree_model.pkl", "rb")),
    "Random Forest": pickle.load(open(f"{folder_path}\\random_forest_model.pkl", "rb"))
}

# -------------------- USER INPUTS --------------------
st.sidebar.header("Input Features")

def user_input_features():
    sepal_length = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.5)
    sepal_width = st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
    petal_length = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width = st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.0)

    return pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                        columns=['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'])

input_df = user_input_features()

# -------------------- MODEL SELECTION --------------------
model_choice = st.sidebar.selectbox(
    "Select ML Model",
    ("KNN", "SVM", "Decision Tree", "Random Forest")
)

selected_model = models[model_choice]

# -------------------- SCALE INPUT FOR KNN/SVM --------------------
if model_choice in ["KNN", "SVM"]:
    input_scaled = scaler.transform(input_df.values)
else:
    input_scaled = input_df.values

# -------------------- MAKE PREDICTION --------------------
prediction = selected_model.predict(input_scaled)[0]

# Prediction probability if available
try:
    prediction_proba = selected_model.predict_proba(input_scaled)
    proba_df = pd.DataFrame(prediction_proba, columns=selected_model.classes_)
except:
    proba_df = None

# -------------------- DISPLAY RESULTS --------------------
st.subheader("Prediction")
st.success(f"The predicted Iris species is: **{prediction}**")

if proba_df is not None:
    st.subheader("Prediction Probability")
    st.dataframe(proba_df.T)

# -------------------- OPTIONAL: EDA VISUALS --------------------
st.subheader("EDA: Feature Relationships")

iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Pairplot (static)
if st.checkbox("Show Pairplot"):
    fig = sns.pairplot(df, hue='species')
    st.pyplot(fig)

# Feature importance (for tree-based models)
if st.checkbox("Show Feature Importance (Tree Models Only)"):
    if model_choice in ["Decision Tree", "Random Forest"]:
        importance = selected_model.feature_importances_
        importance_df = pd.DataFrame({
            "Feature": df.columns[:-1],
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        fig2 = px.bar(importance_df, x="Feature", y="Importance",
                      title=f"{model_choice} Feature Importance")
        st.plotly_chart(fig2)
    else:
        st.info("Feature importance is available only for tree-based models.")

# -------------------- FOOTER --------------------
st.markdown("""
---
Made by **Khuzaima Amir**
""")
