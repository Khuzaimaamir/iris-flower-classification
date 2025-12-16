# iris_streamlit_app.py

import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Iris Flower Classification",
    page_icon="🌸",
    layout="centered"
)

# ==================== TITLE ====================
st.title("🌸 Iris Flower Classification App")
st.markdown(
    """
An interactive **Machine Learning web app** to predict the **species of an Iris flower**
using multiple classification models.
"""
)

# ==================== CLASS NAMES ====================
CLASS_NAMES = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

# ==================== LOAD MODELS ====================
# All files must be in the same folder for deployment
scaler = pickle.load(open("scaler.pkl", "rb"))

models = {
    "KNN": pickle.load(open("knn_model.pkl", "rb")),
    "SVM": pickle.load(open("svm_model.pkl", "rb")),
    "Decision Tree": pickle.load(open("decision_tree_model.pkl", "rb")),
    "Random Forest": pickle.load(open("random_forest_model.pkl", "rb"))
}

# ==================== SIDEBAR INPUT ====================
st.sidebar.header("🌿 Input Flower Features")

def user_inputs():
    return pd.DataFrame(
        [[
            st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.5),
            st.sidebar.slider("Sepal Width (cm)", 2.0, 4.5, 3.0),
            st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 4.0),
            st.sidebar.slider("Petal Width (cm)", 0.1, 2.5, 1.0)
        ]],
        columns=[
            "sepal length (cm)",
            "sepal width (cm)",
            "petal length (cm)",
            "petal width (cm)"
        ]
    )

input_df = user_inputs()

model_choice = st.sidebar.selectbox(
    "🤖 Choose ML Model",
    list(models.keys())
)

model = models[model_choice]

# ==================== SCALING ====================
if model_choice in ["KNN", "SVM"]:
    input_data = scaler.transform(input_df.values)
else:
    input_data = input_df.values

# ==================== PREDICTION ====================
prediction = model.predict(input_data)[0]
predicted_class = CLASS_NAMES[prediction]

st.markdown("### 🔮 Prediction Result")
st.success(f"**Predicted Iris Species:** 🌸 **{predicted_class}**")

# ==================== CONFIDENCE & PROBABILITY ====================
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(input_data)[0]

    proba_df = pd.DataFrame({
        "Species": [CLASS_NAMES[i] for i in model.classes_],
        "Probability": probabilities
    }).sort_values("Probability", ascending=False)

    confidence = proba_df.iloc[0]["Probability"]

    st.metric(
        label="Prediction Confidence",
        value=f"{confidence:.2%}"
    )

    st.markdown("### 📊 Class Probability Distribution")

    fig = px.bar(
        proba_df,
        x="Species",
        y="Probability",
        color="Species",
        text_auto=".2%",
        title="Prediction Probability by Class"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        proba_df.style.format({"Probability": "{:.2%}"})
    )

else:
    st.info("This model does not support probability prediction.")

# ==================== EDA SECTION ====================
st.markdown("---")
st.markdown("## 🔍 Exploratory Data Analysis")

iris = load_iris()
eda_df = pd.DataFrame(iris.data, columns=iris.feature_names)
eda_df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

with st.expander("📈 View Pairplot"):
    fig = sns.pairplot(eda_df, hue="species")
    st.pyplot(fig)

with st.expander("🌲 Feature Importance (Tree Models Only)"):
    if model_choice in ["Decision Tree", "Random Forest"]:
        importance_df = pd.DataFrame({
            "Feature": iris.feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig2 = px.bar(
            importance_df,
            x="Feature",
            y="Importance",
            title=f"{model_choice} Feature Importance"
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Feature importance available only for tree-based models.")

# ==================== ABOUT PROJECT ====================
st.markdown("---")
st.markdown(
    """
### 📌 About This Project
- Built as part of the **100 Data Science Projects Challenge**
- Trained and compared **KNN, SVM, Decision Tree & Random Forest**
- Applied **feature scaling** where required
- Designed for **real-world deployment**, not just notebooks
"""
)

# ==================== FOOTER ====================
st.markdown(
    """
---
👨‍💻 **Developed by Khuzaima Amir**  
🚀 *End-to-End Machine Learning + Streamlit Deployment*
"""
)
