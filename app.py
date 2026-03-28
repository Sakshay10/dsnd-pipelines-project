import streamlit as st
import joblib
import pandas as pd

# Load trained pipeline
model = joblib.load("model.pkl")


# -----------------------------
# UI
# -----------------------------

st.title("Fashion Recommendation Predictor")

st.write("Enter review details to predict whether a customer would recommend the product.")

# User inputs
review_text = st.text_area("Review Text")

age = st.slider("Age", 18, 80, 30)

department = st.selectbox(
    "Department Name",
    ["Tops", "Dresses", "Bottoms", "Intimate", "Jackets"]
)

product_class = st.text_input("Class Name", "General")


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict"):

    # Create input dataframe
    input_data = pd.DataFrame({
        "Review Text": [review_text],
        "Age": [age],
        "Department Name": [department],
        "Class Name": [product_class],
        "doc_length": [len(review_text)],
        "noun_count": [0],
        "verb_count": [0],
        "stopword_count": [0]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.success(f"Recommended ✅ (Confidence: {probability:.2f})")
    else:
        st.error(f"Not Recommended ❌ (Confidence: {probability:.2f})")