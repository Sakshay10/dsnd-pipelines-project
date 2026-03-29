import streamlit as st
import joblib
import pandas as pd

# -------------------------------
# Load trained model
# -------------------------------
model = joblib.load("model.pkl")


# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Fashion Predictor", layout="centered")

st.title("Fashion Recommendation Predictor")

st.write(
    "Enter product review details to predict whether a customer would recommend the product."
)


# -------------------------------
# Inputs
# -------------------------------
review_text = st.text_area("Review Text")

age = st.slider("Age", 18, 80, 30)

department = st.selectbox(
    "Department Name",
    ["Tops", "Dresses", "Bottoms", "Intimate", "Jackets"]
)

product_class = st.text_input("Class Name", "General")


# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    if review_text.strip() == "":
        st.warning("Please enter a review text.")
    else:
        # Create input dataframe (RAW input only)
        input_data = pd.DataFrame({
            "Review Text": [review_text],
            "Age": [age],
            "Department Name": [department],
            "Class Name": [product_class]
        })

        # Prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        # Output
        st.subheader("Prediction Result")

        if prediction == 1:
            st.success(f"Recommended ✅ (Confidence: {probability:.2f})")
        else:
            st.error(f"Not Recommended ❌ (Confidence: {probability:.2f})")


# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built using Machine Learning Pipeline with NLP + Streamlit")