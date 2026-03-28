# Fashion Recommendation Prediction Pipeline

This project builds an end-to-end machine learning pipeline to predict whether a customer would recommend a product based on review text, customer demographics, and product categories.

The project demonstrates how to combine Natural Language Processing (NLP) with structured data in a single pipeline and includes a Streamlit dashboard for real-time predictions.

---

## Getting Started

Follow the steps below to run this project on your local machine.

---

## Dependencies

The project uses the following Python libraries:

- pandas  
- numpy  
- scikit-learn  
- spacy  
- matplotlib  
- joblib  
- streamlit  

---

## Installation

1. Clone the repository:
-git clone https://github.com/Sakshay10/dsnd-pipelines-project.git

-cd dsnd-pipelines-project


2. Install required libraries:
-pip install -r requirements.txt


3. Download spaCy model:
-python -m spacy download en_core_web_sm


4. Run the Jupyter Notebook:
- Open the notebook file
- Run all cells step-by-step to train the model

5. (Optional) Run the dashboard:


---

## Testing

This project does not include unit tests. Instead, model performance is validated using evaluation metrics.

---

## Break Down Tests

The model is evaluated using:

- Accuracy  
- Precision  
- Recall  
- F1-score  

The dataset is split into training and testing sets to ensure proper evaluation on unseen data.

---

## Project Instructions

The project follows these steps:

1. Data exploration to understand structure and missing values  
2. Feature engineering using TF-IDF and spaCy-based features  
3. Building a unified machine learning pipeline using ColumnTransformer  
4. Training a Logistic Regression model  
5. Evaluating model performance using classification metrics  
6. Fine-tuning the model using GridSearchCV  
7. Saving the trained pipeline as a `.pkl` file  
8. Creating a Streamlit dashboard for real-time predictions  

---

## Built With

- pandas – Data manipulation  
- scikit-learn – Machine learning pipeline and modeling  
- spaCy – NLP feature extraction  
- Streamlit – Interactive dashboard  
- matplotlib – Data visualization  

---

## License

This project is for learning and project purposes as part of the Udacity Data Science Nanodegree.
