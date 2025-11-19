# Breast-Cancer-Prediction


A simple, fast, and user-friendly Machine Learning–powered Streamlit application that predicts the likelihood of breast cancer based on medical diagnostic features.


Overview

This project uses a trained machine learning model (Logistic Regression / Random Forest / SVM) to classify whether a tumor is Benign or Malignant using the Breast Cancer Wisconsin Dataset.
The application provides a clean UI built with Streamlit for quick predictions.


Features

1.Accurate Breast Cancer Prediction

2.Standardized Input Feature Handling

3.Real-time Prediction using Streamlit

4.Pre-trained Model Loaded via Pickle

5.Simple UI with clear output labels


Tech Stack


Python 3.9+

Streamlit

scikit-learn

Pandas

NumPy

Pickle


Installation & Setup


1️⃣ Clone the Repository

git clone https://github.com/yourusername/breast-cancer-prediction.git

cd breast-cancer-prediction


2️⃣ Install Dependencies

pip install -r requirements.txt


3️⃣ Run the Streamlit App

streamlit run app.py


How It Works


The input features are preprocessed using the saved StandardScaler.

The data is passed to the pre-trained ML model.

The model outputs the prediction:

0 → Benign (Non-cancerous)

1 → Malignant (Cancerous)

The result is displayed clearly in the UI.



Input Features


The model predicts based on commonly used diagnostic metrics such as:

Mean Radius

Mean Texture

Mean Smoothness

Mean Compactness

Mean Symmetry
…and other numerical features from the dataset.



User Interface (Streamlit)


Sidebar for feature input

Clean result display section

Minimalist design for clarity

Immediate prediction feedback


Dataset


This project uses the Breast Cancer Wisconsin dataset, containing 569 tumor records with 30 numeric features.


Model Training Summary


Algorithm: Logistic Regression / Random Forest / SVM (based on your project)

Train-Test Split: 80/20

Accuracy: ~95–98% depending on the model


Requirements


requirements.txt:

  streamlit
 
  pandas

  numpy

  scikit-learn

  joblib
