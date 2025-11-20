## **Breast Cancer Prediction using SVM**

A complete end-to-end Machine Learning project that uses the Support Vector Machine (SVM) algorithm to classify whether a patient is likely to have breast cancer based on medical features.

This project covers data preprocessing, EDA, model training, hyperparameter tuning, evaluation, model saving, and a Streamlit/Flask frontend (if you add it later).

---

**Project Overview**

Breast cancer is one of the most common cancers affecting women worldwide. Early detection greatly improves treatment outcomes.

This project uses the Breast Cancer Wisconsin dataset and applies SVM classification to predict whether a tumor is:

1.Benign (non-cancerous)

2.Malignant (cancerous)

---

**Tech Stack**

Python

Pandas, NumPy

Matplotlib / Seaborn (optional for EDA)

Scikit-learn

Jupyter Notebook / VS Code

---

**Dataset**

The dataset used is the Breast Cancer dataset from scikit-learn, consisting of:

569 samples

30 medical features

**Target:**

0 → Malignant

1 → Benign

---

**Steps Involved**

1️⃣ Data Loading

Load the dataset from sklearn

Convert to a pandas DataFrame

2️⃣ Exploratory Data Analysis

Statistical summary

Feature correlation

Distribution plots (optional)

3️⃣ Preprocessing

Scaling using StandardScaler

Train-test split

4️⃣ Model Training

Algorithm used:
✔ Support Vector Classifier (SVC) with:

RBF Kernel (best for non-linear separation)

Gamma & C tuning

5️⃣ Model Evaluation

Accuracy

Precision

Recall

Classification report

Confusion matrix

6️⃣ Saving the Model

Save trained model as svm_model.pkl

7️⃣ Deployment (Optional)

Can be integrated with:

Streamlit

Flask
to create a real-time prediction app.

---

**Why SVM?**

SVM is powerful for medical datasets because:

Works well with high-dimensional features

Finds the optimal separating hyperplane

Performs well with non-linear boundaries using kernels

Reduces overfitting with margin maximization

---

**How to Run**

1. Install dependencies

pip install -r requirements.txt


2. Run the Jupyter Notebook

jupyter notebook


3. Run Streamlit (if added)

streamlit run app.py


---

**Results**

SVM performed with high accuracy (usually 95%+)

Able to clearly differentiate malignant vs benign

Model is scalable for production-level healthcare use-cases
