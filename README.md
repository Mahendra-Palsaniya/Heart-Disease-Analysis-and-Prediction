# Heart-Disease-Analysis-and-Prediction

This project explores how machine learning can be used to assess the risk of heart disease based on clinical and physiological attributes. Instead of just building models, the notebook walks through how raw medical data can be transformed into meaningful predictions that may support early detection.

***

## Project Workflow

 - Data Cleaning and Preprocessing
 - Exploratory Data Analysis (EDA)
 - Outlier Removal (IQR Method)
 - Feature Engineering
 - Machine Learning Model Building
 - Hyperparameter Tuning (RandomizedSearchCV)
 - Model Evaluation and Comparison
 - Pipeline Building and Deployment

***

## Models Implemented

Several classification algorithms are explored to understand different learning behaviors:

 - Logistic Regression
 - Decision Tree Classifier
 - Random Forest Classifier
 - Gradient Boosting Classifier
 - Support Vector Machine (SVM)
 - K-Nearest Neighbors (KNN)

Rather than focusing only on accuracy, the notebook emphasizes how models differ in decision-making.

***
## 📊 About the Dataset
 - Source: Cleveland Heart Disease Dataset (1988) — UCI Machine Learning Repository
 - Total Records: 303 patients (after duplicate removal and outlier filtering)
 - Original Features: 14 attributes (demographic and clinical)
 - Engineered Features: 3 additional interaction features created
 - Target Variable: target — Heart Disease Status (0 = No Disease, 1 = Disease)
 - Class Distribution: Balanced (~51% disease, ~49% no disease)

***
## 🌐 Live Demo

***
## Observations & Learnings

Some interesting patterns emerge during analysis:

 - Certain types of **chest pain strongly correlate** with heart disease
 - **Maximum heart rate** shows a noticeable relationship with risk levels
 - Patients with **exercise-induced symptoms** are more likely to be affected
 - No single feature is sufficient—**combinations of features drive predictions**
 - Simpler models can sometimes perform comparably to complex ones

***

## Key Insights

 - Patients aged **40–65 years** represent the highest-risk group for heart disease
 - **Male patients** constitute ~69% of the dataset and show higher overall prevalence
 - **Asymptomatic chest pain (cp=3)** is paradoxically the strongest predictor of disease
 - **Higher maximum heart rate (thalach)** strongly correlates with lower disease risk
 - **ST depression (oldpeak > 2.0)** is a critical indicator of severe cardiac stress
 - **Reversible thalassemia defect (thal=2)** patients show significantly higher disease rates
 - **Flat or downsloping ST segment (slope=1,2)** indicates higher risk versus upsloping
 - **Number of major vessels (ca)** — patients with 0 vessels blocked have much lower risk

***

## Tech Stack
 - Python
 - Pandas & NumPy
 - Matplotlib & Seaborn
 - Scikit-learn
 - Joblib
 - Streamlit

***

## Conclusion
This project demonstrates how to build and deploy a complete machine learning pipeline for heart disease prediction. By packaging all preprocessing and model steps into a single sklearn Pipeline, inference is clean and error-free. Integrating the model into a Streamlit app provides an easy-to-use interface for real-time clinical predictions, making it practical for healthcare screening applications.
