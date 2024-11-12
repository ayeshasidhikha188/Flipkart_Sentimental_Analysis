# Flipkart Sentiment Analysis & MLOps

## Overview

This repository contains a comprehensive project that combines **Sentiment Analysis** and **MLOps** practices to analyze customer sentiment from reviews of the "YONEX MAVIS 350 Nylon Shuttle" product on Flipkart. The project utilizes various machine learning models and pipelines to classify reviews into positive or negative sentiment. Additionally, it demonstrates how to deploy machine learning models with MLflow for experiment tracking, model management, and reproducibility.
![image](https://github.com/user-attachments/assets/13270946-787a-4dba-ac4e-555586d2aaba)

## Table of Contents
- [Project Objective](#project-objective)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Modeling Approach](#modeling-approach)
- [MLflow Integration](#mlflow-integration)
- [App Deployment](#app-deployment)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [License](#license)

---

## Project Objective

The main objective of this project is to:
- Classify Flipkart customer reviews as **positive** or **negative** using sentiment analysis.
- Analyze the pain points of customers writing negative reviews to gain insights into product features contributing to satisfaction or dissatisfaction.
- Implement **MLOps** principles for experiment tracking, model management, and real-time deployment.

---

## Dataset

The dataset contains **8,518 reviews** for the "YONEX MAVIS 350 Nylon Shuttle" product scraped from the Flipkart website. Each review includes the following features:
- **Reviewer Name**
- **Rating**
- **Review Title**
- **Review Text**
- **Place of Review**
- **Date of Review**
- **Up Votes**
- **Down Votes**



## Data Preprocessing

To ensure the model's success, the following preprocessing steps were applied to the review text data:
1. **Text Cleaning**: Removed special characters, punctuation, and stopwords.
2. **Text Normalization**: Applied lemmatization to reduce words to their base form.
3. **Numerical Feature Extraction**: Experimented with various text embedding techniques:
   - Bag-of-Words (BoW)
   - Term Frequency-Inverse Document Frequency (TF-IDF)
   - Word2Vec (W2V)
   - Glove
   - BERT

---

## Modeling Approach

The following models were trained and evaluated for sentiment classification:
- **Naive Bayes with BoW**
- **Logistic Regression with TF-IDF**
- **Random Forest with TF-IDF**
- **Logistic Regression with BERT (Selected for Deployment)**

The model's performance was evaluated based on the **F1 Score**, with BERT-based models showing the best results in terms of accuracy and generalization.

---

## MLflow Integration

For better experiment tracking and model management, **MLflow** was integrated into the workflow:
1. **Tracking experiments**: Log parameters, metrics, and artifacts during model training.
2. **Model Versioning**: Registered models and managed them with tags for better tracking.
3. **Metric and Hyperparameter Plots**: Visualized training progress using MLflow’s built-in plot capabilities.
4. **Deployment Readiness**: Tracked model performance over multiple runs for easy selection and deployment.

---

## App Deployment

To make the sentiment analysis model accessible in real-time, the following steps were performed:
1. **Flask/Streamlit App**: Developed a web application that accepts a user input review and predicts its sentiment (positive or negative).
2. **Model Integration**: Integrated the selected BERT-based model into the app for real-time inference.
3. **Deployment**: Deployed the Flask/Streamlit app on **AWS EC2** for scalability and accessibility.
![image](https://github.com/user-attachments/assets/3a798c56-e466-40e9-adbe-4887c1248160)

---

## Usage

To run the project locally:
1. Clone this repository:
    ```bash
    git clone https://github.com/ayeshasidhikha188/Flipkart_Sentimental_Analysis.git
    cd Flipkart_Sentimental_Analysis
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Start the Flask/Streamlit app:
    ```bash
    python app.py  # or streamlit run app.py for Streamlit
    ```

4. Access the app in your browser at `http://localhost:5000` for Flask or `http://localhost:8501` for Streamlit.

---

## Technologies Used

- **Python**
- **Scikit-learn** for machine learning models
- **TensorFlow** and **Hugging Face Transformers** for BERT-based models
- **MLflow** for experiment tracking and model management
- **Flask/Streamlit** for web app deployment
- **AWS EC2** for model deployment

---


