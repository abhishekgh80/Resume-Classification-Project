# Resume-Classification-Project
Overview:

This project focuses on building a system to classify resumes into different categories (e.g., Data Science, Web Development, SQL Devoloper, etc.) using Natural Language Processing (NLP) and machine learning techniques. By automating resume classification, organizations can efficiently filter candidates based on relevant skills and job roles.

Problem Statement

Recruiters receive a vast number of resumes for job openings, making it challenging to manually analyze and categorize them. This project aims to develop an automated resume classification system that categorizes resumes into predefined job roles or skill-based groups to streamline the hiring process.

Objectives

Process and clean text data from resumes.

Apply NLP techniques to extract meaningful information.

Build and evaluate machine learning models for resume classification.

Automate and optimize the categorization process for faster resume screening.


Technologies Used:

Programming Language: Python

Libraries:

Text Processing: nltk, spacy, re

Vectorization: TF-IDF, CountVectorizer

Machine Learning: scikit-learn, XGBoost

Visualization: matplotlib, seaborn

Model Deployment:Streamlit



Steps/Workflow:

1. Data Preprocessing:

Clean and preprocess raw text data:

Remove punctuation, stopwords, and special characters.

Perform tokenization and lemmatization.

Convert text into numerical format using techniques like TF-IDF.


2. Exploratory Data Analysis (EDA):

Analyze the distribution of categories.

Visualize word frequency and word clouds for key categories.

Check for imbalanced data.


3. Feature Extraction:

Use TF-IDF vectorization or Word2Vec embeddings to transform text into numerical vectors.


4. Model Building:

Train machine learning models:

Logistic Regression

Naive Bayes Classifier

Support Vector Machine (SVM)

Random Forest Classifier

Gradient Boosting (XGBoost)


5. Model Evaluation:

Evaluate models using:

Accuracy, Precision, Recall, and F1-Score.

Confusion Matrix for understanding misclassifications.


6. Model Deployment:

Deploy the best-performing model as a web application using Streamlit for real-time classification of resumes.


Results:

Best Model: Random Forest Clasiifier and XGBoost models achieved an accuracy of 100% on test data.

Insights:

Key words such as "Python", "Machine Learning", and "tika" strongly influence classifications into Data Science roles.

Categories like Web Development and Data Science were the most prevalent in the dataset.





