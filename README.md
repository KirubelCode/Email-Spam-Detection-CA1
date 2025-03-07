Spam Detection using Machine Learning
Produced by: Kirubel Temesgen
College ID: C00260396
Date: 28/02/2025

Overview
This project focuses on building and evaluating a spam classification model using machine learning techniques. Three models—Naïve Bayes, Support Vector Machines (SVM), and K-Nearest Neighbours (KNN)—are implemented and compared to determine the best performer in detecting spam emails.

The Enron Email Dataset, containing over 500,000 real-world emails, is used for training and testing. The CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology structures the workflow, ensuring effective data preparation, feature engineering, and model evaluation.

Features
Dataset Preprocessing – Cleaning and transforming raw email data.
Feature Engineering – TF-IDF vectorisation, stopword removal, and n-gram analysis.
Spam Detection Models – Implementing and evaluating Naïve Bayes, SVM, and KNN.
Performance Metrics – Accuracy, precision, recall, F1-score, and confusion matrix analysis.
Optimisation – Exploring different preprocessing techniques to improve classification.
Installation
1. Clone the Repository
Run the following command in your terminal:

git clone https://github.com/KirubelCode/Spam-Detection-Project.git
cd Spam-Detection-Project

2. Install Dependencies
Ensure Python 3.x is installed, then install the required libraries:

pip install -r requirements.txt

3. Run the Application
python app.py

The application will be available at http://127.0.0.1:5000/

Dataset Information
Source: The Enron Email Dataset.
Size: 500,000+ emails from Enron employees.
Spam Labeling: Spam manually identified based on keywords and sender domain.
Preprocessing:
Removed missing values and irrelevant columns.
Applied TF-IDF vectorisation for text transformation.
Used n-grams to improve spam pattern recognition.
Model Evaluation
1. Naïve Bayes Classifier
Accuracy: 83.91%
Strengths: Fast, scalable, effective for text classification.
Weaknesses: Struggles with complex relationships between words.
2. Support Vector Machine (SVM)
Accuracy: 89.77%
Strengths: Effective for high-dimensional data, finds optimal separation between classes.
Weaknesses: Computationally expensive, slower training time.
3. k-Nearest Neighbours (k-NN)
Accuracy: 82.04%
Strengths: Simple, non-parametric, useful for non-linear classification.
Weaknesses: Slow for large datasets, struggles with spam detection accuracy.
Model Comparison
Model	Accuracy	Strengths	Weaknesses
Naïve Bayes	83.91%	Fast, effective for text	Assumes word independence
SVM	89.77%	High accuracy, optimal separation	Computationally expensive
k-NN	82.04%	Simple, no training phase	Slow, struggles with large data
File Structure
Spam-Detection-Project
│── app.py # Main Flask application
│── dataset/ # Email dataset
│── models/ # Saved trained models
│── templates/ # HTML templates for UI
│── static/ # CSS, JavaScript, and images
│── requirements.txt # Required dependencies
│── README.md # Project documentation

Future Enhancements
Improve feature engineering – Explore deep learning techniques like Word2Vec and transformers.
Reduce false positives – Optimise model hyperparameters to improve spam classification.
Hybrid Model Approach – Combine multiple models for better accuracy.
