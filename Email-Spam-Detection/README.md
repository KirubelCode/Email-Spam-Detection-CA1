pam Detection using Machine Learning Produced by: Kirubel Temesgen College ID: C00260396 Date: 28/02/2025

Overview This project focuses on building and evaluating a spam classification model using machine learning techniques. Three models—Naïve Bayes, Support Vector Machines (SVM), and K-Nearest Neighbours (KNN)—are implemented and compared to determine the best performer in detecting spam emails.

The Enron Email Dataset, containing over 500,000 real-world emails, is used for training and testing. The CRISP-DM (Cross-Industry Standard Process for Data Mining) methodology structures the workflow, ensuring effective data preparation, feature engineering, and model evaluation.

Features Dataset Preprocessing – Cleaning and transforming raw email data. Feature Engineering – TF-IDF vectorisation, stopword removal, and n-gram analysis. Spam Detection Models – Implementing and evaluating Naïve Bayes, SVM, and KNN. Performance Metrics – Accuracy, precision, recall, F1-score, and confusion matrix analysis. Optimisation – Exploring different preprocessing techniques to improve classification. Installation

Clone the Repository Run the following command in your terminal:
git clone https://github.com/KirubelCode/Email-Spam-Detection-CA1 cd Email-Spam-Detection-CA1

Install Dependencies Ensure Python 3.x is installed, then install the required libraries:

Run the Application python app.py

Dataset Information Source: The Enron Email Dataset. Size: 500,000+ emails from Enron employees. Link: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset

Spam Labeling: Spam manually identified based on keywords and sender domain.

Preprocessing: Removed missing values and irrelevant columns. Applied TF-IDF vectorisation for text transformation. Used n-grams to improve spam pattern recognition.

Model Evaluation For:

Naïve Bayes Classifier
Support Vector Machine(SVM)
K-Nearest Neighbour (KNN)
Neural Networks(MLP)
Kmeans-Clustering

Future Enhancements Improve feature engineering – Explore deep learning techniques like transformers. Reduce false positives – Optimise model hyperparameters to improve spam classification. Hybrid Model Approach – Combine multiple models for better accuracy.
