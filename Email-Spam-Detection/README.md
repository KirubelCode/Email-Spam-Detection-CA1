Email Spam Detection (Machine Learning Project)
Project Overview
This project aims to build a machine learning-based email spam detection system. I individually developed the entire project, utilising a variety of machine learning algorithms including Naïve Bayes, Support Vector Machine (SVM), and K-Nearest Neighbors (KNN), Kmeans clustering and Neural Networks to classify emails as spam or ham (non-spam). The project leverages the Enron Email Dataset, processing over 500,000 emails and applying Natural Language Processing (NLP) techniques for text feature extraction. The models were evaluated on multiple metrics to find the most accurate solution.

-----

Tools & Technologies
- Python 3

- Libraries: Pandas, NumPy, scikit-learn, NLTK, Matplotlib, Seaborn

- Techniques: TF-IDF Vectorization, Machine Learning (Naïve Bayes, SVM, KNN), Hyperparameter Tuning

- Dataset: Enron Email Dataset (from Kaggle)


------

Approach
I followed the CRISP-DM methodology to structure the project. Key steps included:

- Data Cleaning: Removed metadata, irrelevant content, and applied stopword removal to focus on email body text.

- Feature Engineering: Used TF-IDF to convert email text into numerical features.

- Model Selection: Implemented and tuned models like Naïve Bayes, SVM, and KNN to predict spam emails based on text features.

- Evaluation: Used accuracy, precision, recall, and F1-score to assess model performance. SVM performed best, achieving 98% accuracy.

----


Challenges & Solutions
- Dataset Imbalance: Handled this by using stratified splitting and emphasing recall to ensure spam emails were not missed.

- Noisy Data: Cleaned the raw email data by removing signatures, headers, and non-relevant text.

- Model Performance: After experimenting with several models, the SVM model emerged as the top performer, balancing both speed and accuracy.


-----

Results
The SVM model achieved an impressive 98% accuracy and showed excellent precision and recall, making it highly effective at detecting spam while minimising false positives. The Naïve Bayes model also performed well, serving as a fast baseline.


----

How to Run the Project
Clone the repository:
- git clone https://github.com/KirubelCode/Email-Spam-Detection-CA1.git
- cd Email-Spam-Detection-CA1

- Run the Jupyter notebook to execute the projects Individually:
python x.ipynb

