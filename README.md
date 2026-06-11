📦 Amazon Customer Reviews – Problem Detection (NLP Project)
📌 Project Overview

This project focuses on analyzing Amazon customer reviews and building machine learning models to detect whether a review indicates a problem (negative experience) or not.

The main goal is to apply NLP techniques and compare multiple machine learning and deep learning approaches for sentiment/problem classification.

🎯 Objectives
Clean and preprocess large-scale review text data
Perform Exploratory Data Analysis (EDA)
Create a binary classification target: problem vs non-problem
Train and compare multiple ML models
Improve performance using class balancing techniques
Apply advanced NLP model (DistilBERT fine-tuning)
📊 Dataset

The dataset contains Amazon product reviews with fields such as:

review_headline
review_body
product_category
star_rating
verified_purchase

A new target variable was created:

is_problem:
- 1 → Problematic review (negative experience)
- 0 → Non-problematic review
🧹 Data Preprocessing

Steps applied:

Merging headline + body into text
Lowercasing text
Removing special characters
Stopword removal (Scikit-learn English stopwords)
Handling missing values
df["text"] = df["review_headline"].fillna("") + " " + df["review_body"].fillna("")
📈 Exploratory Data Analysis (EDA)
Key findings:
Dataset is imbalanced
~75% non-problem reviews
~25% problem reviews
Most frequent words in problem reviews:
product-related: game, sound, battery, quality
negative sentiment: bad, broken, worst, return
Verified purchases showed slightly lower problem rates compared to non-verified ones.
🤖 Machine Learning Models

The following models were trained and compared:

✔ Logistic Regression
Strong baseline model
Good balance between precision and recall
✔ Naive Bayes
Fast but weaker recall for problem class
✔ SVM (LinearSVC)
Best classical ML performance for text data
Improved with class_weight="balanced"
✔ Random Forest
Moderate performance
Lower recall for minority class
✔ XGBoost
Competitive results
Similar performance to Random Forest
🏆 Best Model Comparison
Model	Accuracy	F1 (Problem Class)
Logistic Regression	~0.87	0.76
Naive Bayes	~0.86	0.66
SVM (LinearSVC)	~0.89	0.77
Random Forest	~0.87	0.71
XGBoost	~0.88	0.72

👉 Best classical model: LinearSVC

⚖️ Class Imbalance Handling

To improve minority class detection:

LinearSVC(class_weight="balanced")

This improved recall for problem reviews significantly.

🧠 Deep Learning Approach (Advanced)

As an advanced step, DistilBERT fine-tuning was applied.

Why DistilBERT?
Understands context better than TF-IDF models
Handles negation and complex sentences
Pretrained language model
Approach:
Tokenization using DistilBertTokenizer
Fine-tuning with HuggingFace Trainer
Binary classification head added
🔍 Error Analysis

Error analysis showed:

Most errors occur in short or ambiguous reviews
False negatives are more critical (missed problem reviews)
Product-related complaints dominate dataset
📌 Key Insights
Most complaints are related to product quality and functionality
Delivery and customer service issues are less frequent
Classical ML models perform surprisingly well on TF-IDF features
DistilBERT provides better contextual understanding but is computationally expensive
🚀 Future Improvements
Hyperparameter tuning for ML models
Larger fine-tuning dataset for DistilBERT
Aspect-based sentiment analysis (product vs delivery vs support)
Deployment as API (FastAPI / Flask)
🛠️ Tech Stack
Python 🐍
Pandas, NumPy
Scikit-learn
XGBoost
NLTK / Regex
Transformers (HuggingFace)
PyTorch
📁 Project Structure
├── data/
├── notebooks/
├── models/
├── eda/
├── README.md
📌 Author

NLP Project – Amazon Customer Reviews Classification
Focus: Problem Detection using Machine Learning & Deep Learning
