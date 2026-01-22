🎭 Sentiment Analysis Web Application



!\[Python](https://img.shields.io/badge/Python-3.8+-blue.svg)

!\[Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-red.svg)

!\[scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)

!\[License](https://img.shields.io/badge/License-MIT-green.svg)



A machine learning web application that analyzes sentiment in text reviews using Natural Language Processing (NLP) and provides real-time predictions with confidence scores.



DEMO

App Demo:


<img width="1916" height="1109" alt="image" src="https://github.com/user-attachments/assets/db6a3e77-39d0-487b-9350-1186ccae0e70" />
<img width="1470" height="448" alt="image" src="https://github.com/user-attachments/assets/19ffae7b-4181-4fc9-ade4-aa9e8d074bb5" />
<img width="1479" height="994" alt="image" src="https://github.com/user-attachments/assets/a719d6f3-f827-4c3f-aebe-1de6b0163f74" />
<img width="1919" height="1122" alt="image" src="https://github.com/user-attachments/assets/b5433406-7556-4079-a1f3-f417decaf1af" />
<img width="1510" height="977" alt="image" src="https://github.com/user-attachments/assets/0885210f-ebda-443e-a68f-17d6d555fa47" />
<img width="1473" height="676" alt="image" src="https://github.com/user-attachments/assets/158be900-b035-4048-bc34-0b500c684db7" />



🌟 Features

Real-time Sentiment Analysis: Analyze text instantly with high accuracy

Batch Processing: Upload CSV files to analyze multiple reviews at once

Interactive Dashboard: Beautiful visualizations with Plotly

Confidence Scores: Get probability distributions for predictions

Text Statistics: View detailed metrics about your input

Export Results: Download analysis results as CSV






\*Live Demo: \[Your Deployed App Link]\*



 📊 Model Performance


<img width="989" height="336" alt="image" src="https://github.com/user-attachments/assets/f13f6118-0218-4d8e-b545-ccd3a4424d99" />




Best Model: Linear SVM with 89.1% accuracy



 🛠️ Technologies Used



 Machine Learning & NLP

scikit-learn: Model training and evaluation

NLTK: Natural language processing

TF-IDF: Feature extraction



 Web Development

Streamlit: Interactive web interface

Plotly: Data visualizations

Pandas: Data manipulation



 Dataset

IMDB Movie Reviews: 50,000 labeled reviews (25k train, 25k test)



\## 📁 Project Structure



```

sentiment-analysis-app/

├── app.py                          # Main Streamlit application

├── notebooks

│   ├── 01_data_loading_and_eda.ipynb

│   └── 02_model_training.ipynb

├── models

│   ├── best_model.pkl

│   └── tfidf_vectorizer.pkl

├── data

│   ├── train_processed.csv

│   └── test_processed.csv

├── requirements.txt

├── .gitignore

└── README.md





🔧 Installation



Prerequisites

 Python 3.8 or higher

 pip package manager



 Setup

Installation
1. Clone the repository
git clone https://github.com/RakshaMuthyala07/sentiment-analysis-app.git
cd sentiment-analysis-app

2. Create and activate a virtual environment (recommended)

python -m venv venv
venv\Scripts\activate

3. Install dependencies

pip install -r requirements.txt

4. Download NLTK data

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

5. Train the model (if not already trained)

Run the notebooks in the following order:

jupyter notebook notebooks/01_data_loading_and_eda.ipynb
jupyter notebook notebooks/02_model_training.ipynb

Usage
Run the Web Application
streamlit run app.py
Single Text Analysis

Open the "Single Analysis" tab

Enter a movie review

Click "Analyze Sentiment"

View sentiment prediction and confidence score
Batch Analysis

Open the "Batch Analysis" tab

Upload a CSV file containing reviews

Click "Analyze All Reviews"

Download the results as a CSV file

Model Training Process

Data collection using the IMDB movie reviews dataset

Text preprocessing including HTML removal, lowercasing, tokenization, stopword removal, and lemmatization

Feature extraction using TF-IDF vectorization with unigrams and bigrams

Model training using multiple machine learning algorithms

Model evaluation and selection based on accuracy and F1-score

Example Predictions

"This movie was absolutely fantastic! The acting was superb."
Sentiment: Positive

"Terrible waste of time. The worst movie I've ever seen."
Sentiment: Negative

Visualizations

The application provides sentiment distribution charts, confidence scores, probability graphs, and text statistics.
Word clouds and detailed analysis are available in the EDA notebook.

Deployment

This application can be deployed using Streamlit Community Cloud:

Push the project to GitHub

Visit https://share.streamlit.io

Connect the GitHub repository

Deploy the app

License

This project is licensed under the MIT License.
See the LICENSE file for details.

Authors

Mythri Muthyala
Raksha Muthyala

GitHub: https://github.com/RakshaMuthyala07

Acknowledgments

IMDB for the movie review dataset
Streamlit for the web framework
scikit-learn community for machine learning tools




