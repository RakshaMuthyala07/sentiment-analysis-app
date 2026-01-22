🎭 Sentiment Analysis Web Application



!\[Python](https://img.shields.io/badge/Python-3.8+-blue.svg)

!\[Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-red.svg)

!\[scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)

!\[License](https://img.shields.io/badge/License-MIT-green.svg)



A machine learning web application that analyzes sentiment in text reviews using Natural Language Processing (NLP) and provides real-time predictions with confidence scores.


🌟 Features



Real-time Sentiment Analysis: Analyze text instantly with high accuracy

Batch Processing: Upload CSV files to analyze multiple reviews at once

Interactive Dashboard: Beautiful visualizations with Plotly

Confidence Scores: Get probability distributions for predictions

Text Statistics: View detailed metrics about your input

Export Results: Download analysis results as CSV



Demo



!\[App Demo](demo/screenshot.png)



\*Live Demo: \[Your Deployed App Link]\*



 📊 Model Performance



| Model | Accuracy | Precision | Recall | F1-Score |


| Logistic Regression | 88.5% | 0.89 | 0.88 | 0.88 |

| Naive Bayes | 85.2% | 0.85 | 0.85 | 0.85 |

| Linear SVM | 89.1% | 0.89 | 0.89 | 0.89 |

| Random Forest | 84.8% | 0.85 | 0.85 | 0.85 |



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

├── notebooks/

│   ├── 01\_data\_loading\_and\_eda.ipynb

│   └── 02\_model\_training.ipynb

├── models/

│   ├── best\_model.pkl

│   └── tfidf\_vectorizer.pkl

├── data/

│   ├── train\_processed.csv

│   └── test\_processed.csv

├── requirements.txt

├── .gitignore

└── README.md





🔧 Installation



Prerequisites

\- Python 3.8 or higher

\- pip package manager



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




