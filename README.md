\# 🎭 Sentiment Analysis Web Application



!\[Python](https://img.shields.io/badge/Python-3.8+-blue.svg)

!\[Streamlit](https://img.shields.io/badge/Streamlit-1.30.0-red.svg)

!\[scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)

!\[License](https://img.shields.io/badge/License-MIT-green.svg)



A machine learning web application that analyzes sentiment in text reviews using Natural Language Processing (NLP) and provides real-time predictions with confidence scores.



\## 🌟 Features



\- \*\*Real-time Sentiment Analysis\*\*: Analyze text instantly with high accuracy

\- \*\*Batch Processing\*\*: Upload CSV files to analyze multiple reviews at once

\- \*\*Interactive Dashboard\*\*: Beautiful visualizations with Plotly

\- \*\*Confidence Scores\*\*: Get probability distributions for predictions

\- \*\*Text Statistics\*\*: View detailed metrics about your input

\- \*\*Export Results\*\*: Download analysis results as CSV



\## 🚀 Demo



!\[App Demo](demo/screenshot.png)



\*Live Demo: \[Your Deployed App Link]\*



\## 📊 Model Performance



| Model | Accuracy | Precision | Recall | F1-Score |

|-------|----------|-----------|--------|----------|

| Logistic Regression | 88.5% | 0.89 | 0.88 | 0.88 |

| Naive Bayes | 85.2% | 0.85 | 0.85 | 0.85 |

| Linear SVM | 89.1% | 0.89 | 0.89 | 0.89 |

| Random Forest | 84.8% | 0.85 | 0.85 | 0.85 |



\*\*Best Model\*\*: Linear SVM with 89.1% accuracy



\## 🛠️ Technologies Used



\### Machine Learning \& NLP

\- \*\*scikit-learn\*\*: Model training and evaluation

\- \*\*NLTK\*\*: Natural language processing

\- \*\*TF-IDF\*\*: Feature extraction



\### Web Development

\- \*\*Streamlit\*\*: Interactive web interface

\- \*\*Plotly\*\*: Data visualizations

\- \*\*Pandas\*\*: Data manipulation



\### Dataset

\- \*\*IMDB Movie Reviews\*\*: 50,000 labeled reviews (25k train, 25k test)



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

```



\## 🔧 Installation



\### Prerequisites

\- Python 3.8 or higher

\- pip package manager



\### Setup



1\. \*\*Clone the repository\*\*

```bash

git clone https://github.com/yourusername/sentiment-analysis-app.git

cd sentiment-analysis-app

```



2\. \*\*Create a virtual environment\*\* (recommended)

```bash

python -m venv venv

source venv/bin/activate  # On Windows: venv\\Scripts\\activate

```



3\. \*\*Install dependencies\*\*

```bash

pip install -r requirements.txt

```



4\. \*\*Download NLTK data\*\*

```python

import nltk

nltk.download('stopwords')

nltk.download('wordnet')

nltk.download('omw-1.4')

```



5\. \*\*Train the model\*\* (if not already trained)

```bash

\# Run the notebooks in order:

jupyter notebook notebooks/01\_data\_loading\_and\_eda.ipynb

jupyter notebook notebooks/02\_model\_training.ipynb

```



\## 🎯 Usage



\### Run the Web Application



```bash

streamlit run app.py

```



The app will open in your browser at `http://localhost:8501`



\### Single Text Analysis



1\. Navigate to the "Single Analysis" tab

2\. Enter your review text or select a sample

3\. Click "Analyze Sentiment"

4\. View results with confidence scores and visualizations



\### Batch Analysis



1\. Navigate to the "Batch Analysis" tab

2\. Download the sample CSV or upload your own

3\. Click "Analyze All Reviews"

4\. View aggregate statistics and download results



\## 📈 Model Training Process



1\. \*\*Data Collection\*\*: IMDB dataset with 50,000 movie reviews

2\. \*\*Data Preprocessing\*\*: 

&nbsp;  - Text cleaning (HTML removal, lowercase conversion)

&nbsp;  - Tokenization

&nbsp;  - Stopword removal

&nbsp;  - Lemmatization

3\. \*\*Feature Extraction\*\*: TF-IDF vectorization (5000 features, unigrams + bigrams)

4\. \*\*Model Training\*\*: Tested 4 different algorithms

5\. \*\*Evaluation\*\*: Best model selected based on accuracy and F1-score



\## 🧪 Example Predictions



```python

\# Positive Review

"This movie was absolutely fantastic! The acting was superb."

→ Sentiment: Positive (Confidence: 94.2%)



\# Negative Review

"Terrible waste of time. The worst movie I've ever seen."

→ Sentiment: Negative (Confidence: 96.8%)

```



\## 📊 Visualizations



The app provides:

\- Sentiment distribution pie charts

\- Confidence gauge meters

\- Probability bar charts

\- Text statistics dashboards

\- Word clouds (in EDA notebook)



\## 🚀 Deployment



\### Deploy to Streamlit Cloud (Free)



1\. Push your code to GitHub

2\. Go to \[share.streamlit.io](https://share.streamlit.io)

3\. Connect your GitHub repository

4\. Deploy with one click!



\## 🤝 Contributing



Contributions are welcome! Please feel free to submit a Pull Request.



1\. Fork the project

2\. Create your feature branch (`git checkout -b feature/AmazingFeature`)

3\. Commit your changes (`git commit -m 'Add some AmazingFeature'`)

4\. Push to the branch (`git push origin feature/AmazingFeature`)

5\. Open a Pull Request



\## 📝 License



This project is licensed under the MIT License - see the \[LICENSE](LICENSE) file for details.



\## 👨‍💻 Author



\*\*Your Name\*\*

\- GitHub: \[@yourusername](https://github.com/yourusername)

\- LinkedIn: \[Your Profile](https://linkedin.com/in/yourprofile)

\- Email: your.email@example.com



\## 🙏 Acknowledgments



\- IMDB for the movie review dataset

\- Streamlit for the amazing web framework

\- scikit-learn community for ML tools



\## 📞 Contact



For any queries or suggestions, feel free to reach out!



---



⭐ If you found this project helpful, please consider giving it a star!



\*\*Made with ❤️ and Python\*\*

