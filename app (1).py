"""
Sentiment Analysis Web Application
A machine learning web app for analyzing sentiment in text reviews
Author: Mythri Muthyala & Raksha Muthyala
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def setup_nltk():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

setup_nltk()





# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .positive-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .negative-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .neutral-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Download NLTK data (only once)
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)

download_nltk_data()

# Load model and vectorizer

def load_models():
    model_path = "models/best_model.pkl"
    vectorizer_path = "models/tfidf_vectorizer.pkl"

    if not os.path.exists(model_path):
        st.error("❌ best_model.pkl not found")
        st.stop()

    if not os.path.exists(vectorizer_path):
        st.error("❌ tfidf_vectorizer.pkl not found")
        st.stop()

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # HARD SAFETY CHECK
    if not hasattr(vectorizer, "idf_"):
        st.error("❌ TF-IDF vectorizer is NOT fitted. Cached object detected.")
        st.stop()

    return model, vectorizer


model, vectorizer = load_models()

# Text preprocessing functions
lemmatizer = WordNetLemmatizer()

@st.cache_resource
def load_stopwords():
    return set(stopwords.words('english'))

stop_words = load_stopwords()

def clean_text(text):
    """Clean and preprocess text"""
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

def predict_sentiment(text):
    """Predict sentiment for given text"""
    cleaned = clean_text(text)
    text_tfidf = vectorizer.transform([cleaned])
    prediction = model.predict(text_tfidf)[0]
    
    # Get probability if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(text_tfidf)[0]
        confidence = max(proba) * 100
        positive_prob = proba[1] * 100
        negative_prob = proba[0] * 100
    else:
        # For models without predict_proba (like LinearSVC)
        confidence = None
        positive_prob = None
        negative_prob = None
    
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    
    return sentiment, confidence, positive_prob, negative_prob

# ============================================================================
# MAIN APP
# ============================================================================

# Header
st.markdown('<div class="main-header">🎭 Sentiment Analysis App</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze the sentiment of movie reviews using Machine Learning</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 About")
    st.info("""
    This app uses Machine Learning to analyze sentiment in text reviews.
    
    **Features:**
    - Single text analysis
    - Batch CSV upload
    - Confidence scores
    - Visual analytics
    
    **Model:** Trained on 25,000 IMDB movie reviews
    """)
    
    st.header("🎯 How to Use")
    st.markdown("""
    1. Choose analysis mode (Single/Batch)
    2. Enter or upload your text
    3. Click 'Analyze'
    4. View results and insights
    """)
    
    st.header("👨‍💻 Developer")
    st.markdown("""
    **Mythri M & Raksha M**  
    AI/ML Student  
    [GitHub](https://github.com/RakshaMuthyala07) | [LinkedIn](https://www.linkedin.com/in/mythri-m-25133128b?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📝 Single Analysis", "📊 Batch Analysis", "ℹ️ Model Info"])

# ============================================================================
# TAB 1: SINGLE TEXT ANALYSIS
# ============================================================================

with tab1:
    st.header("Analyze Single Review")
    
    # Sample reviews for quick testing
    sample_reviews = {
        "Select a sample...": "",
        "Positive Review": "This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. Highly recommended!",
        "Negative Review": "Terrible waste of time. The plot was predictable and the acting was wooden. One of the worst movies I've seen.",
        "Mixed Review": "The movie had great visuals but the story was lacking. Some good moments but overall disappointing."
    }
    
    selected_sample = st.selectbox("Quick Test (Optional):", list(sample_reviews.keys()))
    
    # Text input
    user_input = st.text_area(
        "Enter your review here:",
        value=sample_reviews[selected_sample],
        height=150,
        placeholder="Type or paste your movie review here..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 3])
    
    with col1:
        analyze_button = st.button("🔍 Analyze Sentiment", type="primary")
    
    with col2:
        clear_button = st.button("🗑️ Clear")
    
    if clear_button:
        st.rerun()
    
    if analyze_button:
        if user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence, pos_prob, neg_prob = predict_sentiment(user_input)
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Analysis Results")
                
                # Create columns for layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Sentiment result box
                    if sentiment == "Positive":
                        st.markdown(f"""
                        <div class="positive-box">
                            <h2 style="color: #28a745; margin: 0;">✅ Positive Sentiment</h2>
                            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">This review expresses a positive opinion!</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="negative-box">
                            <h2 style="color: #dc3545; margin: 0;">❌ Negative Sentiment</h2>
                            <p style="margin: 0.5rem 0 0 0; font-size: 1.1rem;">This review expresses a negative opinion.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Confidence score
                    if confidence:
                        st.metric("Confidence Score", f"{confidence:.2f}%")
                
                with col2:
                    # Gauge chart for confidence
                    if confidence:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=confidence,
                            title={'text': "Confidence"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#28a745" if sentiment == "Positive" else "#dc3545"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgray"},
                                    {'range': [50, 75], 'color': "gray"},
                                    {'range': [75, 100], 'color': "darkgray"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 90
                                }
                            }
                        ))
                        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig, use_container_width=True)
                
                # Probability distribution
                if pos_prob and neg_prob:
                    st.markdown("---")
                    st.subheader("📈 Sentiment Distribution")
                    
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Negative', 'Positive'],
                            y=[neg_prob, pos_prob],
                            marker_color=['#dc3545', '#28a745'],
                            text=[f'{neg_prob:.1f}%', f'{pos_prob:.1f}%'],
                            textposition='auto',
                        )
                    ])
                    fig.update_layout(
                        title="Probability Distribution",
                        yaxis_title="Probability (%)",
                        height=400,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Text statistics
                st.markdown("---")
                st.subheader("📝 Text Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Word Count", len(user_input.split()))
                with col2:
                    st.metric("Character Count", len(user_input))
                with col3:
                    st.metric("Sentences", user_input.count('.') + user_input.count('!') + user_input.count('?'))
                with col4:
                    avg_word_length = np.mean([len(word) for word in user_input.split()])
                    st.metric("Avg Word Length", f"{avg_word_length:.1f}")
        else:
            st.warning("⚠️ Please enter some text to analyze!")

# ============================================================================
# TAB 2: BATCH ANALYSIS
# ============================================================================

with tab2:
    st.header("Batch Analysis from CSV")
    
    st.info("""
    📤 Upload a CSV file with a column named 'text' or 'review' containing the reviews you want to analyze.
    The app will analyze all reviews and provide aggregate statistics.
    """)
    
    # Sample CSV download
    sample_data = pd.DataFrame({
        'text': [
            "This movie was amazing! Best film of the year.",
            "Terrible plot and bad acting. Very disappointed.",
            "Good movie but could have been better.",
            "Absolutely loved it! Will watch again.",
            "Boring and predictable. Not worth the time."
        ]
    })
    
    csv = sample_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample CSV",
        data=csv,
        file_name="sample_reviews.csv",
        mime="text/csv"
    )
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Find text column
            text_column = None
            for col in ['text', 'review', 'Review', 'Text', 'comment', 'Comment']:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column is None:
                st.error("❌ CSV must have a column named 'text', 'review', or 'comment'")
            else:
                st.success(f"✅ Found {len(df)} reviews in column '{text_column}'")
                
                if st.button("🚀 Analyze All Reviews", type="primary"):
                    with st.spinner("Analyzing all reviews... This may take a moment."):
                        # Analyze all reviews
                        results = []
                        for text in df[text_column]:
                            sentiment, confidence, pos_prob, neg_prob = predict_sentiment(str(text))
                            results.append({
                                'text': text,
                                'sentiment': sentiment,
                                'confidence': confidence if confidence else 'N/A'
                            })
                        
                        results_df = pd.DataFrame(results)
                        
                        # Display summary statistics
                        st.markdown("---")
                        st.subheader("📊 Summary Statistics")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        positive_count = (results_df['sentiment'] == 'Positive').sum()
                        negative_count = (results_df['sentiment'] == 'Negative').sum()
                        total = len(results_df)
                        
                        with col1:
                            st.metric("Total Reviews", total)
                        with col2:
                            st.metric("Positive Reviews", f"{positive_count} ({positive_count/total*100:.1f}%)")
                        with col3:
                            st.metric("Negative Reviews", f"{negative_count} ({negative_count/total*100:.1f}%)")
                        
                        # Pie chart
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig = px.pie(
                                values=[positive_count, negative_count],
                                names=['Positive', 'Negative'],
                                title='Sentiment Distribution',
                                color_discrete_map={'Positive': '#28a745', 'Negative': '#dc3545'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            sentiment_counts = results_df['sentiment'].value_counts()
                            fig = go.Figure(data=[
                                go.Bar(
                                    x=sentiment_counts.index,
                                    y=sentiment_counts.values,
                                    marker_color=['#dc3545', '#28a745'] if sentiment_counts.index[0] == 'Negative' else ['#28a745', '#dc3545'],
                                    text=sentiment_counts.values,
                                    textposition='auto',
                                )
                            ])
                            fig.update_layout(title="Sentiment Count", xaxis_title="Sentiment", yaxis_title="Count")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Display results table
                        st.markdown("---")
                        st.subheader("📋 Detailed Results")
                        st.dataframe(results_df, use_container_width=True, height=400)
                        
                        # Download results
                        csv_result = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv_result,
                            file_name="sentiment_analysis_results.csv",
                            mime="text/csv"
                        )
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")

# ============================================================================
# TAB 3: MODEL INFO
# ============================================================================

with tab3:
    st.header("ℹ️ Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Dataset")
        st.markdown("""
        - **Source:** IMDB Movie Reviews
        - **Training Samples:** 25,000
        - **Test Samples:** 25,000
        - **Classes:** Binary (Positive/Negative)
        - **Balanced:** Yes (50-50 split)
        """)
        
        st.subheader("🔧 Preprocessing")
        st.markdown("""
        - Lowercasing
        - HTML tag removal
        - URL removal
        - Special character removal
        - Stopword removal
        - Lemmatization
        """)
    
    with col2:
        st.subheader("🤖 Model Architecture")
        st.markdown(f"""
        - **Algorithm:** {type(model).__name__}
        - **Features:** TF-IDF (5000 features)
        - **N-grams:** Unigrams + Bigrams
        - **Expected Accuracy:** ~85-90%
        """)
        
        st.subheader("🎯 Use Cases")
        st.markdown("""
        - Movie review analysis
        - Product review sentiment
        - Social media monitoring
        - Customer feedback analysis
        - Brand reputation tracking
        """)
    
    st.markdown("---")
    st.subheader("🚀 Technologies Used")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Machine Learning:**
        - scikit-learn
        - NLTK
        - TF-IDF Vectorization
        """)
    
    with col2:
        st.markdown("""
        **Web Framework:**
        - Streamlit
        - Plotly
        - Pandas
        """)
    
    with col3:
        st.markdown("""
        **Visualization:**
        - Plotly Express
        - Matplotlib
        - WordCloud
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    Made with ❤️ using Python & Streamlit | © 2026 Mythri & Raksha
</div>
""", unsafe_allow_html=True)
