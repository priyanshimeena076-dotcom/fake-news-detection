import streamlit as st
import pandas as pd
import numpy as np
import re
import os

# Import ML libraries with error handling
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
except ImportError as e:
    st.error(f"scikit-learn not found: {e}")
    st.stop()

try:
    from textblob import TextBlob
except ImportError as e:
    st.error(f"TextBlob not found: {e}")
    st.stop()

try:
    import nltk
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        print("Downloading NLTK stopwords...")
        nltk.download('stopwords', quiet=True)
    
    # Download additional NLTK data for better sentiment analysis
    try:
        nltk.data.find('corpora/brown')
    except LookupError:
        print("Downloading NLTK brown corpus...")
        nltk.download('brown', quiet=True)
    
    try:
        nltk.data.find('corpora/movie_reviews')
    except LookupError:
        print("Downloading NLTK movie reviews...")
        nltk.download('movie_reviews', quiet=True)
    
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
except ImportError as e:
    st.warning(f"NLTK not found: {e}. Some features may be limited.")
    stopwords = None
    word_tokenize = None

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    st.warning("WordCloud not available. Word cloud features will be disabled.")
    WORDCLOUD_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    st.warning("Matplotlib not available. Some visualizations will be disabled.")
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    st.warning("Plotly not available. Interactive charts will be disabled.")
    PLOTLY_AVAILABLE = False

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression()
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def train_model(self, texts, labels):
        """Train the fake news detection model"""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Vectorize texts
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Train model
        self.model.fit(X, labels)
        self.is_trained = True
        
        return self.model.score(X, labels)
    
    def predict(self, text):
        """Predict if news is fake or real"""
        if not self.is_trained:
            return None, None
            
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        return prediction, probability
    
    def explain_prediction(self, text):
        """Explain why the model made this prediction"""
        if not self.is_trained:
            return "Model not trained yet"
            
        processed_text = self.preprocess_text(text)
        X = self.vectorizer.transform([processed_text])
        
        # Get feature names and their coefficients
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = self.model.coef_[0]
        
        # Get the features present in this text
        feature_indices = X.nonzero()[1]
        
        # Calculate feature contributions
        contributions = []
        for idx in feature_indices:
            word = feature_names[idx]
            coef = coefficients[idx]
            tfidf_score = X[0, idx]
            contribution = coef * tfidf_score
            contributions.append((word, contribution, coef))
        
        # Sort by absolute contribution
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return contributions[:10]  # Top 10 contributing words

class SentimentAnalyzer:
    def __init__(self):
        pass
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using TextBlob"""
        if not text or len(text.strip()) == 0:
            return {
                'sentiment': 'Neutral',
                'polarity': 0.0,
                'subjectivity': 0.0,
                'confidence': 0.0,
                'explanation': 'No text provided'
            }
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Classify sentiment with more nuanced thresholds
            if polarity > 0.3:
                sentiment = "Very Positive"
            elif polarity > 0.1:
                sentiment = "Positive"
            elif polarity > -0.1:
                sentiment = "Neutral"
            elif polarity > -0.3:
                sentiment = "Negative"
            else:
                sentiment = "Very Negative"
            
            # Create explanation
            explanation = self._create_explanation(polarity, subjectivity)
            
            return {
                'sentiment': sentiment,
                'polarity': polarity,
                'subjectivity': subjectivity,
                'confidence': abs(polarity),
                'explanation': explanation
            }
        except Exception as e:
            return {
                'sentiment': 'Error',
                'polarity': 0.0,
                'subjectivity': 0.0,
                'confidence': 0.0,
                'explanation': f'Analysis error: {str(e)}'
            }
    
    def _create_explanation(self, polarity, subjectivity):
        """Create human-readable explanation of sentiment scores"""
        explanations = []
        
        # Polarity explanation
        if polarity > 0.5:
            explanations.append("Very positive language detected")
        elif polarity > 0.1:
            explanations.append("Positive language detected")
        elif polarity < -0.5:
            explanations.append("Very negative language detected")
        elif polarity < -0.1:
            explanations.append("Negative language detected")
        else:
            explanations.append("Neutral or balanced language")
        
        # Subjectivity explanation
        if subjectivity > 0.7:
            explanations.append("highly subjective (opinion-based)")
        elif subjectivity > 0.4:
            explanations.append("moderately subjective")
        elif subjectivity > 0.2:
            explanations.append("somewhat objective")
        else:
            explanations.append("very objective (fact-based)")
        
        return "; ".join(explanations)
    
    def get_emotion_breakdown(self, text):
        """Get detailed emotion analysis"""
        if not text or len(text.strip()) == 0:
            return {'neutral': 1.0}
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Enhanced emotion mapping
            emotions = {
                'joy': max(0, polarity * 0.8) if polarity > 0.2 else 0,
                'anger': max(0, -polarity * 0.7) if polarity < -0.3 else 0,
                'sadness': max(0, -polarity * 0.6) if polarity < -0.1 and subjectivity > 0.3 else 0,
                'fear': max(0, -polarity * 0.5) if polarity < -0.2 and subjectivity > 0.7 else 0,
                'surprise': subjectivity * 0.4 if abs(polarity) > 0.3 else 0,
                'trust': max(0, polarity * 0.3) if polarity > 0 and subjectivity < 0.5 else 0,
                'anticipation': subjectivity * 0.3 if polarity > 0.1 else 0,
                'disgust': max(0, -polarity * 0.4) if polarity < -0.4 else 0,
                'neutral': max(0.1, 1 - abs(polarity) - subjectivity * 0.5)
            }
            
            # Normalize emotions to sum to 1
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v/total for k, v in emotions.items()}
            else:
                emotions = {'neutral': 1.0}
            
            return emotions
        except Exception:
            return {'neutral': 1.0}

def load_sample_data():
    """Create sample fake news dataset"""
    fake_news = [
        "Scientists discover that drinking water causes cancer in 99% of cases",
        "Breaking: Aliens have landed and are working with the government",
        "Miracle cure discovered that doctors don't want you to know about",
        "Local man discovers one weird trick that makes him millions overnight",
        "Government secretly controlling weather with hidden technology",
        "Vaccines contain microchips for mind control, study reveals",
        "Celebrity death hoax spreads rapidly on social media",
        "Fake miracle diet pill promises 50 pounds weight loss in one week"
    ]
    
    real_news = [
        "Stock market shows steady growth amid economic recovery efforts",
        "New research published in medical journal shows promising results",
        "Local community comes together to support flood victims",
        "Technology company announces quarterly earnings report",
        "University researchers develop new sustainable energy solution",
        "City council approves budget for infrastructure improvements",
        "Weather forecast predicts mild temperatures for the weekend",
        "Sports team wins championship after intense playoff series"
    ]
    
    texts = fake_news + real_news
    labels = [1] * len(fake_news) + [0] * len(real_news)  # 1 for fake, 0 for real
    
    return texts, labels

def main():
    st.set_page_config(
        page_title="AI News & Sentiment Analyzer",
        page_icon="🔍",
        layout="wide"
    )
    
    # Hide Streamlit style elements
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    .stDecoration {display: none;}
    .stToolbar {display: none;}
    #stDecoration {display: none;}
    button[title="View fullscreen"]{visibility: hidden;}
    .viewerBadge_container__1QSob {display: none;}
    .styles_viewerBadge__1yB5_ {display: none;}
    #viewerBadge_container__1QSob {display: none;}
    .viewerBadge_link__1S137 {display: none;}
    .viewerBadge_text__1JaDK {display: none;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    
    st.title("🔍 AI News & Sentiment Analyzer")
    st.markdown("**Detect fake news and analyze sentiment with explainable AI**")
    
    # Initialize models
    if 'fake_news_detector' not in st.session_state:
        st.session_state.fake_news_detector = FakeNewsDetector()
        st.session_state.sentiment_analyzer = SentimentAnalyzer()
        
        # Train with sample data
        texts, labels = load_sample_data()
        accuracy = st.session_state.fake_news_detector.train_model(texts, labels)
        st.session_state.model_accuracy = accuracy
    
    # Sidebar
    st.sidebar.title("📊 Model Info")
    st.sidebar.metric("Model Accuracy", f"{st.session_state.model_accuracy:.2%}")
    st.sidebar.info("This demo uses a simple model trained on sample data. For production use, train with larger, real datasets.")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["🔍 Analyze Text", "📈 Batch Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Single Text Analysis")
        
        # Text input
        user_text = st.text_area(
            "Enter news article or text to analyze:",
            height=150,
            placeholder="Paste your news article or any text here..."
        )
        
        if st.button("🔍 Analyze", type="primary"):
            if user_text.strip():
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🚨 Fake News Detection")
                    
                    # Fake news prediction
                    prediction, probability = st.session_state.fake_news_detector.predict(user_text)
                    
                    if prediction is not None:
                        is_fake = prediction == 1
                        confidence = probability[1] if is_fake else probability[0]
                        
                        # Display result
                        if is_fake:
                            st.error(f"⚠️ **LIKELY FAKE NEWS** (Confidence: {confidence:.2%})")
                        else:
                            st.success(f"✅ **LIKELY REAL NEWS** (Confidence: {confidence:.2%})")
                        
                        # Probability chart
                        if PLOTLY_AVAILABLE:
                            # Create a more informative probability chart
                            real_prob = probability[0]
                            fake_prob = probability[1]
                            
                            fig = go.Figure()
                            
                            # Add bars with better styling
                            fig.add_trace(go.Bar(
                                x=['Real News', 'Fake News'],
                                y=[real_prob, fake_prob],
                                marker_color=['#28a745', '#dc3545'],  # Green for real, red for fake
                                text=[f'{real_prob:.1%}', f'{fake_prob:.1%}'],
                                textposition='auto',
                                textfont=dict(color='white', size=14, family='Arial Black'),
                                hovertemplate='<b>%{x}</b><br>Probability: %{y:.1%}<extra></extra>'
                            ))
                            
                            # Update layout for better appearance
                            fig.update_layout(
                                title={
                                    'text': 'Fake News Detection Probabilities',
                                    'x': 0.5,
                                    'xanchor': 'center',
                                    'font': {'size': 16, 'family': 'Arial'}
                                },
                                yaxis_title="Probability",
                                yaxis=dict(
                                    range=[0, 1],
                                    tickformat='.0%',
                                    gridcolor='lightgray',
                                    gridwidth=1
                                ),
                                xaxis=dict(
                                    tickfont={'size': 12}
                                ),
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                height=400,
                                margin=dict(l=50, r=50, t=80, b=50),
                                showlegend=False
                            )
                            
                            # Add a threshold line at 50%
                            fig.add_hline(
                                y=0.5, 
                                line_dash="dash", 
                                line_color="gray",
                                annotation_text="Decision Threshold (50%)",
                                annotation_position="top right"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add interpretation
                            if fake_prob > 0.7:
                                st.error("🚨 **High Risk**: Strong indicators of fake news detected!")
                            elif fake_prob > 0.5:
                                st.warning("⚠️ **Medium Risk**: Some fake news indicators present.")
                            else:
                                st.success("✅ **Low Risk**: Appears to be legitimate news.")
                                
                        else:
                            # Fallback for when Plotly is not available
                            st.write("**Prediction Probabilities:**")
                            col_real, col_fake = st.columns(2)
                            with col_real:
                                st.metric("Real News", f"{probability[0]:.1%}")
                            with col_fake:
                                st.metric("Fake News", f"{probability[1]:.1%}")
                        
                        # Explanation
                        st.subheader("🧠 Why this prediction?")
                        contributions = st.session_state.fake_news_detector.explain_prediction(user_text)
                        
                        if contributions:
                            # Create explanation dataframe with better formatting
                            explanation_data = []
                            for word, contribution, coef in contributions:
                                impact = 'Fake indicator' if contribution > 0 else 'Real indicator'
                                strength = abs(contribution)
                                explanation_data.append({
                                    'Word': word,
                                    'Impact': impact,
                                    'Influence': f"{contribution:.4f}",
                                    'Strength': strength
                                })
                            
                            explanation_df = pd.DataFrame(explanation_data)
                            
                            # Display as a styled dataframe
                            st.dataframe(
                                explanation_df[['Word', 'Impact', 'Influence']], 
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Show top contributing words as tags
                            st.write("**Key Words Influencing Decision:**")
                            
                            # Separate positive and negative contributors
                            fake_words = [item for item in explanation_data if item['Impact'] == 'Fake indicator'][:3]
                            real_words = [item for item in explanation_data if item['Impact'] == 'Real indicator'][:3]
                            
                            if fake_words:
                                st.write("🔴 **Fake News Indicators:**")
                                fake_tags = " ".join([f"`{word['Word']}`" for word in fake_words])
                                st.markdown(fake_tags)
                            
                            if real_words:
                                st.write("🟢 **Real News Indicators:**")
                                real_tags = " ".join([f"`{word['Word']}`" for word in real_words])
                                st.markdown(real_tags)
                        
                        # Add confidence visualization
                        overall_confidence = max(probability[0], probability[1])
                        st.subheader("📊 Model Confidence")
                        
                        if PLOTLY_AVAILABLE:
                            # Create a gauge chart for confidence
                            fig_gauge = go.Figure(go.Indicator(
                                mode = "gauge+number+delta",
                                value = overall_confidence * 100,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Confidence Level (%)"},
                                delta = {'reference': 50},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "darkblue"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 80], 'color': "yellow"},
                                        {'range': [80, 100], 'color': "green"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': 90
                                    }
                                }
                            ))
                            fig_gauge.update_layout(height=300)
                            st.plotly_chart(fig_gauge, use_container_width=True)
                        else:
                            st.progress(overall_confidence)
                            st.write(f"Confidence: {overall_confidence:.1%}")
                
                with col2:
                    st.subheader("😊 Sentiment Analysis")
                    
                    # Sentiment analysis
                    sentiment_result = st.session_state.sentiment_analyzer.analyze_sentiment(user_text)
                    
                    # Display sentiment with better formatting
                    sentiment_color = {
                        'Very Positive': 'green',
                        'Positive': 'green',
                        'Negative': 'red',
                        'Very Negative': 'red',
                        'Neutral': 'gray'
                    }
                    
                    sentiment_emoji = {
                        'Very Positive': '😄',
                        'Positive': '😊',
                        'Negative': '😞',
                        'Very Negative': '😡',
                        'Neutral': '😐'
                    }
                    
                    emoji = sentiment_emoji.get(sentiment_result['sentiment'], '😐')
                    color = sentiment_color.get(sentiment_result['sentiment'], 'gray')
                    
                    st.markdown(f"**Overall Sentiment:** {emoji} :{color}[{sentiment_result['sentiment']}]")
                    
                    # Create columns for metrics
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        st.metric(
                            "Polarity", 
                            f"{sentiment_result['polarity']:.3f}",
                            help="Range: -1 (very negative) to +1 (very positive)"
                        )
                    
                    with metric_col2:
                        st.metric(
                            "Subjectivity", 
                            f"{sentiment_result['subjectivity']:.3f}",
                            help="Range: 0 (objective/factual) to 1 (subjective/opinion)"
                        )
                    
                    # Show explanation
                    if 'explanation' in sentiment_result:
                        st.info(f"**Analysis:** {sentiment_result['explanation']}")
                    
                    # Confidence indicator
                    confidence = sentiment_result.get('confidence', 0)
                    if confidence > 0.5:
                        st.success(f"High confidence: {confidence:.2%}")
                    elif confidence > 0.2:
                        st.warning(f"Medium confidence: {confidence:.2%}")
                    else:
                        st.info(f"Low confidence: {confidence:.2%}")
                    
                    # Emotion breakdown
                    emotions = st.session_state.sentiment_analyzer.get_emotion_breakdown(user_text)
                    
                    # Emotion chart
                    if PLOTLY_AVAILABLE:
                        emotion_df = pd.DataFrame(list(emotions.items()), columns=['Emotion', 'Score'])
                        # Filter out very small values for cleaner chart
                        emotion_df = emotion_df[emotion_df['Score'] > 0.05]
                        if not emotion_df.empty:
                            fig = px.pie(emotion_df, values='Score', names='Emotion', title="Emotion Breakdown")
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.write("**Emotion Breakdown:**")
                        for emotion, score in emotions.items():
                            if score > 0.05:  # Only show significant emotions
                                st.write(f"- {emotion.title()}: {score:.2%}")
                    
                    # Word cloud
                    if len(user_text.split()) > 5 and WORDCLOUD_AVAILABLE and MATPLOTLIB_AVAILABLE:
                        st.subheader("☁️ Word Cloud")
                        try:
                            wordcloud = WordCloud(
                                width=400, 
                                height=200, 
                                background_color='white',
                                colormap='viridis',
                                max_words=50
                            ).generate(user_text)
                            
                            fig, ax = plt.subplots(figsize=(8, 4))
                            ax.imshow(wordcloud, interpolation='bilinear')
                            ax.axis('off')
                            st.pyplot(fig)
                        except Exception as e:
                            st.warning(f"Could not generate word cloud: {str(e)}")
                
                # Add a summary section
                st.markdown("---")
                st.subheader("📋 Analysis Summary")
                
                # Create summary columns
                sum_col1, sum_col2, sum_col3 = st.columns(3)
                
                with sum_col1:
                    prediction_text = "LIKELY FAKE" if prediction == 1 else "LIKELY REAL"
                    prediction_color = "🔴" if prediction == 1 else "🟢"
                    st.metric(
                        "Final Verdict", 
                        f"{prediction_color} {prediction_text}",
                        f"{max(probability):.1%} confidence"
                    )
                
                with sum_col2:
                    sentiment_emoji = {
                        'Very Positive': '😄', 'Positive': '😊', 'Negative': '😞', 
                        'Very Negative': '😡', 'Neutral': '😐'
                    }
                    emoji = sentiment_emoji.get(sentiment_result['sentiment'], '😐')
                    st.metric(
                        "Sentiment", 
                        f"{emoji} {sentiment_result['sentiment']}",
                        f"{sentiment_result['polarity']:.2f} polarity"
                    )
                
                with sum_col3:
                    word_count = len(user_text.split())
                    reading_time = max(1, word_count // 200)
                    st.metric(
                        "Text Stats", 
                        f"{word_count} words",
                        f"~{reading_time} min read"
                    )
                
                # Recommendations
                st.subheader("💡 Recommendations")
                if prediction == 1:  # Fake news
                    st.warning("""
                    **⚠️ This content shows signs of potentially fake news. Consider:**
                    - Verify information through multiple reliable sources
                    - Check the original source and publication date
                    - Look for supporting evidence and citations
                    - Be cautious about sharing without verification
                    """)
                else:  # Real news
                    if max(probability) < 0.7:
                        st.info("""
                        **ℹ️ While this appears to be legitimate news, always:**
                        - Cross-reference with other reputable sources
                        - Check for recent updates or corrections
                        - Consider the source's credibility and bias
                        """)
                    else:
                        st.success("""
                        **✅ This content appears to be legitimate news, but remember:**
                        - Stay informed by reading multiple perspectives
                        - Verify important claims independently
                        - Consider the publication date and context
                        """)
            else:
                st.warning("Please enter some text to analyze.")
    
    with tab2:
        st.header("📈 Batch Analysis")
        st.info("Upload a CSV file with a 'text' column for batch analysis")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                
                if 'text' in df.columns:
                    st.success(f"Loaded {len(df)} texts for analysis")
                    
                    if st.button("🚀 Analyze All"):
                        progress_bar = st.progress(0)
                        results = []
                        
                        for i, text in enumerate(df['text']):
                            # Fake news detection
                            prediction, probability = st.session_state.fake_news_detector.predict(str(text))
                            
                            # Sentiment analysis
                            sentiment = st.session_state.sentiment_analyzer.analyze_sentiment(str(text))
                            
                            results.append({
                                'text': text[:100] + '...' if len(str(text)) > 100 else text,
                                'is_fake': prediction == 1 if prediction is not None else None,
                                'fake_confidence': probability[1] if prediction is not None else None,
                                'sentiment': sentiment['sentiment'],
                                'polarity': sentiment['polarity']
                            })
                            
                            progress_bar.progress((i + 1) / len(df))
                        
                        results_df = pd.DataFrame(results)
                        
                        # Display results
                        st.subheader("📊 Analysis Results")
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Summary statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            fake_count = results_df['is_fake'].sum() if results_df['is_fake'].notna().any() else 0
                            st.metric("Fake News Detected", fake_count)
                        
                        with col2:
                            positive_count = (results_df['sentiment'] == 'Positive').sum()
                            st.metric("Positive Sentiments", positive_count)
                        
                        with col3:
                            avg_polarity = results_df['polarity'].mean()
                            st.metric("Average Polarity", f"{avg_polarity:.3f}")
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results",
                            data=csv,
                            file_name="analysis_results.csv",
                            mime="text/csv"
                        )
                else:
                    st.error("CSV file must contain a 'text' column")
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
    
    with tab3:
        st.header("ℹ️ About This Project")
        
        st.markdown("""
        ### 🎯 What This App Does
        
        This AI-powered application combines two powerful text analysis capabilities:
        
        1. **Fake News Detection**: Uses machine learning to identify potentially fake or misleading news articles
        2. **Sentiment Analysis**: Analyzes the emotional tone and sentiment of any text
        
        ### 🔧 How It Works
        
        **Fake News Detection:**
        - Uses TF-IDF vectorization to convert text into numerical features
        - Employs Logistic Regression for classification
        - Provides explainable AI - shows which words influenced the decision
        
        **Sentiment Analysis:**
        - Uses TextBlob for polarity and subjectivity analysis
        - Provides emotion breakdown (joy, anger, sadness, etc.)
        - Generates word clouds for visual text representation
        
        ### 🚀 Features
        
        - **Real-time Analysis**: Instant results for any text input
        - **Explainable AI**: Understand why the model made its decision
        - **Batch Processing**: Analyze multiple texts from CSV files
        - **Visual Analytics**: Charts, graphs, and word clouds
        - **Export Results**: Download analysis results as CSV
        
        ### 🛠️ Tech Stack
        
        - **Frontend**: Streamlit
        - **ML Libraries**: scikit-learn, TextBlob, NLTK
        - **Visualization**: Plotly, Matplotlib, WordCloud
        - **Data Processing**: Pandas, NumPy
        
        ### ⚠️ Important Notes
        
        - This is a demo using a simple model trained on sample data
        - For production use, train with larger, real-world datasets
        - Always verify important news through multiple reliable sources
        - Sentiment analysis is subjective and context-dependent
        
        ### 🔮 Future Enhancements
        
        - Browser extension for real-time web page analysis
        - Integration with social media APIs
        - Advanced deep learning models (BERT, GPT)
        - Multi-language support
        - Real-time news feed monitoring
        """)

if __name__ == "__main__":
    main()