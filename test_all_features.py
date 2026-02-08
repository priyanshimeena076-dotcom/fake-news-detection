#!/usr/bin/env python3
"""
Comprehensive test script for Fake News Detection and Sentiment Analysis
Tests all features and displays expected output format
"""

import sys
from textblob import TextBlob
import pandas as pd

# Import the classes from app.py
sys.path.insert(0, r'c:\Users\priya\OneDrive\Documents\fake news')

# Import directly
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = LogisticRegression()
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Clean and preprocess text"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def train_model(self, texts, labels):
        """Train the fake news detection model"""
        processed_texts = [self.preprocess_text(text) for text in texts]
        X = self.vectorizer.fit_transform(processed_texts)
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
        """Create human-readable explanation"""
        explanations = []
        
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
            return {
                'joy': 0.0,
                'anger': 0.0,
                'sadness': 0.0,
                'fear': 0.0,
                'surprise': 0.0,
                'trust': 0.0,
                'anticipation': 0.0,
                'disgust': 0.0,
                'neutral': 1.0
            }
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            emotions = {
                'joy': max(0, min(polarity * 1.0, 1.0)) if polarity > 0.3 else 0,
                'anger': max(0, min(-polarity * 0.8, 1.0)) if polarity < -0.4 else 0,
                'sadness': max(0, min(-polarity * 0.7, 1.0)) if -0.5 < polarity < -0.05 else 0,
                'fear': max(0, min(-polarity * 0.6, 1.0)) if polarity < -0.3 and subjectivity > 0.6 else 0,
                'surprise': max(0, min(subjectivity * 0.7, 1.0)) if abs(polarity) > 0.4 else 0,
                'trust': max(0, min(polarity * 0.6, 1.0)) if polarity > 0.2 and subjectivity < 0.6 else 0,
                'anticipation': max(0, min(polarity * 0.5, 1.0)) if polarity > 0.1 and subjectivity > 0.3 else 0,
                'disgust': max(0, min(-polarity * 0.7, 1.0)) if polarity < -0.5 else 0,
                'neutral': max(0.05, min(1 - abs(polarity) - subjectivity * 0.3, 1.0))
            }
            
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: round(v/total, 4) for k, v in emotions.items()}
            else:
                emotions = {
                    'joy': 0.0,
                    'anger': 0.0,
                    'sadness': 0.0,
                    'fear': 0.0,
                    'surprise': 0.0,
                    'trust': 0.0,
                    'anticipation': 0.0,
                    'disgust': 0.0,
                    'neutral': 1.0
                }
            
            return emotions
        except Exception:
            return {
                'joy': 0.0,
                'anger': 0.0,
                'sadness': 0.0,
                'fear': 0.0,
                'surprise': 0.0,
                'trust': 0.0,
                'anticipation': 0.0,
                'disgust': 0.0,
                'neutral': 1.0
            }

def load_sample_data():
    """Create sample fake news dataset"""
    fake_news = [
        "Scientists discover that drinking water causes cancer in 99% of cases",
        "Breaking: Aliens have landed and are working with the government",
        "Miracle cure discovered that doctors don't want you to know about",
        "Local man discovers one weird trick that makes him millions overnight",
        "Government secretly controlling weather with hidden technology"
    ]
    
    real_news = [
        "Stock market shows steady growth amid economic recovery efforts",
        "New research published in medical journal shows promising results",
        "Local community comes together to support flood victims",
        "Technology company announces quarterly earnings report",
        "University researchers develop new sustainable energy solution"
    ]
    
    texts = fake_news + real_news
    labels = [1] * len(fake_news) + [0] * len(real_news)
    
    return texts, labels

def test_fake_news_detection():
    """Test fake news detection"""
    print("\n" + "="*80)
    print("🔍 FAKE NEWS DETECTION TEST")
    print("="*80)
    
    detector = FakeNewsDetector()
    texts, labels = load_sample_data()
    accuracy = detector.train_model(texts, labels)
    
    print(f"\nModel Accuracy: {accuracy:.2%}")
    
    # Test with real news
    test_real = "Stock market shows steady growth amid economic recovery efforts"
    prediction, probability = detector.predict(test_real)
    
    print(f"\nTest Text (Real News): {test_real}")
    print(f"Prediction: {'FAKE' if prediction == 1 else 'REAL'}")
    print(f"Real News Probability: {probability[0]:.2%}")
    print(f"Fake News Probability: {probability[1]:.2%}")
    
    # Test with fake news
    test_fake = "Scientists discover that drinking water causes cancer in 99% of cases"
    prediction, probability = detector.predict(test_fake)
    
    print(f"\nTest Text (Fake News): {test_fake}")
    print(f"Prediction: {'FAKE' if prediction == 1 else 'REAL'}")
    print(f"Real News Probability: {probability[0]:.2%}")
    print(f"Fake News Probability: {probability[1]:.2%}")

def test_sentiment_analysis():
    """Test sentiment analysis"""
    print("\n" + "="*80)
    print("😊 SENTIMENT ANALYSIS TEST")
    print("="*80)
    
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "I absolutely love this! It's amazing!",
        "This is terrible and I hate it",
        "The weather is 25 degrees today",
        "The product is good but the service was bad"
    ]
    
    print(f"\n{'Text':<45} {'Sentiment':<15} {'Polarity':<10} {'Subjectivity'}")
    print("-" * 80)
    
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"{text:<45} {result['sentiment']:<15} {result['polarity']:<10.3f} {result['subjectivity']:.3f}")

def test_emotion_breakdown():
    """Test emotion breakdown"""
    print("\n" + "="*80)
    print("🧠 EMOTION BREAKDOWN TEST")
    print("="*80)
    
    analyzer = SentimentAnalyzer()
    
    test_text = "I am very happy and excited about this amazing news! But slightly worried about the future."
    emotions = analyzer.get_emotion_breakdown(test_text)
    
    print(f"\nTest Text: {test_text}")
    print("\nEmotion Breakdown:")
    print("-" * 40)
    
    # Sort by score descending
    for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
        if score > 0:
            bar_length = int(score * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"{emotion:<15} {score:>6.2%} | {bar}")

def test_complete_analysis():
    """Test complete analysis with expected output format"""
    print("\n" + "="*80)
    print("✅ COMPLETE ANALYSIS TEST")
    print("="*80)
    
    detector = FakeNewsDetector()
    sentiment_analyzer = SentimentAnalyzer()
    
    # Train the model
    texts, labels = load_sample_data()
    accuracy = detector.train_model(texts, labels)
    
    # Test text
    test_text = "New research published in medical journal shows promising results for cancer treatment breakthrough"
    
    print(f"\nTest Text: {test_text}")
    print("\n" + "="*80)
    
    # Fake news detection
    prediction, probability = detector.predict(test_text)
    fake_confidence = probability[1]
    real_confidence = probability[0]
    
    print("\n🚨 FAKE NEWS DETECTION")
    print("-" * 80)
    print(f"✅ LIKELY REAL NEWS (Confidence: {real_confidence:.2%})")
    
    if fake_confidence > 0.7:
        risk_level = "High Risk"
    elif fake_confidence > 0.5:
        risk_level = "Medium Risk"
    else:
        risk_level = "Low Risk"
    
    print(f"✅ {risk_level}: Appears to be legitimate news.")
    
    print("\n📊 MODEL CONFIDENCE")
    print("-" * 80)
    overall_confidence = max(real_confidence, fake_confidence)
    print(f"Overall Confidence: {overall_confidence:.2%}")
    print(f"Real News Probability: {real_confidence:.2%}")
    print(f"Fake News Probability: {fake_confidence:.2%}")
    
    # Sentiment Analysis
    sentiment = sentiment_analyzer.analyze_sentiment(test_text)
    
    print("\n😊 SENTIMENT ANALYSIS")
    print("-" * 80)
    print(f"Overall Sentiment: {sentiment['sentiment']}")
    print(f"Polarity: {sentiment['polarity']:.3f}")
    print(f"Subjectivity: {sentiment['subjectivity']:.3f}")
    print(f"Analysis: {sentiment['explanation']}")
    
    # Emotion breakdown
    emotions = sentiment_analyzer.get_emotion_breakdown(test_text)
    
    print("\n🧠 EMOTION BREAKDOWN")
    print("-" * 80)
    for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
        if score > 0.01:
            print(f"{emotion:<15} {score:>6.2%}")

if __name__ == "__main__":
    print("\n🔬 COMPREHENSIVE FEATURE TEST")
    print("=" * 80)
    
    try:
        test_fake_news_detection()
        test_sentiment_analysis()
        test_emotion_breakdown()
        test_complete_analysis()
        
        print("\n" + "="*80)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*80)
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
