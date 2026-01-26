#!/usr/bin/env python3
"""
Test script to verify all dependencies are working correctly
"""

def test_imports():
    """Test all required imports"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas and NumPy imported successfully")
    except ImportError as e:
        print(f"❌ Pandas/NumPy import failed: {e}")
        return False
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        print("✅ Scikit-learn imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    try:
        from textblob import TextBlob
        print("✅ TextBlob imported successfully")
    except ImportError as e:
        print(f"❌ TextBlob import failed: {e}")
        return False
    
    try:
        import nltk
        print("✅ NLTK imported successfully")
    except ImportError as e:
        print(f"❌ NLTK import failed: {e}")
        return False
    
    try:
        from wordcloud import WordCloud
        print("✅ WordCloud imported successfully")
    except ImportError as e:
        print(f"⚠️ WordCloud import failed: {e}")
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"⚠️ Matplotlib import failed: {e}")
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"⚠️ Plotly import failed: {e}")
    
    return True

def test_functionality():
    """Test basic functionality"""
    print("\nTesting functionality...")
    
    try:
        # Test TextBlob sentiment analysis
        from textblob import TextBlob
        blob = TextBlob("This is a great day!")
        sentiment = blob.sentiment
        print(f"✅ TextBlob sentiment analysis works: {sentiment}")
    except Exception as e:
        print(f"❌ TextBlob functionality failed: {e}")
        return False
    
    try:
        # Test scikit-learn
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        
        # Simple test data
        texts = ["This is good news", "This is bad news", "Great article", "Terrible content"]
        labels = [0, 1, 0, 1]  # 0 = real, 1 = fake
        
        vectorizer = TfidfVectorizer(max_features=100)
        X = vectorizer.fit_transform(texts)
        
        model = LogisticRegression()
        model.fit(X, labels)
        
        # Test prediction
        test_text = ["This is excellent news"]
        X_test = vectorizer.transform(test_text)
        prediction = model.predict(X_test)
        
        print(f"✅ ML model training and prediction works: {prediction}")
    except Exception as e:
        print(f"❌ ML functionality failed: {e}")
        return False
    
    return True

def main():
    print("🔍 AI News & Sentiment Analyzer - Dependency Test")
    print("=" * 50)
    
    if not test_imports():
        print("\n❌ Critical imports failed. Please install missing dependencies.")
        return False
    
    if not test_functionality():
        print("\n❌ Functionality tests failed.")
        return False
    
    print("\n🎉 All tests passed! The application should work correctly.")
    print("\nTo run the application:")
    print("  streamlit run app.py")
    print("\nTo install browser extension:")
    print("  1. Open Chrome -> chrome://extensions/")
    print("  2. Enable 'Developer mode'")
    print("  3. Click 'Load unpacked' -> Select 'browser_extension' folder")
    
    return True

if __name__ == "__main__":
    main()