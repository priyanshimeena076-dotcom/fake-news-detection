#!/usr/bin/env python3
"""
Test script to demonstrate polarity and subjectivity with examples
"""

from textblob import TextBlob

def test_sentiment_examples():
    """Test sentiment analysis with various examples"""
    
    examples = [
        # Very positive
        "This is absolutely amazing! I love it so much!",
        
        # Positive
        "This is a good product. I like it.",
        
        # Neutral/Objective
        "The weather today is 25 degrees Celsius.",
        
        # Negative
        "This is not good. I don't like it.",
        
        # Very negative
        "This is terrible! I hate it completely!",
        
        # Subjective positive
        "I think this movie is the best thing ever created!",
        
        # Objective positive
        "The study shows positive results with 95% accuracy.",
        
        # Mixed sentiment
        "The food was delicious but the service was terrible.",
        
        # Sarcastic (often misinterpreted)
        "Oh great, another wonderful Monday morning...",
        
        # Factual statement
        "Python is a programming language created by Guido van Rossum."
    ]
    
    print("🔍 Sentiment Analysis Examples")
    print("=" * 80)
    print(f"{'Text':<50} {'Polarity':<10} {'Subjectivity':<12} {'Sentiment'}")
    print("-" * 80)
    
    for text in examples:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Classify sentiment
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
        
        # Truncate text for display
        display_text = text[:47] + "..." if len(text) > 50 else text
        
        print(f"{display_text:<50} {polarity:<10.3f} {subjectivity:<12.3f} {sentiment}")
    
    print("\n📖 Understanding the Scores:")
    print("\nPolarity (-1.0 to +1.0):")
    print("  -1.0 = Very negative (hate, terrible, awful)")
    print("   0.0 = Neutral (facts, balanced statements)")
    print("  +1.0 = Very positive (love, amazing, excellent)")
    
    print("\nSubjectivity (0.0 to 1.0):")
    print("   0.0 = Objective (facts, data, measurements)")
    print("   1.0 = Subjective (opinions, feelings, beliefs)")
    
    print("\n💡 Why values might be 0:")
    print("- Empty or very short text")
    print("- Text with no sentiment words")
    print("- Pure factual statements")
    print("- Technical or scientific content")
    print("- TextBlob corpus not properly loaded")

if __name__ == "__main__":
    try:
        test_sentiment_examples()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure TextBlob is installed: pip install textblob")
        print("You may also need to download corpora: python -m textblob.download_corpora")