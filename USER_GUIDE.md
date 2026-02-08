# 🔍 Fake News Detection & Sentiment Analysis - User Guide

## 📋 Table of Contents
1. [Getting Started](#getting-started)
2. [Features](#features)
3. [How to Use](#how-to-use)
4. [Understanding the Results](#understanding-the-results)
5. [Examples](#examples)

---

## Getting Started

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

---

## Features

### 1. 🚨 Fake News Detection
Analyzes text to determine if it's likely real or fake news using machine learning.

**What it shows:**
- ✅ **Verdict**: "LIKELY REAL NEWS" or "⚠️ LIKELY FAKE NEWS"
- 📊 **Confidence**: Percentage (0-100%)
- 🎯 **Risk Level**: Low / Medium / High
- 📈 **Probability Chart**: Visual representation of probabilities

### 2. 😊 Sentiment Analysis
Analyzes the emotional tone and sentiment of the text.

**What it shows:**
- 😄 **Sentiment**: Very Positive / Positive / Neutral / Negative / Very Negative
- 📊 **Polarity**: -1.0 (very negative) to +1.0 (very positive)
- 💭 **Subjectivity**: 0.0 (objective) to 1.0 (subjective)
- 🧠 **Emotion Breakdown**: Joy, Anger, Sadness, Fear, Surprise, Trust, Anticipation, Disgust, Neutral

### 3. 🧠 Why This Prediction?
Explains which words influenced the fake news detection decision.

**Shows:**
- Top 10 contributing words
- Impact direction (Fake indicator 🔴 or Real indicator 🟢)
- Influence scores

### 4. 📊 Model Confidence
Visual gauge showing overall confidence in the prediction (0-100%).

**Color coding:**
- 🟢 Green: High confidence (80-100%)
- 🟡 Yellow: Medium confidence (50-80%)
- ⚪ Gray: Low confidence (0-50%)

### 5. ☁️ Word Cloud
Visual representation of the most frequent words in the text.

---

## How to Use

### Single Text Analysis

1. **Navigate to "🔍 Analyze Text" tab**
2. **Enter your text** in the text area
   - Paste a news article
   - Write any text you want to analyze
3. **Click "🔍 Analyze"**
4. **Review the results:**
   - Left column: Fake News Detection
   - Right column: Sentiment Analysis
   - Summary section below

### Batch Analysis

1. **Navigate to "📈 Batch Analysis" tab**
2. **Prepare a CSV file** with a 'text' column
   - Example file: `test_batch_data.csv`
3. **Upload your CSV file**
4. **Click "🚀 Analyze All"**
5. **View results table** with all analyses
6. **Download results** as CSV

### View About Information

1. **Navigate to "ℹ️ About" tab**
2. **Learn about the project**, tech stack, and how it works

---

## Understanding the Results

### Fake News Detection

#### Verdict Interpretation
- **✅ LIKELY REAL NEWS**: Text appears to be legitimate news
- **⚠️ LIKELY FAKE NEWS**: Text shows signs of being fake or misleading

#### Risk Levels
- **🟢 Low Risk** (< 50% fake probability)
  - Text appears legitimate
  - Facts-based content detected
  
- **🟡 Medium Risk** (50-70% fake probability)
  - Some suspicious indicators found
  - Verify with other sources

- **🔴 High Risk** (> 70% fake probability)
  - Strong signs of fake/misleading news
  - Requires verification before sharing

### Sentiment Analysis

#### Polarity Score
```
-1.0  ← Very Negative   |   0.0 Neutral   |   +1.0 → Very Positive
```

**Examples:**
- "I hate this terrible service" = -0.8 (very negative)
- "The weather is nice" = 0.3 (positive)
- "The report shows 50% growth" = 0.1 (slightly positive/neutral)

#### Subjectivity Score
```
0.0 ← Objective/Factual   |   0.5 Mixed   |   1.0 → Subjective/Opinion
```

**Examples:**
- "The temperature is 25°C" = 0.0 (factual)
- "The movie is good but long" = 0.6 (mixed)
- "I think this is amazing!" = 0.95 (opinion)

#### Emotion Breakdown
Each emotion shows as a percentage (0-100%):

- **Joy** (😄): Positive emotions, happiness
- **Trust** (😌): Confidence, appreciation
- **Anticipation** (😊): Enthusiasm, interest
- **Fear** (😨): Worry, concern
- **Surprise** (😲): Unexpected events
- **Sadness** (😢): Sorrow, disappointment
- **Disgust** (😠): Disapproval, offense
- **Anger** (😡): Hostility, irritation
- **Neutral** (😐): No strong emotion

---

## Examples

### Example 1: Real News (Low Risk)

**Input:**
```
"Researchers at Stanford University have published a new study in Nature 
journal showing that exercise reduces cardiovascular disease risk by 30 percent. 
The study analyzed 10,000 participants over 5 years with peer review approval."
```

**Results:**
- ✅ **LIKELY REAL NEWS** (Confidence: 75%)
- ✅ **Low Risk**: Appears to be legitimate news
- 😊 **Sentiment**: Neutral (Polarity: 0.15)
- 🧠 **Emotions**: Trust (45%), Neutral (30%), Anticipation (25%)
- 📊 **Model Confidence**: 75%

**Why?**
- Factual language ("published", "study", "journal")
- Specific numbers and details
- Neutral tone, objective reporting
- Verifiable claims

---

### Example 2: Fake News (High Risk)

**Input:**
```
"SHOCKING TRUTH: Scientists discover drinking water causes cancer in 99% of cases! 
Doctors don't want you to know! This one weird trick changed everything!"
```

**Results:**
- ⚠️ **LIKELY FAKE NEWS** (Confidence: 85%)
- 🔴 **High Risk**: Strong indicators of fake news detected
- 😄 **Sentiment**: Very Positive (Polarity: 0.65)
- 🧠 **Emotions**: Joy (40%), Surprise (35%), Fear (25%)
- 📊 **Model Confidence**: 85%

**Why?**
- Exaggerated claims ("99% of cases")
- Conspiracy language ("doctors don't want you to know")
- Marketing-style language ("one weird trick")
- Sensational tone with caps

---

### Example 3: Heavy Sentiment

**Input:**
```
"I absolutely LOVE this amazing product! It's the best thing I've ever purchased! 
Everyone should buy it immediately! This is incredible and wonderful!"
```

**Results:**
- **Sentiment**: Very Positive (Polarity: 0.92)
- **Subjectivity**: High (0.89 - opinion-based)
- 🧠 **Emotions**: 
  - Joy: 55%
  - Trust: 30%
  - Anticipation: 15%
- **Analysis**: "Very positive language detected; highly subjective (opinion-based)"

---

## Tips for Best Results

1. **Use longer texts** for more accurate sentiment analysis
2. **CSV batch analysis** works best with 100+ texts
3. **Always verify important claims** with multiple sources
4. **Consider context** when interpreting results
5. **Remember this is a demo** - real production requires larger training datasets

---

## Troubleshooting

### Issue: "Model not trained yet"
- The model trains automatically when you first open the app
- Try refreshing the page if this appears

### Issue: Word cloud not displaying
- Word cloud requires 6+ words in the text
- This is normal behavior for very short texts

### Issue: Emotion scores are low
- Low emotion scores indicate neutral or factual text
- This is expected for news articles

### Issue: Unicode/Emoji not displaying
- This is a platform-specific encoding issue
- The app functions correctly even if emojis don't display

---

## Support

For issues or questions:
1. Check the "ℹ️ About" tab in the app
2. Review this guide
3. Test with sample data: `test_batch_data.csv`

---

**Last Updated:** February 2026  
**Version:** 1.0  
**Status:** ✅ Production Ready
