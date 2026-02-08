# 🔍 AI News & Sentiment Analyzer

A comprehensive AI-powered application that detects fake news and analyzes sentiment with explainable AI features. Built with Python, Streamlit, and includes a browser extension for real-time web page analysis.

## ✨ What's New (v1.0 - Updated Feb 2026)

✅ **Fixed Fake News Detection** - Shows correct predictions with confidence scores  
✅ **Enhanced Sentiment Analysis** - All emotion options now display correctly  
✅ **Proper Risk Assessment** - Low/Medium/High risk levels clearly displayed  
✅ **Correct Graph Values** - All graphs show proper percentages (0-100%)  
✅ **Complete Emotion Breakdown** - All 9 emotions calculated and displayed  
✅ **Improved "Why This Prediction?"** - Shows top influencing words with impact  
✅ **Better Model Confidence** - Gauge chart with clear confidence metrics  

## 🎯 Features

### Web Application
- **Fake News Detection**: ML-powered detection with explainable AI
  - Shows verdict (LIKELY REAL or LIKELY FAKE)
  - Risk levels: Low/Medium/High
  - Confidence percentage (0-100%)
  - Top influencing words with explanations

- **Sentiment Analysis**: Comprehensive emotion and polarity analysis
  - Overall sentiment classification
  - Polarity score (-1 to +1)
  - Subjectivity score (0 to 1)
  - 9-emotion breakdown: Joy, Anger, Sadness, Fear, Surprise, Trust, Anticipation, Disgust, Neutral

- **Batch Processing**: Analyze multiple texts from CSV files
  - Upload CSV with 'text' column
  - Process multiple entries at once
  - Download results as CSV

- **Visual Analytics**: Interactive charts, word clouds, and emotion breakdowns
  - Probability charts (Real vs Fake)
  - Confidence gauge
  - Emotion pie charts
  - Word clouds
  - Analysis summaries

- **Export Results**: Download analysis results as CSV

### Browser Extension
- **Real-time Analysis**: Analyze any webpage instantly
- **Smart Content Extraction**: Focuses on main content, ignores ads/navigation
- **Client-side Processing**: Fast analysis without sending data to servers
- **Visual Indicators**: Shows analysis progress on web pages

## 🛠️ Tech Stack

- **Backend**: Python, scikit-learn, NLTK, TextBlob
- **Frontend**: Streamlit, Plotly, Matplotlib
- **Browser Extension**: JavaScript (Chrome Extension API)
- **Data Processing**: Pandas, NumPy
- **Visualization**: WordCloud, Seaborn

## 📊 Key Metrics Displayed

### Fake News Detection
- ✅ **Verdict**: LIKELY REAL or LIKELY FAKE with emoji indicators
- 📊 **Confidence**: 0-100% probability
- 🎯 **Risk Level**: Low (<50%), Medium (50-70%), High (>70%)
- 📈 **Probability Chart**: Visual representation of Real vs Fake

### Sentiment Analysis  
- 😊 **Overall Sentiment**: Very Positive / Positive / Neutral / Negative / Very Negative
- 📈 **Polarity**: -1.0 (very negative) to +1.0 (very positive)
- 💭 **Subjectivity**: 0.0 (objective) to 1.0 (subjective)
- 🧠 **Emotion Breakdown**: All 9 emotions with percentages

### Analysis Details
- 🔍 **Why This Prediction?**: Top 10 words influencing the prediction
- 📊 **Model Confidence**: Gauge chart showing confidence level (0-100%)
- 📋 **Summary**: Final verdict, risk level, sentiment, text stats

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Web Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

### 3. Install Browser Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (top right toggle)
3. Click "Load unpacked" and select the `browser_extension` folder
4. The extension icon will appear in your toolbar

### 4. Test the App

Try uploading the sample file `test_batch_data.csv` to test batch analysis features.

## 📚 Documentation

- **[USER_GUIDE.md](USER_GUIDE.md)** - Complete user guide with examples
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Summary of all improvements made
- **[TECHNICAL_SUMMARY.md](TECHNICAL_SUMMARY.md)** - Technical implementation details
- **[EXPECTED_OUTPUTS.md](EXPECTED_OUTPUTS.md)** - Sample outputs and interpretations

## 📝 Example Usage

### Single Text Analysis
1. Paste any news article or text
2. Click "🔍 Analyze"
3. View detailed analysis including:
   - Fake news detection with confidence
   - Sentiment analysis with emotions
   - Key words influencing the prediction
   - Model confidence gauge
   - Recommendations

### Batch Analysis
1. Prepare CSV file with 'text' column
2. Upload to "📈 Batch Analysis" tab
3. Click "🚀 Analyze All"
4. View summary statistics
5. Download results as CSV

## ✅ All Tests Passed

```
TEST 1: Sentiment Analysis ✓
- Polarity: 0.688 (positive)
- Subjectivity: 0.750 (subjective)

TEST 2: Emotion Breakdown ✓
- Joy: 41.83%
- Surprise: 31.94%
- Anticipation: 20.91%
- Neutral: 5.32%

TEST 3: Fake News Detection ✓
- Real News Probability: 59.66%
- Fake News Probability: 40.34%
- Prediction: REAL NEWS ✓
```

## 📖 How to Use

### Web Application

1. **Single Text Analysis**:
   - Paste any news article or text
   - Click "Analyze" to get fake news detection and sentiment analysis
   - View explanations showing which words influenced the decision

2. **Batch Analysis**:
   - Upload a CSV file with a 'text' column
   - Analyze multiple texts at once
   - Download results as CSV

### Browser Extension

1. Navigate to any news website or article
2. Click the extension icon in your toolbar
3. Click "Analyze This Page"
4. View instant results with fake news detection and sentiment analysis
5. Click "View Full Analysis" to open detailed analysis in the web app

## 🧠 How It Works

### Fake News Detection
- **TF-IDF Vectorization**: Converts text into numerical features
- **Logistic Regression**: Classifies text as fake or real
- **Explainable AI**: Shows which words contributed to the decision
- **Heuristic Indicators**: Detects common fake news patterns

### Sentiment Analysis
- **TextBlob Integration**: Polarity and subjectivity analysis
- **Emotion Mapping**: Breaks down emotions (joy, anger, sadness, etc.)
- **Word-based Analysis**: Counts positive/negative sentiment words
- **Visual Representation**: Word clouds and emotion charts

## 📊 Model Performance

The demo model is trained on sample data for demonstration purposes. For production use:

- Train with larger, real-world datasets
- Use advanced models like BERT or GPT
- Implement cross-validation and proper evaluation metrics
- Consider domain-specific training data

## 🔧 Customization

### Adding New Features

1. **Custom ML Models**: Replace the LogisticRegression with advanced models
2. **Additional Languages**: Add multi-language support with language detection
3. **Real-time Feeds**: Integrate with news APIs for live monitoring
4. **Social Media**: Add Twitter/Facebook integration

### Browser Extension Enhancements

1. **Background Processing**: Move analysis to background scripts
2. **API Integration**: Connect to cloud-based ML APIs
3. **User Preferences**: Add settings for analysis sensitivity
4. **History Tracking**: Store analysis history locally

## 📁 Project Structure

```
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── browser_extension/
    ├── manifest.json          # Extension configuration
    ├── popup.html             # Extension popup interface
    ├── popup.js               # Extension popup logic
    ├── popup.css              # Extension styling
    └── content.js             # Content script for web pages
```

## ⚠️ Important Notes

- **Demo Limitations**: Current model uses sample data for demonstration
- **Privacy**: Browser extension processes data locally (no data sent to servers)
- **Accuracy**: Always verify important news through multiple reliable sources
- **Context**: Sentiment analysis is subjective and context-dependent

## 🔮 Future Enhancements

- [ ] Advanced deep learning models (BERT, GPT)
- [ ] Multi-language support
- [ ] Real-time news feed monitoring
- [ ] Social media integration
- [ ] Mobile app version
- [ ] API for third-party integration
- [ ] Advanced visualization dashboards
- [ ] User feedback and model improvement

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **scikit-learn** for machine learning capabilities
- **Streamlit** for the amazing web framework
- **TextBlob** for sentiment analysis
- **NLTK** for natural language processing
- **Plotly** for interactive visualizations

---
