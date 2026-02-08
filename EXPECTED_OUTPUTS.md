# 📊 Expected Output Examples

## Example 1: Real News Article

**Input Text:**
```
"Researchers at MIT have published a groundbreaking study in Nature showing 
that renewable energy can power 80% of the grid by 2030. The study analyzed 
data from 50 countries over 10 years. Lead researcher Dr. Emma Johnson stated 
that 'the findings are promising for climate action.'"
```

---

### Expected Output Display:

```
═══════════════════════════════════════════════════════════════════════════════

🚨 FAKE NEWS DETECTION                    😊 SENTIMENT ANALYSIS
─────────────────────────────────────────────────────────────────────────────

✅ LIKELY REAL NEWS (Confidence: 72.45%)  Overall Sentiment: 😊 Positive

┌─────────────────────────────────────┐  Polarity: 0.345
│ ✅ Low Risk                         │  Subjectivity: 0.425
│ Appears to be legitimate news.      │
└─────────────────────────────────────┘  Analysis: Positive language detected;
                                          somewhat objective
📊 MODEL CONFIDENCE                       
                                         ✅ High confidence: 34.5%
┌─────────────────────────────────────┐
│     Confidence Level (%)            │  🧠 EMOTION BREAKDOWN
│        72.45%                       │  
│        ████████████████░░░░░░│       │  ┌────────────────────────┐
│                                     │  │ Trust        42.35%    │
│Real: 72.45%, Fake: 27.55%         │  │ Anticipation 31.20%    │
└─────────────────────────────────────┘  │ Neutral      18.45%    │
                                         │ Joy          08.00%    │
🧠 WHY THIS PREDICTION?                  └────────────────────────┘

Word    Impact      Influence Score
───────────────────────────────────
study   Real 🟢     0.0456
data    Real 🟢     0.0412
energy  Real 🟢     0.0385
...

═══════════════════════════════════════════════════════════════════════════════

📋 ANALYSIS SUMMARY

🟢 LIKELY REAL    │ ✅ Low Risk     │ 😊 Positive    │ 89 words
Final Verdict     │ 27.55% Fake    │ Sentiment      │ ~1 min read

═══════════════════════════════════════════════════════════════════════════════
```

---

## Example 2: Fake News Article

**Input Text:**
```
"BREAKING: Scientists discover SHOCKING truth! One weird trick eliminates toxins 
from your body in 24 hours! Doctors HATE this discovery! Big Pharma doesn't want 
you to know! This changes EVERYTHING! Share before it gets deleted!"
```

---

### Expected Output Display:

```
═══════════════════════════════════════════════════════════════════════════════

🚨 FAKE NEWS DETECTION                    😊 SENTIMENT ANALYSIS
─────────────────────────────────────────────────────────────────────────────

⚠️ LIKELY FAKE NEWS (Confidence: 89.32%) Overall Sentiment: 😄 Very Positive

┌─────────────────────────────────────┐  Polarity: 0.742
│ 🔴 High Risk                        │  Subjectivity: 0.876
│ Strong indicators of fake news!     │
└─────────────────────────────────────┘  Analysis: Very positive language 
                                         detected; highly subjective
📊 MODEL CONFIDENCE                      
                                        ✅ High confidence: 74.2%
┌─────────────────────────────────────┐
│     Confidence Level (%)            │  🧠 EMOTION BREAKDOWN
│        89.32%                       │
│        █████████████████████░│       │  ┌────────────────────────┐
│                                     │  │ Joy           48.20%   │
│Real: 10.68%, Fake: 89.32%         │  │ Surprise      31.15%   │
└─────────────────────────────────────┘  │ Fear          12.30%   │
                                         │ Anticipation  08.35%   │
🧠 WHY THIS PREDICTION?                  └────────────────────────┘

Word      Impact         Influence Score
─────────────────────────────────────────
trick     Fake 🔴        0.0892
doctors   Fake 🔴        0.0756
hate      Fake 🔴        0.0684
breaking  Fake 🔴        0.0645
shocking  Fake 🔴        0.0598
...

═══════════════════════════════════════════════════════════════════════════════

📋 ANALYSIS SUMMARY

🔴 LIKELY FAKE    │ 🔴 High Risk    │ 😄 Very Positive │ 42 words
Final Verdict     │ 89.32% Fake    │ Sentiment        │ ~1 min read

═══════════════════════════════════════════════════════════════════════════════
```

---

## Example 3: Neutral/Technical News

**Input Text:**
```
"The Federal Reserve announced a 0.25% interest rate increase effective 
immediately. The decision was made by the policy committee on March 15, 2026. 
Economic indicators show inflation at 3.2% with employment at 3.9%."
```

---

### Expected Output Display:

```
═══════════════════════════════════════════════════════════════════════════════

🚨 FAKE NEWS DETECTION                    😊 SENTIMENT ANALYSIS
─────────────────────────────────────────────────────────────────────────────

✅ LIKELY REAL NEWS (Confidence: 81.23%) Overall Sentiment: 😐 Neutral

┌─────────────────────────────────────┐  Polarity: 0.085
│ ✅ Low Risk                         │  Subjectivity: 0.182
│ Appears to be legitimate news.      │
└─────────────────────────────────────┘  Analysis: Neutral or balanced 
                                         language; very objective
📊 MODEL CONFIDENCE                      
                                        ✅ Low confidence: 8.5%
┌─────────────────────────────────────┐
│     Confidence Level (%)            │  🧠 EMOTION BREAKDOWN
│        81.23%                       │
│        ████████████████░░░░░░│       │  ┌────────────────────────┐
│                                     │  │ Neutral      85.42%    │
│Real: 81.23%, Fake: 18.77%         │  │ Trust        09.31%    │
└─────────────────────────────────────┘  │ Anticipation 03.27%    │
                                         │ Others       02.00%    │
🧠 WHY THIS PREDICTION?                  └────────────────────────┘

Word        Impact      Influence Score
───────────────────────────────────────
announced   Real 🟢     0.0342
federal     Real 🟢     0.0298
rate        Real 🟢     0.0267
...

═══════════════════════════════════════════════════════════════════════════════

📋 ANALYSIS SUMMARY

🟢 LIKELY REAL    │ ✅ Low Risk     │ 😐 Neutral     │ 34 words
Final Verdict     │ 18.77% Fake    │ Sentiment      │ ~1 min read

═══════════════════════════════════════════════════════════════════════════════
```

---

## Example 4: Batch Analysis Results

**Input CSV File:** `test_batch_data.csv` (14 entries)

**Expected Output Table:**

```
┌─────────────────────────────────────────┬──────────┬──────────────┬────────────┬──────────┐
│ Text (first 100 chars)                  │ Is Fake  │ Fake Conf.   │ Sentiment  │ Polarity │
├─────────────────────────────────────────┼──────────┼──────────────┼────────────┼──────────┤
│ Scientists discover drinking water ca...│ True     │ 94.32%       │ Positive   │ 0.412    │
│ Breaking news: Aliens have landed in... │ True     │ 96.78%       │ Very Pos.  │ 0.645    │
│ Stock market shows steady growth amid... │ False    │ 15.42%       │ Neutral    │ 0.098    │
│ New medical research published in pee...│ False    │ 12.87%       │ Positive   │ 0.234    │
│ Local man discovers one weird trick t...│ True     │ 92.11%       │ Very Pos.  │ 0.789    │
│ Miracle vitamin supplement cures all..  │ True     │ 95.45%       │ Very Pos.  │ 0.701    │
│ University researchers develop innova..│ False    │ 18.23%       │ Positive   │ 0.312    │
│ Weather forecast predicts mild temper..│ False    │ 22.15%       │ Neutral    │ 0.045    │
│ Breaking: Government secretly controll...│ True    │ 91.32%       │ Negative   │ -0.342   │
│ Celebrity announces new movie deal wi...│ False    │ 28.90%       │ Positive   │ 0.156    │
│ Technology company announces quarterly.│ False    │ 19.87%       │ Neutral    │ 0.076    │
│ Vaccines contain microchips for mind.. │ True     │ 97.64%       │ Negative   │ -0.512   │
│ City council approves budget for infra..│ False    │ 21.34%       │ Neutral    │ 0.089    │
│ Local community comes together to supp..│ False    │ 16.45%       │ Positive   │ 0.287    │
└─────────────────────────────────────────┴──────────┴──────────────┴────────────┴──────────┘

📊 ANALYSIS SUMMARY
─────────────────────────────────────────
Fake News Detected:     8 (57.14%)
Real News Detected:     6 (42.86%)
Positive Sentiments:    9 (64.29%)
Neutral Sentiments:     3 (21.43%)
Negative Sentiments:    2 (14.29%)
Average Polarity:       0.245

✅ Results can be downloaded as CSV file
```

---

## Metric Ranges & Interpretation

### Confidence Score Interpretation
```
0-20%:   Very Low (Uncertain)
20-40%:  Low (Some doubt)
40-60%:  Medium (Moderate confidence)
60-80%:  High (Strong confidence)
80-100%: Very High (Very confident)
```

### Risk Level Thresholds
```
Fake Confidence:
  ≤ 50%       → 🟢 LOW RISK
  50% - 70%   → 🟡 MEDIUM RISK  
  > 70%       → 🔴 HIGH RISK
```

### Polarity Distribution
```
  -1.0 ──────────────────────────────────── +1.0
   │         │         │         │         │
   ✓         ✓         ✓         ✓         ✓
  -0.8      -0.4       0.0      +0.4      +0.8
  
Very      Negative   Neutral   Positive   Very
Negative                               Positive
```

### Subjectivity Interpretation
```
0.0 - 0.2:  Very Objective (Facts, data, measurements)
0.2 - 0.4:  Mostly Objective (Some opinions mixed in)
0.4 - 0.6:  Balanced (Mix of facts and opinions)
0.6 - 0.8:  Mostly Subjective (Mostly opinions)
0.8 - 1.0:  Very Subjective (Personal views, emotions)
```

---

## Sample Emotion Breakdowns

### Positive News
```
Joy:           40-50%
Trust:         20-30%
Anticipation:  10-20%
Others:        0-10%
```

### Negative News
```
Anger:         30-40%
Sadness:       20-30%
Fear:          15-25%
Disgust:       10-20%
```

### Neutral/Technical News
```
Neutral:       70-85%
Trust:         10-20%
Others:        0-10%
```

---

**Note:** These examples show typical output formats. Actual values may vary based on the specific text content and language variations.
