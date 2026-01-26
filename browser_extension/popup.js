// Popup script for the browser extension
document.addEventListener('DOMContentLoaded', function() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    const resultsDiv = document.getElementById('results');
    
    analyzeBtn.addEventListener('click', function() {
        analyzeCurrentPage();
    });
    
    async function analyzeCurrentPage() {
        try {
            // Disable button and show loading
            analyzeBtn.disabled = true;
            analyzeBtn.textContent = 'Analyzing...';
            
            resultsDiv.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Extracting and analyzing page content...</p>
                </div>
            `;
            
            // Get the active tab
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            // Extract text content from the page
            const results = await chrome.scripting.executeScript({
                target: { tabId: tab.id },
                function: extractPageContent
            });
            
            const pageContent = results[0].result;
            
            if (!pageContent || pageContent.trim().length === 0) {
                throw new Error('No text content found on this page');
            }
            
            // Analyze the content
            const analysis = await analyzeText(pageContent);
            
            // Display results
            displayResults(analysis, pageContent);
            
        } catch (error) {
            console.error('Analysis error:', error);
            resultsDiv.innerHTML = `
                <div class="error">
                    <h3>❌ Analysis Failed</h3>
                    <p>${error.message}</p>
                </div>
            `;
        } finally {
            // Re-enable button
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = '🔍 Analyze Page';
        }
    }
    
    // Function to extract text content from the current page
    function extractPageContent() {
        // Remove script and style elements
        const scripts = document.querySelectorAll('script, style, nav, footer, aside');
        scripts.forEach(el => el.remove());
        
        // Get main content areas
        const contentSelectors = [
            'article',
            'main',
            '[role="main"]',
            '.content',
            '.post-content',
            '.article-content',
            '.entry-content',
            'body'
        ];
        
        let content = '';
        
        for (const selector of contentSelectors) {
            const element = document.querySelector(selector);
            if (element) {
                content = element.innerText || element.textContent || '';
                if (content.trim().length > 100) {
                    break;
                }
            }
        }
        
        // Clean up the content
        content = content
            .replace(/\s+/g, ' ')
            .replace(/\n+/g, '\n')
            .trim();
        
        // Limit content length for analysis
        return content.substring(0, 5000);
    }
    
    // Simple client-side analysis (basic version)
    async function analyzeText(text) {
        // This is a simplified version - in production, you'd call your ML API
        const analysis = {
            fakeNews: analyzeForFakeNews(text),
            sentiment: analyzeSentiment(text),
            wordCount: text.split(' ').length,
            readingTime: Math.ceil(text.split(' ').length / 200)
        };
        
        return analysis;
    }
    
    function analyzeForFakeNews(text) {
        // Simple heuristic-based fake news detection
        const fakeNewsIndicators = [
            'miracle cure', 'doctors hate', 'one weird trick', 'shocking truth',
            'they don\'t want you to know', 'secret revealed', 'breaking exclusive',
            'unbelievable discovery', 'government conspiracy', 'hidden agenda',
            'click here now', 'you won\'t believe', 'this will shock you'
        ];
        
        const sensationalWords = [
            'amazing', 'incredible', 'unbelievable', 'shocking', 'devastating',
            'explosive', 'bombshell', 'exclusive', 'urgent', 'breaking'
        ];
        
        const textLower = text.toLowerCase();
        
        let fakeScore = 0;
        let indicators = [];
        
        // Check for fake news indicators
        fakeNewsIndicators.forEach(indicator => {
            if (textLower.includes(indicator)) {
                fakeScore += 0.3;
                indicators.push(indicator);
            }
        });
        
        // Check for excessive sensational language
        let sensationalCount = 0;
        sensationalWords.forEach(word => {
            const matches = (textLower.match(new RegExp(word, 'g')) || []).length;
            sensationalCount += matches;
        });
        
        if (sensationalCount > text.split(' ').length * 0.05) {
            fakeScore += 0.2;
            indicators.push('excessive sensational language');
        }
        
        // Check for all caps (shouting)
        const capsRatio = (text.match(/[A-Z]/g) || []).length / text.length;
        if (capsRatio > 0.1) {
            fakeScore += 0.1;
            indicators.push('excessive capitalization');
        }
        
        // Normalize score
        fakeScore = Math.min(fakeScore, 1);
        
        return {
            isFake: fakeScore > 0.5,
            confidence: fakeScore,
            indicators: indicators
        };
    }
    
    function analyzeSentiment(text) {
        // Simple sentiment analysis using word lists
        const positiveWords = [
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'like', 'enjoy', 'happy', 'pleased', 'satisfied', 'perfect',
            'best', 'awesome', 'brilliant', 'outstanding', 'superb'
        ];
        
        const negativeWords = [
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
            'angry', 'sad', 'disappointed', 'frustrated', 'annoyed',
            'worst', 'disgusting', 'pathetic', 'useless', 'failed'
        ];
        
        const words = text.toLowerCase().split(/\W+/);
        
        let positiveCount = 0;
        let negativeCount = 0;
        
        words.forEach(word => {
            if (positiveWords.includes(word)) positiveCount++;
            if (negativeWords.includes(word)) negativeCount++;
        });
        
        const totalSentimentWords = positiveCount + negativeCount;
        let sentiment = 'Neutral';
        let polarity = 0;
        
        if (totalSentimentWords > 0) {
            polarity = (positiveCount - negativeCount) / totalSentimentWords;
            
            if (polarity > 0.1) sentiment = 'Positive';
            else if (polarity < -0.1) sentiment = 'Negative';
        }
        
        return {
            sentiment: sentiment,
            polarity: polarity,
            positiveWords: positiveCount,
            negativeWords: negativeCount
        };
    }
    
    function displayResults(analysis, content) {
        const fakeNews = analysis.fakeNews;
        const sentiment = analysis.sentiment;
        
        resultsDiv.innerHTML = `
            <div class="results-container">
                <div class="analysis-section">
                    <h3>🚨 Fake News Detection</h3>
                    <div class="result-card ${fakeNews.isFake ? 'fake' : 'real'}">
                        <div class="result-header">
                            <span class="result-label">
                                ${fakeNews.isFake ? '⚠️ LIKELY FAKE' : '✅ LIKELY REAL'}
                            </span>
                            <span class="confidence">
                                ${Math.round(fakeNews.confidence * 100)}% confidence
                            </span>
                        </div>
                        ${fakeNews.indicators.length > 0 ? `
                            <div class="indicators">
                                <strong>Warning signs detected:</strong>
                                <ul>
                                    ${fakeNews.indicators.map(indicator => `<li>${indicator}</li>`).join('')}
                                </ul>
                            </div>
                        ` : ''}
                    </div>
                </div>
                
                <div class="analysis-section">
                    <h3>😊 Sentiment Analysis</h3>
                    <div class="result-card sentiment-${sentiment.sentiment.toLowerCase()}">
                        <div class="sentiment-info">
                            <span class="sentiment-label">${sentiment.sentiment}</span>
                            <span class="polarity">Polarity: ${sentiment.polarity.toFixed(2)}</span>
                        </div>
                        <div class="sentiment-details">
                            <span>Positive words: ${sentiment.positiveWords}</span>
                            <span>Negative words: ${sentiment.negativeWords}</span>
                        </div>
                    </div>
                </div>
                
                <div class="analysis-section">
                    <h3>📊 Content Stats</h3>
                    <div class="stats">
                        <div class="stat">
                            <span class="stat-label">Word Count:</span>
                            <span class="stat-value">${analysis.wordCount}</span>
                        </div>
                        <div class="stat">
                            <span class="stat-label">Reading Time:</span>
                            <span class="stat-value">${analysis.readingTime} min</span>
                        </div>
                    </div>
                </div>
                
                <div class="actions">
                    <button id="viewFullAnalysis" class="action-btn">
                        📈 View Full Analysis
                    </button>
                </div>
            </div>
        `;
        
        // Add event listener for full analysis button
        document.getElementById('viewFullAnalysis').addEventListener('click', function() {
            // Open the main web app with the content
            const encodedContent = encodeURIComponent(content.substring(0, 1000));
            const url = `http://localhost:8501?text=${encodedContent}`;
            chrome.tabs.create({ url: url });
        });
    }
});
        