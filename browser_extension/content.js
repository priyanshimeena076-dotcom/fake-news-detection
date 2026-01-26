// Content script for the browser extension
// This script runs on all web pages and can interact with page content

// Listen for messages from the popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'extractContent') {
        try {
            const content = extractPageContent();
            sendResponse({ success: true, content: content });
        } catch (error) {
            sendResponse({ success: false, error: error.message });
        }
    }
    return true; // Keep the message channel open for async response
});

function extractPageContent() {
    // Remove script and style elements
    const elementsToRemove = document.querySelectorAll('script, style, nav, footer, aside, .advertisement, .ads, .sidebar');
    
    // Get main content areas (in order of preference)
    const contentSelectors = [
        'article',
        'main',
        '[role="main"]',
        '.content',
        '.post-content',
        '.article-content',
        '.entry-content',
        '.main-content',
        'body'
    ];
    
    let content = '';
    
    for (const selector of contentSelectors) {
        const element = document.querySelector(selector);
        if (element) {
            // Clone the element to avoid modifying the original page
            const clone = element.cloneNode(true);
            
            // Remove unwanted elements from the clone
            const unwanted = clone.querySelectorAll('script, style, nav, footer, aside, .advertisement, .ads, .sidebar, .comments, .social-share');
            unwanted.forEach(el => el.remove());
            
            content = clone.innerText || clone.textContent || '';
            if (content.trim().length > 100) {
                break;
            }
        }
    }
    
    // If no content found, try to get text from the entire body
    if (!content || content.trim().length < 100) {
        const bodyClone = document.body.cloneNode(true);
        const unwanted = bodyClone.querySelectorAll('script, style, nav, footer, aside, .advertisement, .ads, .sidebar, .comments, .social-share');
        unwanted.forEach(el => el.remove());
        content = bodyClone.innerText || bodyClone.textContent || '';
    }
    
    // Clean up the content
    content = content
        .replace(/\s+/g, ' ')
        .replace(/\n+/g, '\n')
        .trim();
    
    // Get page metadata
    const title = document.title || '';
    const url = window.location.href;
    const domain = window.location.hostname;
    
    return {
        title: title,
        url: url,
        domain: domain,
        content: content.substring(0, 5000), // Limit content length
        wordCount: content.split(' ').length,
        timestamp: new Date().toISOString()
    };
}

// Add visual indicator when content is being analyzed
function showAnalysisIndicator() {
    const indicator = document.createElement('div');
    indicator.id = 'ai-analysis-indicator';
    indicator.innerHTML = `
        <div style="
            position: fixed;
            top: 20px;
            right: 20px;
            background: #4CAF50;
            color: white;
            padding: 10px 15px;
            border-radius: 5px;
            z-index: 10000;
            font-family: Arial, sans-serif;
            font-size: 14px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        ">
            🔍 AI Analysis in progress...
        </div>
    `;
    document.body.appendChild(indicator);
    
    // Remove indicator after 3 seconds
    setTimeout(() => {
        const element = document.getElementById('ai-analysis-indicator');
        if (element) {
            element.remove();
        }
    }, 3000);
}

// Listen for analysis start
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'showIndicator') {
        showAnalysisIndicator();
        sendResponse({ success: true });
    }
});