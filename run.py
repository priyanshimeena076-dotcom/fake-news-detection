#!/usr/bin/env python3
"""
Quick start script for the AI News & Sentiment Analyzer
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False
    return True

def run_streamlit():
    """Run the Streamlit application"""
    print("🚀 Starting the AI News & Sentiment Analyzer...")
    print("📱 The app will open in your browser at http://localhost:8501")
    print("🔍 To use the browser extension, load the 'browser_extension' folder in Chrome")
    print("\n" + "="*60)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
            "--theme.primaryColor", "#1f77b4"
        ])
    except KeyboardInterrupt:
        print("\n👋 Thanks for using AI News & Sentiment Analyzer!")
    except FileNotFoundError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ])

def main():
    print("🔍 AI News & Sentiment Analyzer")
    print("=" * 40)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return
    
    # Check if app.py exists
    if not os.path.exists("app.py"):
        print("❌ app.py not found!")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Run the application
    run_streamlit()

if __name__ == "__main__":
    main()