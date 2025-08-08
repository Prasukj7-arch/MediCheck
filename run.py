#!/usr/bin/env python3
"""
MediCheck Application Startup Script
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'flask',
        'flask-cors',
        'PyPDF2',
        'sentence-transformers',
        'chromadb',
        'openai',
        'python-dotenv',
        'numpy',
        'requests',
        'werkzeug'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_environment():
    """Check environment configuration"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("💡 Creating .env file with default values...")
        
        with open('.env', 'w') as f:
            f.write("# MediCheck Environment Variables\n")
            f.write("# Get your API key from: https://openrouter.ai/\n")
            f.write("OPENROUTER_API_KEY=your_openrouter_api_key_here\n")
            f.write("SECRET_KEY=your-secret-key-change-this-in-production\n")
        
        print("✅ .env file created")
        print("⚠️  Please update the .env file with your actual API keys")
        return False
    
    return True

def main():
    """Main startup function"""
    print("🏥 MediCheck - AI-Powered Medical Assistant")
    print("=" * 50)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ All dependencies found")
    
    # Check environment
    print("🔍 Checking environment...")
    if not check_environment():
        print("⚠️  Please configure your .env file before continuing")
        print("   You can still run the application, but some features may not work")
    
    # Create necessary directories
    print("🔍 Creating necessary directories...")
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('chroma_db', exist_ok=True)
    print("✅ Directories created")
    
    # Start the application
    print("\n🚀 Starting MediCheck application...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
