#!/usr/bin/env python3
"""
Startup script for Medical Report RAG Extractor
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask',
        'PyPDF2', 
        'sentence_transformers',
        'chromadb',
        'openai',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ Dependencies installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies")
            print("Please run: pip install -r requirements.txt")
            return False
    
    return True

def check_env_file():
    """Check if .env file exists and has API key"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("Creating .env file template...")
        
        # Copy from env_example.txt if it exists
        example_file = Path('env_example.txt')
        if example_file.exists():
            with open(example_file, 'r') as f:
                content = f.read()
            
            with open(env_file, 'w') as f:
                f.write(content)
            
            print("✅ Created .env file from template")
            print("⚠️  Please edit .env file and add your OpenRouter API key")
        else:
            # Create basic .env file
            with open(env_file, 'w') as f:
                f.write("# OpenRouter API Key\n")
                f.write("# Get your API key from https://openrouter.ai/\n")
                f.write("OPENROUTER_API_KEY=your_openrouter_api_key_here\n")
            
            print("✅ Created .env file")
            print("⚠️  Please edit .env file and add your OpenRouter API key")
        
        return False
    
    # Check if API key is set
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key or api_key == 'your_openrouter_api_key_here':
        print("⚠️  OpenRouter API key not configured")
        print("Please edit .env file and set your OPENROUTER_API_KEY")
        return False
    
    print("✅ OpenRouter API key configured")
    return True

def main():
    """Main startup function"""
    print("🏥 Medical Report RAG Extractor")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    print("\n🔍 Checking dependencies...")
    if not check_dependencies():
        return False
    
    print("\n🔧 Checking configuration...")
    if not check_env_file():
        print("\n📝 Setup Instructions:")
        print("1. Get a free API key from https://openrouter.ai/")
        print("2. Edit the .env file and replace 'your_openrouter_api_key_here' with your actual API key")
        print("3. Run this script again")
        return False
    
    print("\n🚀 Starting application...")
    print("The application will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("-" * 40)
    
    try:
        # Import and run the Flask app
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
