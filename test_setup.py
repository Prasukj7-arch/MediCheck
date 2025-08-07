#!/usr/bin/env python3
"""
Test script to verify the Medical Report RAG Extractor setup
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported"""
    print("🔍 Testing package imports...")
    
    try:
        import flask
        print("✅ Flask imported successfully")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return False
    
    try:
        import PyPDF2
        print("✅ PyPDF2 imported successfully")
    except ImportError as e:
        print(f"❌ PyPDF2 import failed: {e}")
        return False
    
    try:
        from sentence_transformers import SentenceTransformer
        print("✅ sentence-transformers imported successfully")
    except ImportError as e:
        print(f"❌ sentence-transformers import failed: {e}")
        return False
    
    try:
        import chromadb
        print("✅ ChromaDB imported successfully")
    except ImportError as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False
    
    try:
        import openai
        print("✅ OpenAI imported successfully")
    except ImportError as e:
        print(f"❌ OpenAI import failed: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
    except ImportError as e:
        print(f"❌ python-dotenv import failed: {e}")
        return False
    
    return True

def test_embedding_model():
    """Test if the embedding model can be loaded"""
    print("\n🧠 Testing embedding model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        print("✅ Embedding model loaded successfully")
        
        # Test encoding
        test_text = ["This is a test sentence."]
        embeddings = model.encode(test_text)
        print(f"✅ Embedding generation successful (shape: {embeddings.shape})")
        
        return True
    except Exception as e:
        print(f"❌ Embedding model test failed: {e}")
        return False

def test_chromadb():
    """Test if ChromaDB can be initialized"""
    print("\n🗄️ Testing ChromaDB...")
    
    try:
        import chromadb
        
        client = chromadb.PersistentClient(path="./test_chroma_db")
        
        # Test collection creation
        collection = client.get_or_create_collection(
            name="test_collection"
        )
        print("✅ ChromaDB initialized successfully")
        
        # Cleanup
        client.delete_collection("test_collection")
        print("✅ ChromaDB test collection cleaned up")
        
        return True
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    print("\n🔧 Testing environment configuration...")
    
    # Check if .env file exists
    if os.path.exists('.env'):
        print("✅ .env file found")
    else:
        print("⚠️  .env file not found - you'll need to create one with your OpenRouter API key")
    
    # Check for API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENROUTER_API_KEY')
    if api_key and api_key != 'your_openrouter_api_key_here':
        print("✅ OpenRouter API key found")
    else:
        print("⚠️  OpenRouter API key not configured - you'll need to set OPENROUTER_API_KEY in .env file")
    
    return True

def test_flask_app():
    """Test if Flask app can be imported"""
    print("\n🌐 Testing Flask application...")
    
    try:
        # Import the app (this will test all the RAG system initialization)
        from app import app, rag_system
        print("✅ Flask app imported successfully")
        print("✅ RAG system initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Flask app test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🏥 Medical Report RAG Extractor - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Package Imports", test_imports),
        ("Embedding Model", test_embedding_model),
        ("ChromaDB", test_chromadb),
        ("Environment", test_environment),
        ("Flask App", test_flask_app)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready.")
        print("\n🚀 To run the application:")
        print("1. Make sure you have your OpenRouter API key in .env file")
        print("2. Run: python app.py")
        print("3. Open http://localhost:5000 in your browser")
    else:
        print("⚠️  Some tests failed. Please check the errors above and fix them.")
        print("\n💡 Common solutions:")
        print("- Run: pip install -r requirements.txt")
        print("- Create .env file with your OpenRouter API key")
        print("- Check your internet connection for model downloads")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
