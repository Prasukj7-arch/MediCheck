#!/usr/bin/env python3
"""
Test script to verify MediCheck application setup
"""

import os
import sys
import requests
import json

def test_application():
    """Test the application endpoints"""
    base_url = "http://localhost:5000"
    
    print("🏥 Testing MediCheck Application Setup")
    print("=" * 50)
    
    # Test 1: Check if application is running
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Application is running successfully")
        else:
            print(f"⚠️ Application responded with status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Application is not running. Please start it with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error connecting to application: {e}")
        return False
    
    # Test 2: Check login page
    try:
        response = requests.get(f"{base_url}/login", timeout=5)
        if response.status_code == 200:
            print("✅ Login page is accessible")
        else:
            print(f"⚠️ Login page responded with status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error accessing login page: {e}")
    
    # Test 3: Check signup page
    try:
        response = requests.get(f"{base_url}/signup", timeout=5)
        if response.status_code == 200:
            print("✅ Signup page is accessible")
        else:
            print(f"⚠️ Signup page responded with status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error accessing signup page: {e}")
    
    # Test 4: Check system status
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ System status: {data.get('status', 'unknown')}")
            print(f"   Documents stored: {data.get('documents_stored', 0)}")
        else:
            print(f"⚠️ Status endpoint responded with status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking system status: {e}")
    
    print("\n🎉 Setup verification completed!")
    print("\n📋 Next Steps:")
    print("1. Open your browser and go to: http://localhost:5000")
    print("2. Create a new account using the signup page")
    print("3. Login and explore the features")
    print("4. Try uploading a medical report PDF")
    print("5. Test the chatbot with voice input")
    
    return True

if __name__ == "__main__":
    test_application()
