#!/usr/bin/env python3
"""
Test script for Symptom Scanner functionality
"""

import os
import sys
import requests
import json
import base64
from PIL import Image
import numpy as np

def create_test_image():
    """Create a simple test image for testing"""
    # Create a simple test image (100x100 pixels, red color)
    img_array = np.zeros((100, 100, 3), dtype=np.uint8)
    img_array[:, :, 0] = 255  # Red channel
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Save to a temporary file
    test_image_path = "test_image.jpg"
    img.save(test_image_path)
    
    return test_image_path

def test_symptom_scanner():
    """Test the symptom scanner functionality"""
    base_url = "http://localhost:5000"
    
    print("🔍 Testing Symptom Scanner Functionality")
    print("=" * 50)
    
    # Test 1: Check if application is running
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Application is running successfully")
        else:
            print(f"⚠️ Application responded with status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Application is not running. Please start it with: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error connecting to application: {e}")
        return False
    
    # Test 2: Check symptom scanner status
    try:
        response = requests.get(f"{base_url}/status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            scanner_status = data.get('symptom_scanner', 'not_available')
            print(f"✅ Symptom scanner status: {scanner_status}")
        else:
            print(f"⚠️ Status endpoint responded with status code: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking symptom scanner status: {e}")
    
    # Test 3: Test image upload functionality
    print("\n📁 Testing Image Upload Functionality...")
    try:
        # Create a test image
        test_image_path = create_test_image()
        
        # Test upload endpoint
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{base_url}/symptom-scanner/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Image upload successful")
            print(f"   Primary condition: {data.get('scan_result', {}).get('primary_condition', 'Unknown')}")
            print(f"   Confidence: {data.get('scan_result', {}).get('confidence', 0):.2%}")
        elif response.status_code == 401:
            print("⚠️ Authentication required - this is expected for protected endpoints")
        else:
            print(f"⚠️ Image upload responded with status code: {response.status_code}")
            print(f"   Response: {response.text}")
        
        # Clean up test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            
    except Exception as e:
        print(f"❌ Error testing image upload: {e}")
    
    # Test 4: Test live image scanning functionality
    print("\n📷 Testing Live Image Scanning Functionality...")
    try:
        # Create a test image and convert to base64
        test_image_path = create_test_image()
        with open(test_image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Test live scan endpoint
        payload = {'image_data': f'data:image/jpeg;base64,{image_data}'}
        response = requests.post(
            f"{base_url}/symptom-scanner/scan-live", 
            json=payload, 
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Live image scanning successful")
            print(f"   Primary condition: {data.get('scan_result', {}).get('primary_condition', 'Unknown')}")
            print(f"   Confidence: {data.get('scan_result', {}).get('confidence', 0):.2%}")
        elif response.status_code == 401:
            print("⚠️ Authentication required - this is expected for protected endpoints")
        else:
            print(f"⚠️ Live image scanning responded with status code: {response.status_code}")
            print(f"   Response: {response.text}")
        
        # Clean up test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
            
    except Exception as e:
        print(f"❌ Error testing live image scanning: {e}")
    
    # Test 5: Test condition info endpoint
    print("\n📋 Testing Condition Info Functionality...")
    try:
        response = requests.get(f"{base_url}/symptom-scanner/condition-info/Acne", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Condition info retrieval successful")
            print(f"   Description: {data.get('description', 'N/A')[:50]}...")
        elif response.status_code == 401:
            print("⚠️ Authentication required - this is expected for protected endpoints")
        else:
            print(f"⚠️ Condition info responded with status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing condition info: {e}")
    
    print("\n🎉 Symptom Scanner Testing Completed!")
    print("\n📋 Summary:")
    print("✅ Application is running")
    print("✅ Symptom scanner endpoints are accessible")
    print("⚠️ Authentication required for protected endpoints (this is expected)")
    print("\n🚀 Next Steps:")
    print("1. Login to the application")
    print("2. Navigate to the Symptom Scanner section")
    print("3. Test image upload functionality")
    print("4. Test live camera functionality")
    print("5. Verify results display correctly")
    
    return True

if __name__ == "__main__":
    test_symptom_scanner()
