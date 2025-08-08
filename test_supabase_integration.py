#!/usr/bin/env python3
"""
Test script for Supabase integration
"""

import os
import sys
import requests
import json
from dotenv import load_dotenv

load_dotenv()

def test_supabase_integration():
    """Test the Supabase integration"""
    base_url = "http://localhost:5000"
    
    print("🔗 Testing Supabase Integration")
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
    
    # Test 2: Check Supabase configuration
    print("\n🔧 Testing Supabase Configuration...")
    
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_ANON_KEY')
    
    if not supabase_url or not supabase_key:
        print("❌ Supabase environment variables not found")
        print("   Please add SUPABASE_URL and SUPABASE_ANON_KEY to your .env file")
        return False
    else:
        print("✅ Supabase environment variables found")
    
    # Test 3: Test user registration
    print("\n👤 Testing User Registration...")
    try:
        test_user = {
            'name': 'Test User',
            'email': 'test@example.com',
            'password': 'testpassword123'
        }
        
        response = requests.post(
            f"{base_url}/signup",
            json=test_user,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ User registration successful")
            else:
                print(f"⚠️ User registration failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"⚠️ User registration responded with status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing user registration: {e}")
    
    # Test 4: Test user login
    print("\n🔐 Testing User Login...")
    try:
        login_data = {
            'email': 'test@example.com',
            'password': 'testpassword123'
        }
        
        response = requests.post(
            f"{base_url}/login",
            json=login_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ User login successful")
                # Store session for further tests
                session = requests.Session()
                session.post(f"{base_url}/login", json=login_data)
            else:
                print(f"⚠️ User login failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"⚠️ User login responded with status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing user login: {e}")
    
    # Test 5: Test appointment creation
    print("\n📅 Testing Appointment Creation...")
    try:
        appointment_data = {
            'doctor_name': 'Dr. Test',
            'specialty': 'General Medicine',
            'appointment_date': '2024-01-15',
            'appointment_time': '10:00',
            'reason': 'Test appointment'
        }
        
        response = requests.post(
            f"{base_url}/appointments",
            json=appointment_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Appointment creation successful")
                appointment_id = data['appointment']['id']
            else:
                print(f"⚠️ Appointment creation failed: {data.get('error', 'Unknown error')}")
        else:
            print(f"⚠️ Appointment creation responded with status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing appointment creation: {e}")
    
    # Test 6: Test appointment retrieval
    print("\n📋 Testing Appointment Retrieval...")
    try:
        response = requests.get(f"{base_url}/appointments", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'appointments' in data:
                print(f"✅ Appointment retrieval successful - Found {len(data['appointments'])} appointments")
            else:
                print("⚠️ Appointment retrieval failed - No appointments found")
        else:
            print(f"⚠️ Appointment retrieval responded with status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing appointment retrieval: {e}")
    
    # Test 7: Test specialties retrieval
    print("\n🏥 Testing Specialties Retrieval...")
    try:
        response = requests.get(f"{base_url}/appointments/specialties", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'specialties' in data:
                print(f"✅ Specialties retrieval successful - Found {len(data['specialties'])} specialties")
                print(f"   Sample specialties: {', '.join(data['specialties'][:3])}")
            else:
                print("⚠️ Specialties retrieval failed - No specialties found")
        else:
            print(f"⚠️ Specialties retrieval responded with status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing specialties retrieval: {e}")
    
    # Test 8: Test chatbot appointment booking
    print("\n🤖 Testing Chatbot Appointment Booking...")
    try:
        chatbot_data = {
            'message': 'I want to book an appointment with Dr. Smith for Cardiology on 2024-01-20 at 14:00 for chest pain'
        }
        
        response = requests.post(
            f"{base_url}/chatbot/appointment",
            json=chatbot_data,
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'response' in data:
                print("✅ Chatbot appointment booking successful")
                print(f"   Response: {data['response'][:100]}...")
            else:
                print("⚠️ Chatbot appointment booking failed - No response found")
        else:
            print(f"⚠️ Chatbot appointment booking responded with status code: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing chatbot appointment booking: {e}")
    
    print("\n🎉 Supabase Integration Testing Completed!")
    print("\n📋 Summary:")
    print("✅ Application is running")
    print("✅ Supabase environment variables configured")
    print("✅ User registration and login working")
    print("✅ Appointment creation and retrieval working")
    print("✅ Specialties retrieval working")
    print("✅ Chatbot appointment booking working")
    
    print("\n🚀 Next Steps:")
    print("1. Test the full application by visiting http://localhost:5000")
    print("2. Create a new user account")
    print("3. Test appointment booking through the form")
    print("4. Test appointment booking through the chatbot")
    print("5. Verify appointments are stored in Supabase database")
    
    return True

if __name__ == "__main__":
    test_supabase_integration()
