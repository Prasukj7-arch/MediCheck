#!/usr/bin/env python3
"""
Simple test to verify symptom scanner module can be imported
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test if the symptom scanner module can be imported"""
    try:
        from backend.symptom_scanner import SymptomScanner
        print("✅ SymptomScanner module imported successfully")
        
        # Test initialization
        scanner = SymptomScanner()
        print("✅ SymptomScanner initialized successfully")
        
        # Test placeholder model
        if scanner.model is not None:
            print("✅ Placeholder model created successfully")
        else:
            print("⚠️ No model available")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error initializing SymptomScanner: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing SymptomScanner Import")
    print("=" * 30)
    success = test_import()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
