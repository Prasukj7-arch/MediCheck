#!/usr/bin/env python3
"""
Test script to check available vision models on OpenRouter
"""

import os
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_vision_models():
    """Test which vision models are available on OpenRouter"""
    
    # Check if API key is available
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ OPENROUTER_API_KEY not found in environment variables")
        return False
    
    # Initialize OpenAI client
    try:
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )
        print("✅ OpenAI client initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing OpenAI client: {e}")
        return False
    
    # List of vision models to test
    vision_models = [
        "openai/gpt-4o",
        "openai/gpt-4o-mini", 
        "anthropic/claude-3-5-sonnet",
        "anthropic/claude-3-haiku",
        "google/gemini-pro-1.5"
    ]
    
    print("\n🔍 Testing Vision Models on OpenRouter")
    print("=" * 50)
    
    working_models = []
    
    for model in vision_models:
        try:
            print(f"\nTesting {model}...")
            
            # Test with a simple text prompt first
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": "Hello, this is a test message."}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            print(f"✅ {model} - Text completion works")
            
            # Now test with image (if supported)
            try:
                # Create a simple test image data (1x1 pixel)
                import base64
                test_image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "What do you see in this image?"
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{test_image_data}"
                                    }
                                }
                            ]
                        }
                    ],
                    max_tokens=100,
                    temperature=0.1
                )
                
                print(f"✅ {model} - Vision capabilities work")
                working_models.append(model)
                
            except Exception as vision_error:
                print(f"⚠️ {model} - Vision not supported: {vision_error}")
                
        except Exception as e:
            print(f"❌ {model} - Failed: {e}")
    
    print(f"\n🎉 Testing completed!")
    print(f"✅ Working vision models: {working_models}")
    
    if working_models:
        print(f"\n🚀 Recommended model: {working_models[0]}")
        return True
    else:
        print("\n❌ No working vision models found")
        return False

if __name__ == "__main__":
    test_vision_models()
