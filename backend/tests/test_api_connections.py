"""
Debug script to verify API keys and connectivity for all providers.
Run with: python -m tests.test_api_connections
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.llm_client import LLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connections():
    print("\n🔍 API CONNECTION DIAGNOSTIC TOOL")
    print("="*60)
    
    # Force reload .env
    load_dotenv(override=True)
    
    # 1. Check Environment Variables
    print("\n1. CHECKING ENVIRONMENT VARIABLES:")
    keys = {
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "HUGGINGFACE_API_KEY": os.getenv("HUGGINGFACE_API_KEY"),
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY")
    }
    
    for name, value in keys.items():
        if value:
            masked = f"{value[:4]}...{value[-4:]}" if len(value) > 8 else "PRESENT"
            print(f"   ✅ {name}: {masked}")
        else:
            print(f"   ❌ {name}: MISSING")

    # 2. Test Gemini (Google)
    print("\n2. TESTING GEMINI (Librarian):")
    if keys["GOOGLE_API_KEY"]:
        try:
            client = LLMClient(provider="gemini")
            response = client.generate("Hello, say 'Gemini OK'.")
            if response:
                print(f"   ✅ Success: {response.strip()[:50]}...")
            else:
                print("   ❌ Failed: No response (Check logs for 429)")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print("   ⚠️ Skipped (No Key)")

    # 3. Test Hugging Face (Inspector)
    print("\n3. TESTING HUGGING FACE (Inspector):")
    if keys["HUGGINGFACE_API_KEY"]:
        try:
            client = LLMClient(provider="huggingface")
            response = client.generate("Return only the word 'Connected'.")
            if response:
                print(f"   ✅ Success: {response.strip()[:50]}...")
            else:
                print("   ❌ Failed: No response or 404/401 Error")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print("   ⚠️ Skipped (No Key)")

    # 4. Test Groq (Fixer/Diagnostician)
    print("\n4. TESTING GROQ (Fixer):")
    if keys["GROQ_API_KEY"]:
        try:
            client = LLMClient(provider="groq")
            response = client.generate("Say 'Groq Connected'.")
            if response:
                print(f"   ✅ Success: {response.strip()[:50]}...")
            else:
                print("   ❌ Failed: No response")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    else:
        print("   ⚠️ Skipped (No Key)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    test_connections()