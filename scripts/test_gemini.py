"""
Simple test script to verify Gemini integration.
Run:
    python scripts/test_gemini.py
"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from app.core.config import settings
from app.services.llm import build_llm

def main():
    print("=== Testing Gemini API Integration ===")
    
    # Temporarily set provider to gemini
    provider = "gemini"
    print(f"Gemini Model in config: {settings.gemini_model}")
    print(f"Google API Key present in settings: {bool(settings.google_api_key)}")
    
    try:
        print("Initializing ChatGoogleGenerativeAI...")
        llm = build_llm(
            provider=provider,
            gemini_model=settings.gemini_model,
            google_api_key=settings.google_api_key or "",
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        print("LLM built successfully.")
        
        print("Sending a test prompt: 'السلام عليكم'")
        response = llm.invoke("السلام عليكم")
        print("\n=== Response ===")
        print(response.content)
        print("================")
        print("Success! Gemini integration works perfectly.")
        
    except ModuleNotFoundError as e:
        print(f"\n[Error] Missing dependency: {e}")
        print("Please run: pip install langchain-google-genai")
    except Exception as e:
        print(f"\n[Error] Failed to communicate with Gemini: {e}")

if __name__ == "__main__":
    main()
