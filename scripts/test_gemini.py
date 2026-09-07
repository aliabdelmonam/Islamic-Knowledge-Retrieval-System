"""Verify the configured Gemini provider can be constructed."""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from app.core.config import settings
from app.providers import LLMProviderFactory

def main():
    print("=== Testing Gemini API Integration ===")
    
    print(f"Gemini Model in config: {settings.gemini_model}")
    print(f"Google API Key present in settings: {bool(settings.google_api_key)}")
    
    try:
        print("Initializing ChatGoogleGenerativeAI through the provider factory...")
        llm = LLMProviderFactory.create(settings, provider="gemini").create_chat_model()
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
