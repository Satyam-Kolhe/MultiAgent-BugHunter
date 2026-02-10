"""
Unified LLM Client - The "Brain Switcher"
Allows agents to use Gemini, HuggingFace (Mistral), or Groq (Llama3) seamlessly.
"""

import os
import logging
import json
import requests
import time
import re
from typing import Optional
from dotenv import load_dotenv

# Import Google GenAI
try:
    from google import genai
except ImportError:
    genai = None

load_dotenv()
logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, provider: str = "gemini", api_key: str = None):
        """
        Initialize the LLM Client.
        
        Args:
            provider: "gemini", "huggingface", or "groq"
            api_key: Specific API key for the provider
        """
        self.provider = provider.lower()
        self.api_key = api_key
        
        # Load default keys from environment if not provided
        if not self.api_key:
            if self.provider == "gemini":
                self.api_key = os.getenv("GOOGLE_API_KEY")
            elif self.provider == "huggingface":
                self.api_key = os.getenv("HUGGINGFACE_API_KEY")
            elif self.provider == "groq":
                self.api_key = os.getenv("GROQ_API_KEY")

        # Initialize specific clients
        self.gemini_client = None
        if self.provider == "gemini" and self.api_key:
            try:
                self.gemini_client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.error(f"Gemini Init Error: {e}")

    def generate(self, prompt: str, system_instruction: str = None) -> Optional[str]:
        """Generate text using the selected provider"""
        try:
            if self.provider == "gemini":
                return self._generate_gemini(prompt)
            elif self.provider == "huggingface":
                return self._generate_huggingface(prompt)
            elif self.provider == "groq":
                return self._generate_groq(prompt, system_instruction)
            else:
                logger.error(f"Unknown provider: {self.provider}")
                return None
        except Exception as e:
            logger.error(f"{self.provider.upper()} Generation Failed: {e}")
            return None

    def _generate_gemini(self, prompt: str, max_retries: int = 2) -> Optional[str]:
        """Generate text using Gemini API with retry logic for rate limits"""
        if not self.gemini_client: 
            return None
        
        for attempt in range(max_retries):
            try:
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash", contents=prompt
                )
                return response.text if response else None
            except Exception as e:
                error_str = str(e)
                # Check for Rate Limit (429) errors
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        # Extract retry delay or use default
                        wait_time = 20 
                        if "retryDelay" in error_str or "retry in" in error_str.lower():
                            match = re.search(r'retry in (\d+(?:\.\d+)?)s', error_str.lower())
                            if match:
                                wait_time = int(float(match.group(1))) + 2
                        
                        logger.warning(f"GEMINI Rate limit (429). Waiting {wait_time}s before retry {attempt+1}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.warning(f"GEMINI Quota Exceeded. Switching to fallback.")
                        return None
                else:
                    logger.warning(f"GEMINI Error: {e}")
                    return None
        
        return None

    def _generate_huggingface(self, prompt: str) -> Optional[str]:
        """Uses Mistral-7B-Instruct via Hugging Face Router (Fixes 410 Error)"""
        if not self.api_key: 
            return None
        
        # Updated endpoints: Use the router URL first as it is the new standard
        api_endpoints = [
            "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.3",
            "https://router.huggingface.co/hf-inference/models/mistralai/Mistral-7B-Instruct-v0.2",
            # Fallback to old API just in case
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
        ]
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        formatted_prompt = f"<s>[INST] {prompt} [/INST]"
        
        for api_url in api_endpoints:
            try:
                response = requests.post(
                    api_url, 
                    headers=headers, 
                    json={
                        "inputs": formatted_prompt,
                        "parameters": {"max_new_tokens": 1000, "return_full_text": False}
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0].get('generated_text', '')
                    elif isinstance(result, dict):
                        return result.get('generated_text', '')
                    return str(result)
                
                elif response.status_code == 503:
                    logger.warning(f"HF Model loading (503) for {api_url}, trying next endpoint...")
                    time.sleep(5) # Wait a bit for loading
                    continue
                elif response.status_code == 404:
                    continue # Try next model version
                else:
                    logger.warning(f"HF Error {response.status_code} for {api_url}")
                    continue
                    
            except Exception as e:
                logger.warning(f"HF Request failed for {api_url}: {e}")
                continue
        
        logger.error("HF: All endpoints failed.")
        return None

    def _generate_groq(self, prompt: str, system_instruction: str = None) -> Optional[str]:
        """Uses Llama3-70b via Groq (Fixes 400 Error & JSON parsing)"""
        if not self.api_key: return None
        
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Check if prompt asks for JSON to enable strict mode
        is_json_request = "json" in prompt.lower()
        
        # Helper to ensure system instruction exists for JSON mode
        sys_content = system_instruction or "You are a helpful AI assistant."
        if is_json_request and "json" not in sys_content.lower():
            sys_content += " You must output valid JSON."

        payload = {
            # UPDATED: Use the supported Llama 3 model
            "model": "llama-3.3-70b-versatile", 
            "messages": [
                {"role": "system", "content": sys_content},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1 # Low temp for code/JSON stability
        }
        
        if is_json_request:
            payload["response_format"] = {"type": "json_object"}
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                # Clean up any potential markdown wrappers even in JSON mode
                content = content.replace("```json", "").replace("```", "").strip()
                return content
            else:
                logger.error(f"Groq Error {response.status_code}: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Groq Request Failed: {e}")
            return None