"""
Utility functions for Gemini AI integration
Wrapper for consistent Gemini API calls across the application
"""

import os
import logging
import json
import re
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Updated import
from google import genai

load_dotenv()

logger = logging.getLogger(__name__)

class GeminiClient:
    """Wrapper for Gemini API calls using the new google-genai SDK"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini client
        
        Args:
            api_key: Gemini API key (uses GOOGLE_API_KEY env var if not provided)
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.client = None
        self.model_name = "gemini-2.0-flash"
        
        if not self.api_key:
            logger.warning("Gemini API key not found. Mock mode will be used.")
            self.available = False
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
                self.available = True
                logger.info(f"Gemini Client initialized with model: {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.available = False
    
    def generate_text(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Generate text using Gemini
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated text or error message
        """
        if not self.available or not self.client:
            return f"Mock response: {prompt[:50]}... [Gemini API key required]"
        
        try:
            # New API syntax
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    'max_output_tokens': max_tokens,
                    'temperature': 0.3
                }
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return f"Error generating response: {str(e)}"
    
    def generate_structured(self, prompt: str, expected_format: str = "JSON") -> Dict:
        """
        Generate structured response (attempt to parse as JSON)
        
        Args:
            prompt: Input prompt with format instructions
            expected_format: Expected format (JSON, YAML, etc.)
            
        Returns:
            Parsed response or dict with text
        """
        if not self.available or not self.client:
            return {"error": "Gemini API not available"}
        
        try:
            # Add explicit format instruction
            format_prompt = f"{prompt}\n\nIMPORTANT: Return ONLY a valid {expected_format} object. No Markdown formatting."
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=format_prompt
            )
            
            text = response.text
            
            if expected_format.upper() == "JSON":
                # Clean up markdown if present (```json ... ```)
                text = text.replace("```json", "").replace("```", "").strip()
                
                # Try to extract JSON if there's extra text
                json_pattern = r'\{.*\}|\[.*\]'
                match = re.search(json_pattern, text, re.DOTALL)
                
                if match:
                    json_str = match.group()
                    return json.loads(json_str)
                else:
                    # Try parsing the whole text
                    return json.loads(text)
            else:
                return {"text": text}
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}. Raw text: {text[:100]}...")
            return {"error": "Failed to parse JSON", "raw_text": text}
        except Exception as e:
            logger.error(f"Structured generation failed: {e}")
            return {"error": str(e)}
    
    def is_available(self) -> bool:
        """Check if Gemini API is available"""
        return self.available


# Singleton instance
_gemini_client = None

def get_gemini_client() -> GeminiClient:
    """Get or create Gemini client singleton"""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = GeminiClient()
    return _gemini_client