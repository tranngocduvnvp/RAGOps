import google.generativeai as genai
import openai
import requests
import re
from typing import List, Dict


class OnLineLLMs:
    def __init__(self, model_name: str, api_key: str, model_version: str, base_url: str = None):
        """Initialize model with the specified name, API key, and model version."""
        self.model_name = model_name.lower()
        self.model_version = model_version

        if self.model_name == "gemini" and api_key:
            self.client = openai.OpenAI(
                                api_key=api_key,
                                base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                            )
        elif self.model_name == "openai" and api_key:
            self.client = openai.OpenAI(api_key=api_key)
        elif self.model_name == "together" and api_key:
            self.base_url = f"{base_url}/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        else:
            raise ValueError("Unsupported model name or missing API key.")

    def remove_think_blocks(self,text):
        """Remove <think> blocks and their content from text"""
        # Pattern to match <think>...</think> blocks (including multiline)
        pattern = r'<think>.*?</think>'
        # Remove the think blocks using re.sub with DOTALL flag for multiline matching
        cleaned_text = re.sub(pattern, '', text, flags=re.DOTALL)
        # Clean up any extra whitespace that might be left
        cleaned_text = re.sub(r'\n\s*\n', '\n', cleaned_text).strip()
        return cleaned_text
    
    def generate_content(self, prompt: List[Dict[str, str]]) -> str:
        """Generate content using the online LLM based on the provided prompt.
            input: prompt (str): The prompt to generate content for.
            output: str: The generated content.
        """
        if self.model_name == "gemini":
            response = self.client.chat.completions.create(
            model=self.model_version,  
            messages=prompt
            )

            assistant_content = self.remove_think_blocks(response.choices[0].message.content)

            if response.usage:
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                total_tokens = response.usage.total_tokens

                cache_read_tokens = getattr(response.usage, "prompt_tokens_details", {}).get("cached_tokens", 0) if hasattr(response.usage, "prompt_tokens_details") else 0
            else:
                prompt_tokens = completion_tokens = total_tokens = cache_read_tokens = None

            return assistant_content, {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cache_read_tokens": cache_read_tokens
            }

        elif self.model_name == "openai":
            response = self.client.chat.completions.create(
                model=self.model_version,
                messages=prompt
            )
            return response.choices[0].message.content
        elif self.model_name == "together":
            data = {
                "model": self.model_version,
                "messages": prompt,
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 512,
            }
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            response_data = response.json()["choices"][0]["message"]["content"].strip()
            return self.remove_think_blocks(response_data)
        else:
            raise ValueError(f"Unsupported model name: {self.name}")


