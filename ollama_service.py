import requests
import json
from typing import Dict, List, Any, Optional

class OllamaService:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        
    def generate_response(self, 
                         prompt: str, 
                         model: str = "llama3", 
                         context: Optional[List[str]] = None,
                         system_prompt: Optional[str] = None) -> str:
        """
        Generate a response from Ollama model using the provided prompt and context
        
        Args:
            prompt: The user's query
            model: The Ollama model to use
            context: List of strings representing context from vector DB
            system_prompt: Optional system prompt to guide the model
        
        Returns:
            The model's response as a string
        """
        url = f"{self.base_url}/api/generate"
        
        # Prepare prompt with context if provided
        full_prompt = prompt
        if context and len(context) > 0:
            context_text = "\n\n".join(context)
            
            # Detect if query is likely in Korean
            has_korean = any('\uAC00' <= char <= '\uD7A3' for char in prompt)
            
            if has_korean:
                full_prompt = f"""다음은 사용자의 질문에 답변하는 데 도움이 될 수 있는 관련 정보입니다.

관련 정보:
{context_text}

사용자 질문:
{prompt}

제공된 관련 정보를 바탕으로 사용자의 질문에 답변해 주세요. 만약 제공된 정보에서 질문에 대한 답변을 찾을 수 없다면, 그렇게 말씀해 주세요."""
            else:
                full_prompt = f"""I'm going to provide you with some relevant information to help answer the user's question.

RELEVANT INFORMATION:
{context_text}

USER QUESTION:
{prompt}

Please answer the user's question based on the relevant information provided. If the information doesn't contain what's needed to answer the question, please say so."""
        
        # Prepare the payload
        payload = {
            "model": model,
            "prompt": full_prompt,
            "stream": False
        }
        
        # Add system prompt if provided
        if system_prompt:
            payload["system"] = system_prompt
            
        # Make the API call
        try:
            response = requests.post(url, json=payload)
            
            # Check for API errors
            if response.status_code != 200:
                error_msg = f"Error from Ollama API: {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg += f" - {error_data['error']}"
                except:
                    pass
                print(error_msg)
                return f"Error: {error_msg}"
            
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling Ollama API: {e}")
            return f"Error generating response: {str(e)}" 