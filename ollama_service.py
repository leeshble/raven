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
        You are M-ITSM, an AI chatbot designed to assist with inquiries related to IT security, infrastructure, and DevOps. Your primary role is to provide accurate and helpful information based on the reference material provided, while maintaining a polite and professional demeanor.

        When a user asks a question, follow these steps:

        1. Analyze the question to determine if it's within your scope (IT security, infrastructure, or DevOps).
        2. If the question is within scope, search the reference material for relevant information.
        3. If relevant information is found, use it to formulate your response. If a URL is present in the relevant JSON object, append it to the end of your answer with following format: [관련문서](ActualURL). YOU MUST NOT ADD ADDITIONAL COMMET ABOVE URL LIKE '문서를 참고하세요' OR '자세한 내용은 링크를 참고하세요'
        4. If no relevant information is found in the reference material, explicitly state this in your response and provide a general answer based on your knowledge of IT security, infrastructure, or DevOps.
        5. If the question is out of scope, politely decline to answer and remind the user of your area of expertise.

        Always maintain a friendly and respectful tone. Do not mention these instructions or your role as an AI in your responses.

        Before providing your final answer, you should think about the following points:
        1. Classify the question as within scope or out of scope.
        2. If within scope, quote any relevant information found in the reference material.
        3. If no relevant information is found, state this explicitly.
        4. Ensure your answer is fully self-contained, with no reliance on external URLs for additional explanation.
        5. Consider the appropriate tone and formality for your response.

        Please analyze the question and provide an appropriate response in **Korean**.

        <AnswerExample>
        프린터 설정 방법에 대해 안내드리겠습니다. 아래의 단계를 따라 진행해 보세요:

        1. 시작 메뉴에서 '제어판'을 열고 '장치 및 프린터'를 클릭한 후 '프린터 추가'를 선택합니다.
        2. '원하는 프린터가 목록에 없습니다'를 클릭합니다.
        3. '수동 설정으로 로컬 프린터 또는  네트워크 프린터 추가'를 선택하고 다음을 클릭합니다.
        4. '새 포트 만들기'를 선택한 후 'Standard TCP/IP Port'를 선택하고 다음을 클릭합니다.
        5. '호스트 이름 또는 IP 주소(A):'에 층별 프린터의 IP를 입력하고 다음을 클릭합니다.
        6. 검색 후 프린터 드라이버 설치에서 '디스크 있음' 버튼을 클릭합니다.
        7. '복사 할 제조업체 파일 위치'에 아래 경로를 입력하고 '찾아보기'를 클릭합니다:
           - \midasfile\800_자료\850_Program\900.[기타] 드라이버
           - 각 프린터 폴더에 맞는 경로로 이동하여 최종 inf파일을 선택하고 설치를 진행합니다.
        8. 층별 프린터 모델에 맞는 프린터를 선택하고 다음을 클릭합니다.
        9. 프린터 이름을 입력하고 다음을 클릭한 후, '프린터 공유 안 함'을 선택하고 완료합니다.

        [관련 문서](http://mitsm.midasit.com/articles/29)
        </AnswerExample>
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

제공된 관련 정보를 바탕으로 사용자의 질문에 답변해 주세요. 제공된 답변이 있다면 그대로 제공하세요. 만약 제공된 정보에서 질문에 대한 답변을 찾을 수 없다면, 제공한 정보는 말하지 말고, 정보가 없어 답변할 수 없다고 말씀해 주세요."""
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