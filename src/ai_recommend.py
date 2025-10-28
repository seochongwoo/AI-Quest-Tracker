from google import genai
from typing import Optional
import os

# 클라이언트 객체를 저장할 전역 변수. 초기화 전에는 None
GEMINI_CLIENT: Optional[genai.Client] = None

# 초기화 시도 상태를 기록하여 중복 경고를 방지
_INITIALIZED_ATTEMPTED = False

def get_gemini_client() -> Optional[genai.Client]:
    """
    Gemini 클라이언트를 초기화하거나, 이미 초기화된 클라이언트를 반환합니다.
    환경 변수 GEMINI_API_KEY가 없거나 초기화에 실패하면 None을 반환합니다.
    """
    global GEMINI_CLIENT, _INITIALIZED_ATTEMPTED
    
    # 1. 이미 성공적으로 초기화된 경우
    if GEMINI_CLIENT is not None:
        return GEMINI_CLIENT
        
    # 2. 이미 한 번 시도했으나 실패한 경우 (두 번째 호출부터 경고 생략)
    if _INITIALIZED_ATTEMPTED:
        return None 
    
    _INITIALIZED_ATTEMPTED = True # 초기화 시도 시작 플래그 설정
    
    # 3. API 키 환경 변수 확인
    if "GEMINI_API_KEY" not in os.environ:
        print("⚠️ 경고: GEMINI_API_KEY 환경 변수가 설정되지 않았습니다. AI 기능을 사용할 수 없습니다.")
        return None

    # 4. 클라이언트 초기화 시도
    try:
        GEMINI_CLIENT = genai.Client()
        print("✅ Gemini 클라이언트 초기화 성공.")
        return GEMINI_CLIENT
    except Exception as e:
        print(f"❌ 경고: Gemini 클라이언트 초기화 중 오류 발생: {e}")
        return None

# Gemini API 호출하여 퀘스트 성공률에 기반한 맞춤형 조언 생성
def generate_ai_recommendation(
    quest_name: str, 
    duration: int, 
    difficulty: int, 
    success_rate: float
) -> str:
    # 중앙 집중화된 get_gemini_client 함수를 통해 클라이언트 객체 가져오기
    client: Optional[genai.Client] = get_gemini_client() 
    
    if client is None:
        return "AI 추천 기능을 사용할 수 없습니다. API 키 설정이 필요합니다."
        
    percent = round(success_rate * 100)
    
    # 퀘스트 정보를 기반으로 구체적인 코치 역할 프롬프트 작성
    prompt_template = f"""
    당신은 퀘스트 성공률을 높여주는 최고의 AI 코치입니다.
    사용자의 목표: "{quest_name}" (예상 기간: {duration}일, 난이도: {difficulty}/5)
    당신의 AI 예측 성공률은: {percent}% 입니다.
    
    사용자에게 이 퀘스트를 성공적으로 완수할 수 있도록 구체적이고 현실적인 맞춤형 조언을 2~3 문장으로 작성해 주세요. 
    특히, 성공률이 50% 미만이라면, 목표를 주간/일간 작은 단계로 나누는 구체적인 방법(예시 포함)을 제안해 주세요.
    응답은 조언 내용만 포함해야 하며, 인사말, 마무리 문구("감사합니다" 등), 그리고 'AI 코치'와 같은 지칭어는 생략합니다.
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_template,
        )
        return response.text.strip()
    except Exception as e:
        # FastAPI 서버 로그에 오류 출력
        print(f"Gemini API 호출 오류: {e}")
        return "AI 코치 서비스와의 연결에 문제가 발생했습니다. (API 오류)"