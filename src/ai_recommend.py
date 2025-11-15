from google import genai
from typing import Optional
import os

# 클라이언트 객체를 저장할 전역 변수. 초기화 전에는 None
GEMINI_CLIENT: Optional[genai.Client] = None

# 초기화 시도 상태를 기록하여 중복 경고를 방지
_INITIALIZED_ATTEMPTED = False

def get_gemini_client() -> Optional[genai.Client]:
    """
    Gemini 클라이언트를 초기화하거나, 이미 초기화된 클라이언트를 반환
    환경 변수 GEMINI_API_KEY가 없거나 초기화에 실패하면 None을 반환
    """
    global GEMINI_CLIENT, _INITIALIZED_ATTEMPTED
    
    # 1. 이미 성공적으로 초기화된 경우
    if GEMINI_CLIENT is not None:
        return GEMINI_CLIENT
        
    # 2. 이미 한 번 시도했으나 실패한 경우
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
    consistency_score: int,
    risk_aversion_score: int,
    total_quests: int,
    completed_quests: int,
    preferred_category: str = None
) -> str:
    # 중앙 집중화된 get_gemini_client 함수를 통해 클라이언트 객체 가져오기
    client: Optional[genai.Client] = get_gemini_client() 
    
    if client is None:
        return "AI 추천 기능을 사용할 수 없습니다. API 키 설정이 필요합니다."
        
    
    # 퀘스트 정보를 기반으로 구체적인 코치 역할 프롬프트 작성
    prompt_template = f"""
    당신은 사용자의 행동 패턴과 성향을 분석하여 퀘스트 성공률을 높이는 AI 코치입니다.

    [사용자 프로필]
    - 전체 퀘스트 수행 수: {total_quests}
    - 완료한 퀘스트 수: {completed_quests}
    - 꾸준함 점수(consistency): {consistency_score}/5
    - 도전적인 성향(risk_aversion): {risk_aversion_score}/5
    - 선호 카테고리: {preferred_category or "없음"}

    [현재 퀘스트]
    - 이름: "{quest_name}"
    - 예상 기간: {duration}일
    - 난이도: {difficulty}/5

    당신의 목표는 사용자의 성향과 경험 수준을 바탕으로, 
    이번 퀘스트를 어떻게 접근해야 지속 가능한 성장을 이룰 수 있을지 조언하는 것입니다.

    작성 지침:
    1. 조언은 2~3문장으로 구성합니다.
    2. 전체 퀘스트 대비 완료한 퀘스트가 적다면 “작은 성공의 축적”, “습관의 루틴화”를 강조합니다.
    3. 전체 퀘스트 대비 완료한 퀘스트가 높다면 “난이도 조절 제안”과 “지속적 성장”을 제안합니다.
    4. 꾸준함 점수가 낮다면 “시간 루틴 만들기”, “기록 습관화” 중심으로,
       위험 회피 점수가 낮다면 “도전의 보상과 리스크 관리”를 중심으로 합니다.
    5. 말투는 따뜻하고 실질적이며, 인사말이나 결론 문구는 생략합니다.
    6. 마지막으로 사용자의 전체적인 정보에 따라 추천 퀘스트 1~2개를 추천합니다.
    7. 마지막 추천 퀘스트는 사용자가 요청한 퀘스트에 관련된 것만 추천합니다.
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