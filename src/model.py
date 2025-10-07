"""
데이터 분석, 시각화 및 ML
train.py가 저장한 model.pkl을 로드하고, main.py나 crud.py가 전달한 데이터를 받아 성공 확률을 예측하여 반환
"""

import joblib
import pandas as pd
from typing import Optional
import numpy as np
from src.database import SessionLocal, Quest

# train.py가 저장한 모델 파일 경로
MODEL_PATH = "model/model.pkl"

# 서버 시작 시 모델을 메모리에 로드하여 저장할 변수
ML_MODEL = None 

def load_ml_model():
    """joblib 파일을 로드하여 전역 변수 ML_MODEL에 저장합니다."""
    global ML_MODEL
    try:
        ML_MODEL = joblib.load(MODEL_PATH)
        print("ML 모델이 성공적으로 로드되었습니다.")
        return ML_MODEL
    except FileNotFoundError:
        print(f"오류: 모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다. train.py를 먼저 실행하세요.")
        return None
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None

#  유틸 함수: 사용자 평균 성공률
def get_user_success_rate(user_id: int):
    db = SessionLocal()
    quests = db.query(Quest).filter(Quest.user_id == user_id).all()
    db.close()

    if not quests:
        return 0.5  # 기본값 (새 사용자)
    
    completed = sum(1 for q in quests if q.completed)
    return completed / len(quests)

#  난이도 + 기간 보정
def adjust_by_difficulty_duration(difficulty: int, duration: int):
    diff_factor = 1 - (difficulty - 3) * 0.1   
    dur_factor = 1 - min(duration / 100, 0.15) # 기간이 길면 15%까지 감소
    return max(0.2, min(1.0, diff_factor * dur_factor))

# ✅ 최종 예측 함수
def predict_success_rate(user_id: int, quest_name: str, duration: int, difficulty: int) -> float:
    """
    난이도, 기간, 과거 사용자 성공률 기반 + ML 모델 예측값
    """
    try:
        model = joblib.load(MODEL_PATH)
    except:
        # 모델이 없으면 기본값 반환
        return 0.5

    # AI 모델 기본 예측값 (오류 시 fallback)
    try:
        X = np.array([[user_id, duration, difficulty]]).astype(float)
        model_pred = model.predict_proba(X)[0][1]
    except Exception:
        model_pred = 0.5

    # 사용자 과거 평균 성공률
    user_rate = get_user_success_rate(user_id)
    # 난이도 / 기간 조정값
    diff_dur_factor = adjust_by_difficulty_duration(difficulty, duration)

    # ✅ 가중 평균으로 최종 성공률 계산
    final_rate = (
        0.5 * model_pred + 
        0.3 * user_rate + 
        0.2 * diff_dur_factor
    )

    # 확률 범위 [0,1] 보정
    return round(float(max(0.05, min(final_rate, 0.95))), 3)

# 서버 시작 시 자동으로 모델 로드
load_ml_model()

# 테스트용 함수 (선택 사항)
if __name__ == "__main__":
    test_rate = predict_success_rate(user_id=1, duration=30, difficulty=4)
    print(f"테스트 사용자(ID 1)의 성공 확률: {test_rate:.4f}")