"""
데이터 분석, 시각화 및 ML
train.py가 저장한 model.pkl을 로드하고, main.py나 crud.py가 전달한 데이터를 받아 성공 확률을 예측하여 반환
"""

import joblib
import pandas as pd
import numpy as np
from typing import Optional
from src.database import SessionLocal, Quest
from sentence_transformers import SentenceTransformer # 임베딩 객체 사용을 위한 임포트 추가

# train.py가 저장한 모델 파일 경로
MODEL_PATH = "model/model.pkl"

# 서버 시작 시 모델을 메모리에 로드하여 저장할 전역 변수
ML_MODEL = None 
EMBEDDER = None 

def get_user_success_rate(user_id: int):
    db = SessionLocal()
    quests = db.query(Quest).filter(Quest.user_id == user_id).all()
    db.close()
    if not quests:
        return 0.5
    completed = sum(1 for q in quests if q.completed)
    return completed / len(quests)

def load_ml_model():
    """joblib 파일을 로드하여 전역 변수 ML_MODEL과 EMBEDDER에 저장합니다."""
    global ML_MODEL, EMBEDDER
    try:
        # train.py에서 (model, embedder) 튜플을 저장했다고 가정하고 로드
        loaded_objects = joblib.load(MODEL_PATH)
        
        # 로드된 객체가 튜플인지 확인
        if isinstance(loaded_objects, tuple) and len(loaded_objects) == 2:
            ML_MODEL, EMBEDDER = loaded_objects
        else:
            # 튜플이 아니면, 로드된 객체를 모델로 간주하고 임베딩 객체는 수동 로드
            ML_MODEL = loaded_objects
            # 임베딩 객체는 매번 수동 로드하면 성능이 저하되므로,
            EMBEDDER = SentenceTransformer('all-MiniLM-L6-v2') 
            print("경고: 모델 파일에 임베딩 객체가 포함되지 않았습니다. 임베딩 객체를 수동 로드합니다.")
            
        print("ML 모델과 임베딩 객체가 성공적으로 로드되었습니다.")
        return ML_MODEL
    except FileNotFoundError:
        print(f"오류: 모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다. train.py를 먼저 실행하세요.")
        return None
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None

# crud.py가 호출하는 표준 함수 이름(predict_success_rate)으로 변경
def predict_success_rate(user_id: int, quest_name: str, duration: Optional[int], difficulty: Optional[int], category: Optional[str] = None):
    """
    quest_name (한국어 포함)을 받아 의미 기반 임베딩 후 성공 확률을 예측합니다.
    """
    try:
        model, embedder = joblib.load(MODEL_PATH)
        print("✅ 모델 및 임베딩 로드 완료")
    except:
        print("⚠️ 모델 파일을 찾을 수 없습니다. train.py를 먼저 실행하세요.")
        return 0.5

    # 사용자 과거 성공률
    user_rate = get_user_success_rate(user_id)

    # quest_name 임베딩 (한국어 문장 의미 반영)
    emb = embedder.encode(str(quest_name))

    # feature 구성
    row = {
        "user_id": user_id,
        "user_success_rate": user_rate,
        "days": duration or 0,
        "difficulty": difficulty or 1,
        "success_rate": user_rate,
        "category_" + (category or "general"): 1
    }

    for c in ["health", "study", "exercise", "reading", "work", "hobby", "general"]:
        if "category_" + c not in row:
            row["category_" + c] = 0

    for i, v in enumerate(emb):
        row[f"emb_{i}"] = v

    X = pd.DataFrame([row])
    
    try:
        prob = model.predict_proba(X)[0][1]
    except Exception as e:
        print(f"⚠️ 예측 중 오류 발생: {e}")
        return 0.5

    return round(float(np.clip(prob, 0.05, 0.95)), 3)

