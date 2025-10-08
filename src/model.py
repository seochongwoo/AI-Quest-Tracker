"""
데이터 분석, 시각화 및 ML
train.py가 저장한 model.pkl을 로드하고, main.py나 crud.py가 전달한 데이터를 받아 성공 확률을 예측하여 반환
"""

import joblib
import pandas as pd
import numpy as np
from src.database import SessionLocal, Quest

MODEL_PATH = "model/model.pkl"

def get_user_success_rate(user_id: int):
    db = SessionLocal()
    quests = db.query(Quest).filter(Quest.user_id == user_id).all()
    db.close()
    if not quests:
        return 0.5
    completed = sum(1 for q in quests if q.completed)
    return completed / len(quests)

def predict_success_rate_text(user_id: int, quest_name: str, duration: int, difficulty: int):
    try:
        model, embedder = joblib.load(MODEL_PATH)
    except:
        print("⚠️ 모델 파일을 찾을 수 없습니다. train.py를 먼저 실행하세요.")
        return 0.5

    user_rate = get_user_success_rate(user_id)
    emb = embedder.encode(str(quest_name))
    row = {
        'user_success_rate': user_rate,
        'days': duration,
        'difficulty': difficulty
    }
    for i, v in enumerate(emb):
        row[f'emb_{i}'] = v

    X = pd.DataFrame([row])
    prob = model.predict_proba(X)[0][1]

    # 값 보정
    return round(float(np.clip(prob, 0.05, 0.95)), 3)


