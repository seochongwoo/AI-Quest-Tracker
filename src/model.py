"""
데이터 분석, 시각화 및 ML
train.py가 저장한 model.pkl을 로드하고, main.py나 crud.py가 전달한 데이터를 받아 성공 확률을 예측하여 반환
"""

import joblib
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
from src.database import SessionLocal, Quest, User
from sentence_transformers import SentenceTransformer # 임베딩 객체 사용을 위한 임포트 추가

KNOWN_CATEGORIES = ['reading', 'study', 'exercise', 'work', 'hobby', 'health', 'general', 'none']

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

def get_user_stats_for_prediction(user_id: int) -> Dict[str, Any]:
    """DB의 User 테이블에서 ML 모델이 요구하는 모든 통계 피처를 가져옵니다."""
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    db.close()
    
    if user:
        return {
            'user_id': user.id,
            'total_quests': user.total_quests or 0, # None 방지
            'completed_quests': user.completed_quests or 0, # None 방지
            'streak_days': user.streak_days or 0, # None 방지
            'preferred_category': user.preferred_category or 'none',
            'average_success_rate': user.average_success_rate or 0.5, # None 방지
            'user_success_rate': user.average_success_rate or 0.5 # None 방지
        }
    
    # 사용자가 DB에 없는 경우 기본값 반환
    return {
        'user_id': user_id, 
        'total_quests': 0, 
        'completed_quests': 0, 
        'streak_days': 0, 
        'preferred_category': 'none', 
        'average_success_rate': 0.5,
        'user_success_rate': 0.5
    }

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
def predict_success_rate(
    user_id: int, 
    quest_name: str, 
    duration: Optional[int], 
    difficulty: Optional[int],
    category: Optional[str] = None, 
    motivation: Optional[str] = None 
) -> float:
    """
    입력 피처를 사용하여 퀘스트 성공 확률 (0.0 ~ 1.0)을 예측합니다.
    """
    global ML_MODEL, EMBEDDER
    try:
        model, embedder = joblib.load(MODEL_PATH)
        print("✅ 모델 및 임베딩 로드 완료")
    except:
        print("⚠️ 모델 파일을 찾을 수 없습니다. train.py를 먼저 실행하세요.")
        return 0.5

    # 1. 사용자 통계 피처 로드
    user_stats = get_user_stats_for_prediction(user_id)
    
    # 2. 퀘스트 피처 및 누락된 값 처리
    quest_features = {
        'days': duration if duration is not None and duration > 0 else 5,
        'difficulty': difficulty if difficulty is not None else 3,
        'success_rate': user_stats['average_success_rate'], 
        # category는 train.py에서 OHE되었으므로, 예측 시에도 OHE에 사용
        'category': category or user_stats['preferred_category'] or 'general', 
        'preferred_category': user_stats['preferred_category'],
        'name': quest_name,
        'motivation': motivation or "",
    }
    
    # 3. 임베딩 생성 (train.py와 동일한 방식으로 text_features 구성)
    text_features = quest_features['name'] + " " + quest_features['motivation']
    emb = EMBEDDER.encode(text_features)
    
    # 4. 입력 DataFrame 구성
    input_row = {**user_stats, **quest_features}
    df = pd.DataFrame([input_row])

    # 5. One-Hot Encoding (train.py와 동일하게 적용)
    cols_to_dummy = ["category", "preferred_category"]
    
    # [수정] 모든 카테고리를 포함하는 가상의 DataFrame을 생성하여 One-Hot Encoding을 수행합니다.
    # 이렇게 하면 학습 시에 존재했던 모든 OHE 컬럼이 생성됩니다.
    dummy_df = pd.DataFrame([
        {'category': c, 'preferred_category': c} for c in KNOWN_CATEGORIES
    ])
    
    # 5-1. OHE 컬럼명 미리 확보 (drop_first=True 고려)
    full_dummy_cols = pd.get_dummies(dummy_df, columns=cols_to_dummy, drop_first=True, prefix_sep='_').columns

    # 5-2. 실제 예측 데이터에 OHE 적용
    df = pd.get_dummies(df, columns=cols_to_dummy, drop_first=True, prefix_sep='_') 
    
    # 6. 임베딩 피처 추가
    emb_cols = [f"emb_{i}" for i in range(emb.shape[0])]
    emb_df = pd.DataFrame(emb.reshape(1, -1), columns=emb_cols)
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    # 7. 불필요한 컬럼 제거
    # user_id, name, motivation, last_completed_at (병합되지 않았으므로)
    df = df.drop(columns=['user_id', 'name', 'motivation'], errors='ignore')

    # 8. 학습 데이터셋의 컬럼 구조에 맞게 보정
    
    # 모든 예상 OHE 컬럼을 DataFrame에 추가하고 0으로 채움
    for col in full_dummy_cols:
        # 이미 존재하는 컬럼은 무시하고, OHE 이후 남은 컬럼만 추가
        if col not in df.columns and col.startswith(('category_', 'preferred_category_')):
             df[col] = 0
             
    # train.py에서 사용된 수치형 컬럼 목록
    required_num_cols = [
        "days", "difficulty", "success_rate", "user_success_rate", 
        "total_quests", "completed_quests", "streak_days", 
        "average_success_rate"
    ]
    
    # 임베딩 컬럼을 제외한 모든 OHE 컬럼 이름 추출
    ohe_cols = sorted([col for col in df.columns if col.startswith(('category_', 'preferred_category_'))])
    
    # 최종적으로 필요한 모든 컬럼
    final_cols = [c for c in required_num_cols if c in df.columns] + ohe_cols + emb_cols
    
    # DataFrame을 예상 컬럼 순서로 재정렬하고, 필요한 경우 0으로 채웁니다.
    df = df.reindex(columns=final_cols, fill_value=0)
    
    try:
        # 9. 예측 수행
        prediction = ML_MODEL.predict_proba(df)[:, 1][0]
        return float(np.clip(prediction, 0.05, 0.95))
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        # 오류 발생 시 user's average success rate로 대체
        return float(np.clip(user_stats['average_success_rate'], 0.05, 0.95))

