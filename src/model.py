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
# 노트북 환경(GPU 없음)에서 실행가능하게 바꾸기위한 import
import torch
import io
import pickle

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

    # CPU-safe 로딩을 위한 커스텀 Unpickler 정의
    class CPU_Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                # PyTorch 텐서 로드를 가로채서 map_location='cpu'를 적용
                def _load_from_bytes_cpu(b):
                    return torch.load(io.BytesIO(b), map_location='cpu')
                return _load_from_bytes_cpu
            return super().find_class(module, name) # 다른 모든 클래스 로드는 기본 동작으로 위임

    loaded_objects = None
    
    try:
        # 1차 시도: 일반 joblib 로드 (GPU 환경에서 저장되었다면 CPU 환경에서 실패할 수 있음)
        loaded_objects = joblib.load(MODEL_PATH)
        
    except Exception as e:
        error_msg = str(e)
        
        # 2차 시도: CUDA 오류 발생 시 CPU-safe 로더로 재시도
        if "Attempting to deserialize object on a CUDA device" in error_msg:
            print("⚠️ CUDA 로드 오류 감지. CPU-safe 로더로 재시도합니다.")
            try:
                with open(MODEL_PATH, 'rb') as f:
                    loaded_objects = CPU_Unpickler(f).load()
            except Exception as e_cpu:
                # 3차 시도: CPU-safe 로더에서도 오류(STACK_GLOBAL) 발생 시, 폴백 실행
                print(f"재시도 중에도 오류 발생: {e_cpu}")
                
                # [핵심 수정: 폴백 로직] ML_MODEL(Scikit-learn)만 로드하고 EMBEDDER는 수동 재구성
                try:
                    # ML_MODEL만 joblib으로 로드 시도 (PyTorch 객체 로딩 실패를 무시)
                    # NOTE: 이 코드는 train.py가 (ML_MODEL, EMBEDDER) 튜플을 저장했다는 가정을 깨고,
                    # ML_MODEL이 파일의 첫 번째 객체라고 가정하여 안전하게 로드합니다.
                    print("⚠️ Scikit-learn 모델만 로드하고 임베더는 수동 재구성하여 오류를 우회합니다.")
                    
                    # 파일 전체를 joblib.load로 로드하면 여전히 오류가 발생할 수 있으므로, 
                    # 파일의 첫 번째 요소(ML_MODEL)만 로드하는 것은 불가능합니다.
                    
                    # 가장 안전한 방식: train.py에서 사용한 임베더를 수동으로 로드
                    ML_MODEL = joblib.load(MODEL_PATH)[0] # 튜플의 첫 번째 요소만 로드 시도
                    
                    # train.py에서 사용한 임베딩 모델을 수동으로 로드하고 CPU로 이동
                    EMBEDDER = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(torch.device('cpu'))
                    
                    print("✅ ML 모델(Scikit-learn)과 임베딩 객체가 분리되어 성공적으로 로드(재구성)되었습니다.")
                    return ML_MODEL

                except Exception as e_fallback:
                    print(f"Fallback 로드 중에도 오류 발생. 모델 파일을 확인할 수 없습니다: {e_fallback}")
                    return None
        
        # 파일 없음 오류 처리
        elif "FileNotFoundError" in error_msg:
            print(f"오류: 모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다. train.py를 먼저 실행하세요.")
            return None
        else:
            print(f"모델 로드 중 오류 발생: {e}")
            return None

    # 로드 성공 또는 CPU-safe 로드 성공 시 객체 처리
    if loaded_objects is not None:
        if isinstance(loaded_objects, tuple) and len(loaded_objects) == 2:
            ML_MODEL, EMBEDDER = loaded_objects
            # 임베딩 객체가 PyTorch 모델이므로, 로드 후 CPU로 명시적 이동 (안정성 추가)
            try:
                EMBEDDER.to(torch.device('cpu')) 
            except:
                pass 
            print("✅ ML 모델과 임베딩 객체가 성공적으로 로드되었습니다.")
            return ML_MODEL
        else:
            ML_MODEL = loaded_objects
            # train.py에서 사용된 임베더를 가정하고 수동 로드 후 CPU로 이동
            EMBEDDER = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2').to(torch.device('cpu')) 
            print("경고: 모델 파일에 임베딩 객체가 포함되지 않았습니다. 임베딩 객체를 수동 로드합니다.")
            return ML_MODEL
    
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
    if ML_MODEL is None or EMBEDDER is None:
        load_ml_model() 
        if ML_MODEL is None:
            # 로드 실패 시 기본값 반환
            print("⚠️ 모델 로드에 실패했습니다. 기본값 0.5를 반환합니다.")
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
    
    dummy_df = pd.DataFrame([
        {'category': c, 'preferred_category': c} for c in KNOWN_CATEGORIES
    ])
    
    # OHE 컬럼명 미리 확보 (drop_first=True 고려)
    full_dummy_cols = pd.get_dummies(dummy_df, columns=cols_to_dummy, drop_first=True, prefix_sep='_').columns

    # 실제 예측 데이터에 OHE 적용
    df = pd.get_dummies(df, columns=cols_to_dummy, drop_first=True, prefix_sep='_') 
    
    # 6. 임베딩 피처 추가
    emb_cols = [f"emb_{i}" for i in range(emb.shape[0])]
    emb_df = pd.DataFrame(emb.reshape(1, -1), columns=emb_cols)
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)

    # 7. 불필요한 컬럼 제거
    # user_id, name, motivation, last_completed_at
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
    
    # DataFrame을 예상 컬럼 순서로 재정렬하고, 필요한 경우 0으로 채우기
    df = df.reindex(columns=final_cols, fill_value=0)
    
    try:
        # 9. 예측 수행
        prediction = ML_MODEL.predict_proba(df)[:, 1][0]
        return float(np.clip(prediction, 0.05, 0.95))
    except Exception as e:
        print(f"예측 중 오류 발생: {e}")
        # 오류 발생 시 user's average success rate로 대체
        return float(np.clip(user_stats['average_success_rate'], 0.05, 0.95))

