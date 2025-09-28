"""
데이터 분석, 시각화 및 ML
train.py가 저장한 model.pkl을 로드하고, main.py나 crud.py가 전달한 데이터를 받아 성공 확률을 예측하여 반환
"""

from joblib import load
import pandas as pd
from typing import Optional

# train.py가 저장한 모델 파일 경로
MODEL_PATH = "model/model.pkl"

# 서버 시작 시 모델을 메모리에 로드하여 저장할 변수
ML_MODEL = None 

def load_ml_model():
    """joblib 파일을 로드하여 전역 변수 ML_MODEL에 저장합니다."""
    global ML_MODEL
    try:
        # joblib을 사용해 모델 로드 (scikit-learn 모델 로드)
        ML_MODEL = load(MODEL_PATH)
        print("ML 모델이 성공적으로 로드되었습니다.")
        return ML_MODEL
    except FileNotFoundError:
        print(f"오류: 모델 파일 '{MODEL_PATH}'을 찾을 수 없습니다. train.py를 먼저 실행하세요.")
        return None
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None

def predict_success_rate(user_id: int, duration: Optional[int], difficulty: Optional[int]) -> float:
    """
    입력 피처를 사용하여 퀘스트 성공 확률 (0.0 ~ 1.0)을 예측합니다.
    """
    if ML_MODEL is None:
        # 모델이 로드되지 않았으면 임시 값 반환
        return 0.5 

    # 1. 누락된 값 처리 (train.py의 전처리 로직과 일치해야 함)
    # 현재는 단순하게 None이면 0 또는 임의의 값으로 처리합니다.
    days_val = duration if duration is not None else 5  # 임의의 평균값

    # 2. 모델 입력 형식 (DataFrame) 생성
    # train.py에서 사용했던 features의 순서와 이름을 정확히 맞춰야 합니다.
    input_data = pd.DataFrame([[user_id, days_val]], 
                              columns=['user_id', 'days'])
    
    # 3. 모델 예측 (확률 예측: predict_proba)
    # [0]은 실패 확률, [1]은 성공 확률입니다. 1 (성공)의 확률을 선택합니다.
    prediction_proba = ML_MODEL.predict_proba(input_data)[0][1]
    
    return float(prediction_proba)

# 서버 시작 시 자동으로 모델 로드
load_ml_model()

# 테스트용 함수 (선택 사항)
if __name__ == "__main__":
    test_rate = predict_success_rate(user_id=1, duration=30, difficulty=4)
    print(f"테스트 사용자(ID 1)의 성공 확률: {test_rate:.4f}")