'''
데이터 분석, 시각화 및 ML
utils.py의 load_data()를 사용하여 데이터를 불러오고, completed 컬럼의 평균값을 계산
이 평균값을 pickle 라이브러리를 사용하여 model/model.pkl 파일로 저장하여, 추후 API에서 사용하도록 준비
'''

import pandas as pd
# scikit-learn에서 로지스틱 회귀 모델과 모델 저장/로드 도구를 가져옵니다.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from joblib import dump # 모델 저장에 pickle 대신 joblib 사용
from src.utils import load_data # src 폴더 외부에서 내부 모듈 임포트 시 src.utils

MODEL_PATH = "model/model.pkl"

def train_model():
    print("--- 1. 데이터 로드 및 전처리 시작 ---")
    data = load_data()
    
    # ML 학습을 위한 피처(Features) 선택
    # user_id, duration, difficulty 등을 피처로 사용하고 'completed'를 예측합니다.
    # NOTE: 'quest'와 같은 문자열 데이터는 One-Hot 인코딩이 필요하지만, 여기서는 간단화를 위해 숫자 피처만 사용합니다.
    features = ['user_id', 'duration', 'difficulty']
    
    # 널(Null) 값 처리: ML 모델에 넣기 전에 결측치를 평균으로 채웁니다.
    data['duration'].fillna(data['duration'].mean(), inplace=True)
    data['difficulty'].fillna(data['difficulty'].mean(), inplace=True)

    X = data[features] # 입력 피처
    y = data['completed'] # 목표 변수 (성공 여부: 0 또는 1)
    
    # 훈련 세트와 테스트 세트로 분리 (모델 검증을 위해)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"훈련 데이터 크기: {len(X_train)}, 테스트 데이터 크기: {len(X_test)}")
    
    print("--- 2. 로지스틱 회귀 모델 학습 시작 ---")
    # 로지스틱 회귀 모델 초기화 및 학습
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    
    # 모델 성능 평가 (선택 사항이지만 중요)
    accuracy = model.score(X_test, y_test)
    print(f"모델 테스트 정확도 (Accuracy): {accuracy:.2f}")

    print("--- 3. 학습된 모델 저장 ---")
    # joblib을 사용하여 모델 객체 전체를 파일로 저장
    dump(model, MODEL_PATH)
    print(f"✅ 모델이 성공적으로 저장되었습니다: {MODEL_PATH}")

if __name__ == "__main__":
    # model 폴더가 없으면 만들어야 합니다.
    import os
    if not os.path.exists('model'):
        os.makedirs('model')
        
    train_model()

