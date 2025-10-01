'''
데이터 분석, 시각화 및 ML
utils.py의 load_data()를 사용하여 데이터를 불러오고, completed 컬럼의 평균값을 계산
이 평균값을 pickle 라이브러리를 사용하여 model/model.pkl 파일로 저장하여, 추후 API에서 사용하도록 준비
'''

import pandas as pd
# scikit-learn에서 로지스틱 회귀 모델과 모델 저장/로드 도구를 가져옵니다.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from joblib import dump
from src.utils import load_data

MODEL_PATH = "model/model.pkl"

def train_model():
    print("--- 1. 데이터 로드 및 전처리 시작 ---")
    data = load_data()
    
    # user_id, duration, difficulty 등을 피처로 사용하고 'completed'를 예측합니다.
    DURATION_COL = 'days'
    QUEST_NAME_COL = 'quest' 
    features = ['user_id', DURATION_COL, QUEST_NAME_COL, 'difficulty']
    target = 'completed'
    
    categorical_features = [QUEST_NAME_COL]
    numerical_features = ['user_id', DURATION_COL, 'difficulty']
    # 널(Null) 값 처리: ML 모델에 넣기 전에 결측치를 평균으로 채웁니다.
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # ColumnTransformer를 사용하여 모든 전처리 단계를 결합합니다.
    preprocessor = ColumnTransformer(
        transformers=[
            # 'quest' 컬럼에 OneHotEncoder 적용
            ('cat', OneHotEncoder(handle_unknown='ignore'), [QUEST_NAME_COL]), 
            # 수치형 피처에 Imputer 및 Scaler 적용
            ('num', numerical_transformer, ['user_id', DURATION_COL, 'difficulty'])
        ],
        remainder='drop'
    )

    # 훈련 데이터 및 목표 변수 설정
    if 'difficulty' not in data.columns:
        # DB 구조에 맞추기 위해 'difficulty' 컬럼을 추가하고, 일단 평균값 3으로 채웁니다.
        # 실제 데이터가 들어오면 이 평균이 사용됩니다.
        data['difficulty'] = 3 
    
    X = data[features] # 입력 피처
    y = data[target] # 목표 변수 (성공 여부: 0 또는 1)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"훈련 데이터 크기: {len(X_train)}, 테스트 데이터 크기: {len(X_test)}")
    
    print("--- 2. 랜덤 포레스트 모델 학습 시작 ---")
    
    # 랜덤 포레스트 파이프라인
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,       # 트리 개수
            max_depth=None,         # 제한 없음 (자동 조정)
            random_state=42,
            n_jobs=-1               # 병렬 처리
        ))
    ])
    model.fit(X_train, y_train)

    print("--- 3. 성능 평가 및 저장 ---")
    
    # 성능 평가
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    # F1-Score는 이진 분류에서 유용합니다.
    f1 = f1_score(y_test, y_pred, zero_division=0) 
    # ROC-AUC는 클래스 불균형에 강한 성능 지표입니다.
    roc = roc_auc_score(y_test, y_proba) 

    print(f"✅ 정확도 (Accuracy): {acc:.2f}")
    print(f"✅ F1 점수: {f1:.2f}")
    print(f"✅ ROC-AUC: {roc:.2f}")

    # 모델 저장
    dump(model, MODEL_PATH)
    print(f"모델 저장 완료 → {MODEL_PATH}")


if __name__ == "__main__":
    # model 폴더가 없으면 만들어야 합니다.
    import os
    if not os.path.exists('model'):
        os.makedirs('model')
        
    train_model()

# python -m src.train