'''
데이터 분석, 시각화 및 ML
utils.py의 load_data()를 사용하여 데이터를 불러오고, completed 컬럼의 평균값을 계산
이 평균값을 pickle 라이브러리를 사용하여 model/model.pkl 파일로 저장하여, 추후 API에서 사용하도록 준비
'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sentence_transformers import SentenceTransformer
from joblib import dump
from src.utils import load_data
import numpy as np

MODEL_PATH = "model/model.pkl"

def train_model():
    print("--- 1. 데이터 로드 및 임베딩 시작 ---")
    data = load_data()
    expected_cols = ['user_id', 'days', 'difficulty', 'completed', 'name']
    for c in expected_cols:
        if c not in data.columns:
            raise ValueError(f"누락된 컬럼: {c}")

    data['user_success_rate'] = data.groupby('user_id')['completed'].transform('mean').fillna(0.5)

    # SentenceTransformer 임베딩
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    quest_embeddings = np.vstack(data['name'].apply(lambda x: embedder.encode(str(x))).values)
    emb_df = pd.DataFrame(quest_embeddings, columns=[f"emb_{i}" for i in range(quest_embeddings.shape[1])])
    data = pd.concat([data.reset_index(drop=True), emb_df], axis=1)

    # features: 수치형 + 임베딩
    numeric_features = ['user_success_rate', 'days', 'difficulty']
    embedding_features = [f"emb_{i}" for i in range(quest_embeddings.shape[1])]
    all_features = numeric_features + embedding_features
    target = 'completed'

    data = data.fillna(data.mean(numeric_only=True))

    X = data[all_features]
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print("--- 2. 모델 학습 시작 ---")
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_features)
    ], remainder='passthrough')

    base_clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    clf = CalibratedClassifierCV(base_clf, cv=3)

    model = Pipeline([
        ('pre', preprocessor),
        ('clf', clf)
    ])
    model.fit(X_train, y_train)

    print("--- 3. 성능 평가 ---")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"✅ 정확도: {accuracy_score(y_test, y_pred):.3f}")
    print(f"✅ F1: {f1_score(y_test, y_pred):.3f}")
    print(f"✅ ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    dump((model, embedder), MODEL_PATH)
    print(f"모델 저장 완료 → {MODEL_PATH}")

if __name__ == "__main__":
    train_model()

# python -m src.train
