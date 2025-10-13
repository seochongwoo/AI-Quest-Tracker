'''
데이터 분석, 시각화 및 ML
utils.py의 load_data()를 사용하여 데이터를 불러오고, completed 컬럼의 평균값을 계산
이 평균값을 pickle 라이브러리를 사용하여 model/model.pkl 파일로 저장하여, 추후 API에서 사용하도록 준비
'''
import pandas as pd
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from src.utils import load_data
from src.database import init_db

MODEL_PATH = "model/model.pkl"

def train_model():
    print("--- 1. 데이터 로드 및 임베딩 시작 ---")
    df = load_data()
    if df.empty:
        raise ValueError("데이터셋이 비어 있습니다. seed.py를 먼저 실행하세요.")

    # 누락 컬럼 확인
    required = ["name", "days", "difficulty", "completed"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"누락된 컬럼: {c}")

    # 한국어 멀티언어 임베딩 모델
    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # quest 컬럼 임베딩 (한국어 의미 반영)
    print("임베딩 생성 중 (한국어 포함)...")
    embeddings = embedder.encode(df["name"].astype(str).tolist(), show_progress_bar=True)
    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
    df = df.drop(columns=["name"], errors="ignore")

    # 훈련용 입력과 출력
    X = df.drop(columns=["completed"])
    y = df["completed"]

    # 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 수치형 컬럼 처리
    num_cols = ["days", "difficulty"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), num_cols)
        ],
        remainder="passthrough"
    )

    # 모델 구성 (RandomForest + 확률 보정)
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=15,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    model = Pipeline([
        ("pre", preprocessor),
        ("clf", CalibratedClassifierCV(rf, cv=3))
    ])

    print("--- 2. 모델 학습 중 ---")
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    print(f"✅ 모델 학습 완료. 테스트 정확도: {score:.3f}")

    print("--- 3. 모델 저장 중 ---")
    joblib.dump((model, embedder), MODEL_PATH)
    print(f"✅ 모델 저장 완료: {MODEL_PATH}")

if __name__ == "__main__":
    init_db()
    train_model()

# python -m src.train
