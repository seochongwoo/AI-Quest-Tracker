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

    required = ["difficulty", "completed"]
    if "name" not in df.columns:
        if "quest_name" in df.columns:
            df.rename(columns={"quest_name": "name"}, inplace=True)
        else:
            df["name"] = "Unknown"

    if "days" not in df.columns:
        if "duration" in df.columns:
            df.rename(columns={"duration": "days"}, inplace=True)

    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("임베딩 생성 중 (한국어 포함)...")
    text_features = df["name"].astype(str) + " " + df.get("motivation", "")
    embeddings = embedder.encode(text_features.tolist(), show_progress_bar=True)
    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
    df = df.drop(columns=["name"], errors="ignore")

    # feature 다양화: success_rate, category 추가
    if "success_rate" in df.columns:
        df["success_rate"] = df["success_rate"].fillna(df["success_rate"].mean())
    if "category" in df.columns:
        df = pd.get_dummies(df, columns=["category"], drop_first=True)

    X = df.drop(columns=["completed"])
    y = df["completed"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    num_cols = ["days", "difficulty", "success_rate"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ]), [c for c in num_cols if c in X.columns])
        ],
        remainder="passthrough"
    )

    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=18,
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
