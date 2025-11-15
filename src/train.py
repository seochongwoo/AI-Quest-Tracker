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
from src.database import init_db, SessionLocal, User, QuestHistory, Quest
from sqlalchemy import func
from sqlalchemy.sql import case
import torch

MODEL_PATH = "model/model.pkl"

# DB에서 사용자 통계를 계산하고 반환(User table 갱신)
def get_user_statistics_df(db):

    # 1. Quest 데이터를 기반으로 최신 통계를 계산하고 DB에 업데이트
    quest_stats = db.query(
        Quest.user_id,
        func.count(Quest.id).label('calc_total_quests'),
        # completed=True(1)인 퀘스트의 합계
        func.sum(case((Quest.completed == True, 1), else_=0)).label('calc_completed_quests'),
        func.avg(Quest.success_rate).label('calc_avg_success_rate'),
    ).group_by(
        Quest.user_id
    ).all()

    category_counts = db.query(
        Quest.user_id,
        Quest.category,
        func.count(Quest.category).label('category_count')
    ).group_by(
        Quest.user_id,
        Quest.category
    ).order_by(
        Quest.user_id,
        func.count(Quest.category).desc()
    ).all()
    
    preferred_categories = {}
    last_user_id = None
    for user_id, category, count in category_counts:
        # User ID가 바뀔 때마다 가장 많은 카테고리(desc로 정렬됨)를 선택
        if user_id != last_user_id:
            preferred_categories[user_id] = category
            last_user_id = user_id

    # User 테이블에 계산된 값 업데이트 및 커밋
    for user_id, total, completed, avg_success in quest_stats:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            user.total_quests = total
            user.completed_quests = completed
            user.average_success_rate = avg_success if avg_success is not None else 0.0
            user.preferred_category = preferred_categories.get(user_id)
    
    db.commit() # DB에 변경사항 영구 저장

    # 2. 업데이트된 User 테이블의 통계를 로드하여 DataFrame 생성
    users = db.query(User).all()
    user_stats = [{
        'user_id': u.id,
        'total_quests': u.total_quests,
        'completed_quests': u.completed_quests,
        'streak_days': u.streak_days,
        'preferred_category': u.preferred_category,
        'average_success_rate': u.average_success_rate,
        'user_success_rate': u.average_success_rate,
    } for u in users]
    user_df = pd.DataFrame(user_stats)
    
    # 2. QuestHistory를 사용하여 최근 활동 피처 보강 (기존 로직 유지)
    latest_completion = db.query(
        QuestHistory.user_id,
        func.max(QuestHistory.timestamp).label('last_completed_at')
    ).filter(
        QuestHistory.action == "completed"
    ).group_by(
        QuestHistory.user_id
    ).all()
    
    history_df = pd.DataFrame(latest_completion, columns=['user_id', 'last_completed_at'])
    
    # 3. 데이터프레임 병합
    final_df = pd.merge(user_df, history_df, on='user_id', how='left')

    return final_df

def train_model():
    print("--- 1. 데이터 로드 및 임베딩 시작 ---")
    df = load_data()
    if df.empty:
        raise ValueError("데이터셋이 비어 있습니다. seed.py를 먼저 실행하세요.")
    
    db = SessionLocal()
    user_stats_df = get_user_statistics_df(db)
    db.close()

    df = pd.merge(df, user_stats_df, on='user_id', how='left')

    required = ["difficulty", "completed"]
    if "name" not in df.columns:
        if "quest_name" in df.columns:
            df.rename(columns={"quest_name": "name"}, inplace=True)
        else:
            df["name"] = "Unknown"

    if "days" not in df.columns:
        if "duration" in df.columns:
            df.rename(columns={"duration": "days"}, inplace=True)
    
    if 'motivation' not in df.columns:
        df['motivation'] = ''

    df['total_quests'] = df['total_quests'].fillna(0)
    df['completed_quests'] = df['completed_quests'].fillna(0)
    df['streak_days'] = df['streak_days'].fillna(0)

    mean_rate = df['average_success_rate'].mean()
    df['average_success_rate'] = df['average_success_rate'].fillna(mean_rate)
    df['user_success_rate'] = df['user_success_rate'].fillna(mean_rate)
    df['preferred_category'] = df['preferred_category'].fillna('none')

    embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    print("임베딩 생성 중 ...")

    text_features = df["name"].astype(str) + " " + df["motivation"].astype(str)

    embeddings = embedder.encode(text_features.tolist(), show_progress_bar=True)
    emb_df = pd.DataFrame(embeddings, columns=[f"emb_{i}" for i in range(embeddings.shape[1])])
    df = pd.concat([df.reset_index(drop=True), emb_df], axis=1)
    df = df.drop(columns=["name", "motivation"], errors="ignore")

    # 퀘스트의 success_rate (seed에서 예측값) 결측치 채우기
    if "success_rate" not in df.columns:
        # 컬럼이 없으면 user_success_rate(사용자 평균) 사용
        df["success_rate"] = df["user_success_rate"] 
    else:
        df["success_rate"] = df["success_rate"].fillna(df["success_rate"].mean())
    
    cols_to_dummy = [c for c in ["category", "preferred_category"] if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_dummy, drop_first=True, prefix_sep='_')

    cols_to_drop = ["completed", "last_completed_at", "user_id"]

    X = df.drop(columns=cols_to_drop, errors='ignore')
    y = df["completed"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    num_cols = [
        "days", "difficulty", "success_rate", "user_success_rate", 
        "total_quests", "completed_quests", "streak_days", 
        "average_success_rate"
    ]

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
        class_weight={0: 1.0, 1: 3.0},
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
    try:
        if isinstance(embedder, SentenceTransformer):
            embedder.to(torch.device('cpu')) # torch 임포트 필요 (train.py에 torch 임포트가 없으면 추가)
    except NameError:
        # torch가 임포트되지 않은 경우 무시하거나, train.py 최상단에 import torch 추가
        pass
    except Exception as e:
        print(f"경고: 임베더를 CPU로 이동 중 오류 발생: {e}")

    joblib.dump((model, embedder), MODEL_PATH)
    print(f"✅ 모델 저장 완료: {MODEL_PATH}")

if __name__ == "__main__":
    init_db()
    train_model()

# python -m src.train