from fastapi import FastAPI
import pandas as pd
from pathlib import Path

# 데이터 경로
data_path = Path(__file__).parent.parent / "data" / "sample_quest.csv"
data = pd.read_csv(data_path)

# 앱 생성
app = FastAPI(title="AI Quest Tracker API")

# 기본 루트
@app.get("/")
def root():
    return {"message": "AI Quest Tracker API 준비 완료!"}

# 사용자별 완료 퀘스트 수
@app.get("/user_completed")
def user_completed():
    result = data.groupby("user_id")["completed"].sum().to_dict()
    return result

# 퀘스트별 완료율 
@app.get("/quest_completion_rate")
def quest_completion_rate():
    result = (data.groupby("quest")["completed"].mean() * 100).round(2).to_dict()
    return result

# uvicorn src.main:app --reload