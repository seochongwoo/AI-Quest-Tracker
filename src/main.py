'''
FastAPI 인스턴스를 생성하고, /, /plot/user, /users/ 등 모든 API 엔드포인트를 정의
get_db() 함수를 통해 DB 세션을 각 요청에 주입하고, /users/ 라우트에서는 crud.py 함수를 호출하여 DB 작업을 수행
'''
# fast api 백엔드를 위한 import
from fastapi import FastAPI, Depends
from fastapi.responses import HTMLResponse
from .utils import plot_user_completed, plot_quest_completion_rate
# Db를 위한 import
from sqlalchemy.orm import Session
from src.database import SessionLocal, User
from .database import SessionLocal, init_db
from . import crud, schemas

app = FastAPI(title="AI Quest Tracker API")

# DB 연결 의존성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head><title>AI Quest Tracker</title></head>
        <body>
            <h1>AI Quest Tracker API 준비 완료!</h1>
            <p><a href="/plot/user"><button>사용자별 완료 퀘스트 그래프</button></a></p>
            <p><a href="/plot/quest"><button>퀘스트별 완료율 그래프</button></a></p>
        </body>
    </html>
    """

@app.get("/plot/user", response_class=HTMLResponse)
def user_plot():
    img_base64 = plot_user_completed()
    return f'<html><body><h2>사용자별 완료 퀘스트</h2><img src="data:image/png;base64,{img_base64}"/></body></html>'

@app.get("/plot/quest", response_class=HTMLResponse)
def quest_plot():
    img_base64 = plot_quest_completion_rate()
    return f'<html><body><h2>퀘스트별 완료율</h2><img src="data:image/png;base64,{img_base64}"/></body></html>'

# DB 관련 라우트 추가
# 1. 사용자 생성 (Pydantic 스키마 적용)
@app.post("/users/", response_model=schemas.User)
def create_user_endpoint(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Pydantic 모델을 인수로 받아 crud 함수로 전달
    return crud.create_user(db=db, user=user)

# 2. 사용자 목록 조회 (Pydantic 스키마 적용)
@app.get("/users/", response_model=list[schemas.User])
def get_users_endpoint(db: Session = Depends(get_db)):
    return crud.get_users(db=db)

# 3. 퀘스트 생성 추가
@app.post("/quests/", response_model=schemas.Quest)
def create_quest_for_user(quest: schemas.QuestCreate, db: Session = Depends(get_db)):
    # user_id가 schemas.QuestCreate에 포함되어 있으므로 바로 전달
    # 실제로는 user_id가 존재하는지 확인하는 로직이 필요합니다.
    return crud.create_user_quest(db=db, quest=quest)

# 4. 특정 사용자 퀘스트 조회 추가
@app.get("/users/{user_id}/quests/", response_model=list[schemas.Quest])
def get_user_quests(user_id: int, db: Session = Depends(get_db)):
    return crud.get_quests(db=db, user_id=user_id)


# uvicorn src.main:app --reload