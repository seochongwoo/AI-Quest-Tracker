'''
FastAPI 인스턴스를 생성하고, /, /plot/user, /users/ 등 모든 API 엔드포인트를 정의
get_db() 함수를 통해 DB 세션을 각 요청에 주입하고, /users/ 라우트에서는 crud.py 함수를 호출하여 DB 작업을 수행
'''
# fast api 백엔드를 위한 import
from fastapi import FastAPI, Depends, HTTPException 
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
# Db를 위한 import
from .database import SessionLocal, init_db
from . import crud, schemas
from .utils import plot_user_completed, plot_quest_completion_rate

app = FastAPI(title="AI Quest Tracker API")

# 앱 인스턴스 생성 직후 호출하여 서버 시작 전에 테이블이 만들어지게 합니다.
init_db() 

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
        <head>
            <title>AI Quest Tracker Demo</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin-bottom: 25px; padding: 15px; border: 1px solid #ccc; border-radius: 5px; }
                h1 { color: #333; }
                button, a { padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 5px; text-decoration: none; cursor: pointer; }
                button:hover, a:hover { background-color: #0056b3; }
                .api-link { background-color: #28a745; margin-right: 10px; }
            </style>
        </head>
        <body>
            <h1>🌟 AI Quest Tracker API 준비 완료!</h1>

            <div class="section">
                <h2>📊 데이터 시각화 (CSV 기반)</h2>
                <p>
                    <a href="/plot/user" target="_blank" class="api-link">사용자별 완료 퀘스트 그래프 보기</a>
                    <a href="/plot/quest" target="_blank" class="api-link">퀘스트별 완료율 그래프 보기</a>
                </p>
            </div>
            
            <div class="section">
                <h2>💻 API 테스트 및 데이터 입력</h2>
                <p>
                    <a href="/docs" target="_blank">전체 API 문서 (Swagger UI)로 이동</a>
                </p>
                <p style="font-size: 0.9em; color: #555;">
                    DB 조작 및 AI 기능 테스트는 /docs에서 가능합니다. (사용자, 퀘스트 생성/조회)
                </p>
            </div>
            
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

# DB 관련 라우트 (CRUD) 

# 1. 사용자 생성 
@app.post("/users/", response_model=schemas.User)
def create_user_endpoint(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Pydantic 모델을 인수로 받아 crud 함수로 전달
    return crud.create_user(db=db, user=user)

# 2. 사용자 목록 조회 
@app.get("/users/", response_model=list[schemas.User])
def get_users_endpoint(db: Session = Depends(get_db)):
    return crud.get_users(db=db, skip=0, limit=100) # limit 추가

# 3. 퀘스트 생성 추가
@app.post("/quests/", response_model=schemas.Quest)
def create_quest_for_user(quest: schemas.QuestCreate, db: Session = Depends(get_db)):
    # user_id가 존재하는지 확인하는 로직 추가 (필수)
    if crud.get_user(db, quest.user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
        
    return crud.create_user_quest(db=db, quest=quest)

# 4. 특정 사용자 퀘스트 조회 추가
@app.get("/users/{user_id}/quests/", response_model=list[schemas.Quest])
def get_user_quests(user_id: int, db: Session = Depends(get_db)):
    quests = crud.get_quests(db=db, user_id=user_id)
    if not quests and crud.get_user(db, user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    return quests
# uvicorn src.main:app --reload