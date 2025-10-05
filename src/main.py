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

from fastapi.responses import HTMLResponse

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
    <head>
        <title>AI Quest Tracker</title>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                background-color: #f9fafc;
                margin: 0;
                padding: 0;
                text-align: center;
                color: #222;
            }
            header {
                background: linear-gradient(120deg, #02071e, #030928);
                color: white;
                padding: 40px 0;
                box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            }
            h1 { font-size: 2.2em; margin: 0; }
            p.desc { font-size: 1.1em; color: #ddd; margin-top: 10px; }

            .container {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 20px;
                margin: 40px auto;
                max-width: 900px;
            }

            .card {
                background: white;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                width: 260px;
                padding: 25px;
                transition: transform 0.2s ease;
            }
            .card:hover {
                transform: translateY(-5px);
            }
            .card h2 {
                margin-bottom: 10px;
                color: #02071e;
            }
            .card p {
                color: #555;
                font-size: 0.95em;
                margin-bottom: 15px;
            }
            .card a {
                display: inline-block;
                text-decoration: none;
                background-color: #030928;
                color: white;
                padding: 10px 18px;
                border-radius: 6px;
                transition: background-color 0.2s;
            }
            .card a:hover {
                background-color: #02071e;
            }
            footer {
                margin-top: 50px;
                font-size: 0.9em;
                color: #888;
            }
            footer a {
                color: #007bff;
                text-decoration: none;
            }
            footer a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <header>
            <h1>🚀 AI Quest Tracker</h1>
            <p class="desc">습관을 쌓고, AI로 성장하세요</p>
        </header>

        <div class="container">
            <div class="card">
                <h2>🧭 퀘스트 관리</h2>
                <p>퀘스트를 추가하고, 완료 여부를 관리하세요.</p>
                <a href="/quests/list">바로가기</a>
            </div>

            <div class="card">
                <h2>📊 데이터 시각화</h2>
                <p>사용자별, 퀘스트별 완료 현황을 한눈에 확인해요.</p>
                <a href="/plot/user">시각화 보기</a>
            </div>

            <div class="card">
                <h2>🤖 AI 퀘스트 추천</h2>
                <p>AI가 당신의 패턴을 학습하고 맞춤 퀘스트를 제안합니다.</p>
                <a href="/recommend">추천받기</a>
            </div>
        </div>

        <footer>
            <p>🔗 <a href="/docs">Swagger API 문서 보기</a></p>
        </footer>
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

# 퀘스트 목록 UI 엔드포인트
@app.get("/quests/list", response_class=HTMLResponse)
def list_quests_ui(db: Session = Depends(get_db)):
    """DB에 저장된 퀘스트 목록 + CRUD UI"""
    quests = crud.get_quests(db, limit=50)

    table_rows = ""
    for q in quests:
        rate = getattr(q, 'success_rate', 0.0)
        rate_percent = f"{rate * 100:.1f}%"
        status_color = 'green' if q.completed else 'red'
        toggle_label = "✅ 완료" if not q.completed else "↩️ 취소"

        table_rows += f"""
        <tr>
            <td>{q.id}</td>
            <td>{q.user_id}</td>
            <td>{q.name}</td>
            <td>{q.duration or '-'}</td>
            <td>{q.difficulty or '-'}</td>
            <td style='color:{status_color}'>{'완료' if q.completed else '미완료'}</td>
            <td>{rate_percent}</td>
            <td>
                <button onclick="toggleComplete({q.id})">{toggle_label}</button>
                <button onclick="deleteQuest({q.id})" style="color:red;">🗑️ 삭제</button>
            </td>
        </tr>
        """

    html = f"""
    <html>
    <head>
        <title>Quest Dashboard</title>
        <style>
            body {{ font-family: Arial; margin: 20px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: center; }}
            th {{ background-color: #f0f0f0; }}
            button {{ padding: 5px 10px; border: none; border-radius: 5px; cursor: pointer; }}
            button:hover {{ opacity: 0.8; }}
        </style>
    </head>
    <body>
        <h2>🧭 퀘스트 관리 대시보드</h2>
        <a href="/"><button>메인으로</button></a>
        <form id="add-form" style="margin-top:20px;">
            <h3>✨ 새 퀘스트 추가</h3>
            <input type="number" name="user_id" placeholder="User ID" required min="1">
            <input type="text" name="name" placeholder="퀘스트 이름" required>
            <input type="number" name="duration" placeholder="소요 일수" min="1">
            <input type="number" name="difficulty" placeholder="난이도 (1-5)" min="1" max="5">
            <button type="submit" style="background-color:#007bff;color:white;">추가</button>
        </form>

        <table>
            <tr>
                <th>ID</th><th>User</th><th>퀘스트</th><th>기간</th><th>난이도</th><th>상태</th><th>AI 성공률</th><th>조작</th>
            </tr>
            {table_rows}
        </table>

        <script>
        async function toggleComplete(id) {{
            const res = await fetch(`/quests/${{id}}/toggle`, {{ method: "PATCH" }});
            if (res.ok) location.reload();
            else alert("변경 실패");
        }}

        async function deleteQuest(id) {{
            if (!confirm("정말 삭제할까요?")) return;
            const res = await fetch(`/quests/${{id}}`, {{ method: "DELETE" }});
            if (res.ok) location.reload();
            else alert("삭제 실패");
        }}

        document.getElementById("add-form").addEventListener("submit", async (e) => {{
            e.preventDefault();
            const data = Object.fromEntries(new FormData(e.target).entries());
            data.user_id = parseInt(data.user_id);
            data.duration = data.duration ? parseInt(data.duration) : null;
            data.difficulty = data.difficulty ? parseInt(data.difficulty) : null;

            const res = await fetch("/quests/", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify(data)
            }});

            if (res.ok) location.reload();
            else alert("추가 실패");
        }});
        </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

# 퀘스트 완료 토글 (PATCH)
@app.patch("/quests/{quest_id}/toggle")
def toggle_quest(quest_id: int, db: Session = Depends(get_db)):
    quest = crud.get_quest(db, quest_id)
    if not quest:
        raise HTTPException(status_code=404, detail="Quest not found")
    quest.completed = not quest.completed
    db.commit()
    db.refresh(quest)
    return {"id": quest.id, "completed": quest.completed}

# 퀘스트 삭제 (DELETE)
@app.delete("/quests/{quest_id}")
def delete_quest(quest_id: int, db: Session = Depends(get_db)):
    quest = crud.get_quest(db, quest_id)
    if not quest:
        raise HTTPException(status_code=404, detail="Quest not found")
    db.delete(quest)
    db.commit()
    return {"detail": "Deleted"}



# uvicorn src.main:app --reload
