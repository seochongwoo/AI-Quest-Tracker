'''
FastAPI 인스턴스를 생성하고, /, /plot/user, /users/ 등 모든 API 엔드포인트를 정의
get_db() 함수를 통해 DB 세션을 각 요청에 주입하고, /users/ 라우트에서는 crud.py 함수를 호출하여 DB 작업을 수행
'''
# fast api 백엔드를 위한 import
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from src import crud, schemas, database
from . import crud, schemas, model
# Db를 위한 import
from .database import SessionLocal, init_db, Quest
from . import crud, schemas
from .utils import plot_user_completed, plot_quest_completion_rate
from joblib import load 
#  AI 예측 및 시간 관리를 위한 임포트 추가
from sklearn.preprocessing import OneHotEncoder 
import pandas as pd 
from datetime import datetime

app = FastAPI(title="AI Quest Tracker API")
MODEL_PATH = "model/model.pkl"

# 앱  생성 직후 호출하여 서버 시작 전에 테이블 생성 (버그 방지)
init_db() 

# 모델을 전역적으로 로드(서버 시작시 한번만)
model.load_ml_model()

# DB 연결 의존성
def get_db():
    db = database.SessionLocal()
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
                <h2>💡 AI 퀘스트 추천</h2>
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


## 시각화 관련 라우트 (habit_analyis), 데이터 시각화 페이지

# 예시 1
@app.get("/plot/user", response_class=HTMLResponse)
def user_plot():
    img_base64 = plot_user_completed()
    return f'<html><body><h2>사용자별 완료 퀘스트</h2><img src="data:image/png;base64,{img_base64}"/></body></html>'

# 예시 2
@app.get("/plot/quest", response_class=HTMLResponse)
def quest_plot():
    img_base64 = plot_quest_completion_rate()
    return f'<html><body><h2>퀘스트별 완료율</h2><img src="data:image/png;base64,{img_base64}"/></body></html>'

## DB 관련 라우트 (CRUD), 퀘스트 관리 페이지

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
def create_quest(quest: schemas.QuestCreate, db: Session = Depends(get_db)):
    """
    새로운 퀘스트 추가 (AI 성공률 자동 계산)
    """
    try:
        predicted_rate = model.predict_success_rate(
            quest.user_id,
            quest.name,
            quest.duration or 1,
            quest.difficulty or 3
        )

        # DB에 저장
        db_quest = crud.create_quest(
            db=db,
            quest_data={
                "user_id": quest.user_id,
                "name": quest.name,
                "duration": quest.duration,
                "difficulty": quest.difficulty,
                "success_rate": predicted_rate,
            }
        )
        return db_quest

    except Exception as e:
        print(f"[ERROR] 퀘스트 생성 실패: {e}")
        raise HTTPException(status_code=400, detail="퀘스트 생성 중 오류가 발생했습니다.")


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
    """사용자 퀘스트 관리 대시보드 (더미 데이터 제외 + UI 개선)"""
    quests = [q for q in crud.get_quests(db, limit=100) if q.user_id > 5]  # 더미 제외

    table_rows = ""
    for q in quests:
        rate = getattr(q, 'success_rate', 0.0)
        rate_percent = f"{rate * 100:.1f}%"
        # 성공률 색상 그라데이션
        color = (
            "red" if rate < 0.4 else
            "orange" if rate < 0.7 else
            "green"
        )
        status_color = 'green' if q.completed else 'gray'
        toggle_label = "✅ 완료" if not q.completed else "↩️ 취소"

        table_rows += f"""
        <tr>
            <td>{q.id}</td>
            <td>{q.name}</td>
            <td>{q.duration or '-'}</td>
            <td>{q.difficulty or '-'}</td>
            <td style='color:{status_color}'>{'완료' if q.completed else '미완료'}</td>
            <td style='color:{color};font-weight:bold;'>{rate_percent}</td>
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
            body {{
                font-family: 'Segoe UI', sans-serif;
                background-color: #f8f9fc;
                margin: 0;
                padding: 20px;
            }}
            header {{
                background: linear-gradient(120deg, #02071e, #030928);
                color: white;
                padding: 15px 25px;
                position: sticky;
                top: 0;
                z-index: 100;
                display: flex;
                justify-content: space-between;
                align-items: center;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
            }}
            header h2 {{ margin: 0; }}
            header button {{
                background-color: #007bff;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 6px;
                cursor: pointer;
            }}
            header button:hover {{ background-color: #0056b3; }}
            .form-card {{
                background: white;
                padding: 20px;
                margin-top: 20px;
                border-radius: 10px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                max-width: 600px;
                margin: 20px auto;
            }}
            .form-card h3 {{ margin-top: 0; }}
            input {{
                padding: 8px;
                margin: 5px;
                border-radius: 6px;
                border: 1px solid #ccc;
                width: 120px;
            }}
            button {{
                padding: 6px 12px;
                border-radius: 5px;
                cursor: pointer;
                border: none;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 30px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 10px;
                text-align: center;
            }}
            th {{
                background-color: #f1f3f8;
            }}
            tr:nth-child(even) {{
                background-color: #fafbff;
            }}
            tr:hover {{
                background-color: #eef2ff;
            }}
        </style>
    </head>
    <body>
        <header>
            <h2>🧭 퀘스트 관리 대시보드</h2>
            <a href="/"><button>메인으로</button></a>
        </header>

        <div class="form-card">
            <h3>✨ 새 퀘스트 추가</h3>
            <form id="add-form">
                <input type="number" name="user_id" placeholder="User ID" required min="6">
                <input type="text" name="name" placeholder="퀘스트 이름" required>
                <input type="number" name="duration" placeholder="소요 일수" min="1">
                <input type="number" name="difficulty" placeholder="난이도 (1-5)" min="1" max="5">
                <button type="submit" style="background-color:#28a745;color:white;">추가</button>
            </form>
        </div>

        <table>
            <tr>
                <th>ID</th><th>퀘스트</th><th>기간</th><th>난이도</th>
                <th>상태</th><th>AI 성공률</th><th>조작</th>
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

## AI 퀘스트 추천 페이지
@app.get("/recommend", response_class=HTMLResponse)
def recommend_page():
    return """
    <html>
        <head>
            <title>AI 퀘스트 추천</title>
            <style>
                body { font-family: 'Segoe UI', sans-serif; text-align:center; margin-top:40px; background-color:#f8f9fa; color:#222; }
                form { margin: 20px auto; padding: 20px; width: 400px; background: white; border-radius: 12px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
                input, select { width: 90%; padding: 10px; margin: 8px 0; border-radius: 8px; border: 1px solid #ccc; }
                button { padding: 10px 15px; background-color: #0078d4; color: white; border: none; border-radius: 8px; cursor: pointer; }
                button:hover { background-color: #005fa3; }
                .gauge-container { width: 400px; margin: 30px auto; text-align:center; }
                .gauge-bar { height: 25px; border-radius: 10px; background-color: #e9ecef; overflow:hidden; }
                .gauge-fill { height: 100%; background-color: #28a745; text-align:right; color:white; font-weight:bold; padding-right:8px; border-radius: 10px; }
            </style>
        </head>
        <body>
            <h1>💡 AI 퀘스트 추천</h1>
            <p>아래 정보를 입력하면 AI가 성공 확률과 추천 난이도를 예측합니다.</p>

            <form action="/recommend/result" method="post">
                <input type="text" name="quest_name" placeholder="퀘스트 이름" required><br>
                <input type="number" name="duration" placeholder="예상 기간 (일)" required><br>
                <select name="difficulty">
                    <option value="1">난이도 1 (매우 쉬움)</option>
                    <option value="2">난이도 2</option>
                    <option value="3" selected>난이도 3</option>
                    <option value="4">난이도 4</option>
                    <option value="5">난이도 5 (매우 어려움)</option>
                </select><br>
                <button type="submit">AI 예측 실행 🚀</button>
            </form>
        </body>
    </html>
    """

@app.post("/recommend/result", response_class=HTMLResponse)
async def recommend_result(request: Request):
    form = await request.form()
    quest_name = form.get("quest_name")
    duration = int(form.get("duration"))
    difficulty = int(form.get("difficulty"))
    
    # 현재 로그인 기능이 없으므로 user_id=1로 가정
    success_rate = model.predict_success_rate(1, quest_name, duration, difficulty)
    percent = round(success_rate * 100, 1)
    
    # 성공 확률에 따른 메시지
    if percent >= 80:
        message = "🔥 도전해볼 만한 목표예요!"
    elif percent >= 60:
        message = "💪 충분히 가능성이 있습니다!"
    elif percent >= 40:
        message = "⚖️ 조금 어렵지만 해볼 수 있어요."
    else:
        message = "💀 난이도가 높습니다. 단계를 낮춰보세요."
    
    # 성공 확률 게이지 색상 변경
    if percent >= 70:
        color = "#28a745"
    elif percent >= 50:
        color = "#ffc107"
    else:
        color = "#dc3545"
    
    return f"""
    <html>
        <head>
            <title>AI 추천 결과</title>
            <style>
                body {{ font-family:'Segoe UI', sans-serif; text-align:center; background-color:#f8f9fa; margin-top:60px; }}
                .result-box {{ background:white; width:400px; margin:0 auto; border-radius:12px; padding:20px; box-shadow:0 4px 10px rgba(0,0,0,0.1); }}
                .gauge-bar {{ height:25px; border-radius:10px; background-color:#e9ecef; overflow:hidden; margin-top:15px; }}
                .gauge-fill {{ height:100%; background-color:{color}; width:{percent}%; text-align:right; color:white; font-weight:bold; padding-right:8px; border-radius:10px; transition:width 0.6s ease-in-out; }}
                a {{ text-decoration:none; color:#0078d4; font-weight:bold; }}
            </style>
        </head>
        <body>
            <div class="result-box">
                <h2>🧠 AI 예측 결과</h2>
                <p><b>{quest_name}</b> 퀘스트의 성공 확률은</p>
                <div class="gauge-bar">
                    <div class="gauge-fill">{percent}%</div>
                </div>
                <h3>{message}</h3>
                <br>
                <a href="/recommend">🔁 다시 예측하기</a> | <a href="/">🏠 홈으로</a>
            </div>
        </body>
    </html>
    """
# uvicorn src.main:app --reload
