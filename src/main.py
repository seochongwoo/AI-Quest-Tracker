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
                
            </style>
        </head>
        <body>
            <h1>AI Quest Tracker</h1>
            <div class="section">
                <h2>📊 데이터 시각화</h2>
                <p><a href="/plot/user"><button>사용자별 완료 퀘스트 그래프</button></a></p>
                <p><a href="/plot/quest"><button>퀘스트별 완료율 그래프</button></a></p>
            </div>
            <div class="section">
                <h2>💻 API 및 UI</h2>
                <p>API 테스트는 <a href="/docs">Swagger UI (/docs)</a>를 이용하세요.</p>
                <p>실시간 데이터 확인 및 퀘스트 입력은 <a href="/quests/list"><button>실시간 퀘스트 목록 (UI)</button></a>에서.</p>
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

# 퀘스트 목록 UI 엔드포인트
@app.get("/quests/list", response_class=HTMLResponse)
def list_quests_ui(db: Session = Depends(get_db)):
    """DB에 저장된 퀘스트 목록과 AI 예측 결과를 HTML 테이블로 보여줍니다. (UI 포함)"""
    
    # 1. DB에서 퀘스트 목록 조회
    quests = crud.get_quests(db, limit=50) 
    
    # 2. HTML 테이블 내용 생성
    table_rows = ""
    for q in quests:
        # success_rate 속성이 없을 경우를 대비하여 0.0을 사용
        rate = getattr(q, 'success_rate', 0.0) 
        rate_percent = f"{rate * 100:.1f}%"
        
        # 완료 여부에 따라 색상을 다르게 표시
        status_color = 'green' if q.completed else 'red'
        
        # NOTE: table_rows는 f-string으로 깔끔하게 처리합니다.
        table_rows += f"""
        <tr>
            <td>{q.id}</td>
            <td>{q.user_id}</td>
            <td>{q.name}</td>
            <td>{q.duration}일</td>
            <td>{q.difficulty if q.difficulty is not None else '-'}</td>
            <td style="color: {status_color};">{'✅' if q.completed else '❌'}</td>
            <td>{rate_percent}</td>
        </tr>
        """
    
    # 3. 전체 HTML 구조 (입력 폼 및 JavaScript 포함)
    # 전체를 f-string으로 정의하며, HTML 내부의 중괄호는 전부 {{ }}로 이스케이프합니다.
    html_content = f"""
    <html>
        <head>
            <title>퀘스트 목록 및 AI 예측</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                table {{ border-collapse: collapse; width: 80%; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                h2 {{ color: #333; }}
                label {{ display: inline-block; margin-top: 10px; font-weight: bold; }}
                input[type="text"], input[type="number"] {{ padding: 8px; margin: 5px 10px 10px 0; border: 1px solid #ccc; border-radius: 4px; }}
                button {{ cursor: pointer; }}
            </style>
        </head>
        <body>
            <h2>퀘스트 목록 및 AI 예측 결과 (최신순)</h2>
            <a href="/"><button style="padding: 8px 15px; cursor: pointer;">메인으로 돌아가기</button></a>
            
            <hr style="margin: 20px 0;">
            
            <form id="quest-form" action="/quests/" method="post" style="padding: 15px; border: 1px solid #007bff; border-radius: 5px; background-color: #e6f7ff;">
                <h3>✨ 새로운 퀘스트 추가</h3>
                
                <label for="user_id">User ID (필수):</label>
                <input type="number" id="user_id" name="user_id" value="1" required min="1" style="width: 80px;">
                
                <label for="name">퀘스트 이름 (필수):</label>
                <input type="text" id="name" name="name" required style="width: 200px;">
                
                <label for="duration">소요 일수 (기간):</label>
                <input type="number" id="duration" name="duration" min="1" max="365" style="width: 80px;">
                
                <label for="difficulty">난이도 (1-5):</label>
                <input type="number" id="difficulty" name="difficulty" min="1" max="5" style="width: 80px;">
                
                <br>
                <button type="submit" style="margin-top: 10px; padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 5px;">
                    퀘스트 등록 및 AI 예측 받기
                </button>
                <p style="color: #0056b3; font-size: small; margin-top: 10px;">등록 후 페이지가 새로고침되며 AI 예측 결과가 목록에 추가됩니다.</p>
            </form>
            <hr style="margin: 20px 0;">
            
            <table>
                <tr>
                    <th>ID</th>
                    <th>User ID</th>
                    <th>퀘스트 이름</th>
                    <th>소요 일수</th>
                    <th>난이도</th>
                    <th>완료 여부</th>
                    <th>AI 성공률</th>
                </tr>
                {table_rows}
            </table>

            <script>
                document.getElementById('quest-form').addEventListener('submit', async function(e) {{
                    e.preventDefault(); // 기본 폼 제출 방지

                    const form = this;
                    const formData = new FormData(form);
                    const data = {{}};
                    
                    // 폼 데이터를 JSON 객체로 변환
                    formData.forEach((value, key) => {{
                        // user_id, duration, difficulty는 정수로 변환 시도
                        if (key === 'user_id' || key === 'duration' || key === 'difficulty') {{
                            const numValue = parseInt(value);
                            data[key] = isNaN(numValue) ? null : numValue; // 숫자가 아니면 (빈 칸) null 처리
                        }} else {{
                            data[key] = value;
                        }}
                    }});

                    // duration, difficulty가 null이면 제거 (스키마 Optional[int]에 맞춤)
                    if (data.duration === null) delete data.duration;
                    if (data.difficulty === null) delete data.difficulty;

                    try {{
                        const response = await fetch(form.action, {{
                            method: form.method,
                            headers: {{
                                'Content-Type': 'application/json'
                            }},
                            body: JSON.stringify(data) // JSON 문자열로 전송
                        }});

                        if (response.ok) {{
                            alert("퀘스트 등록 성공! 목록을 새로고침합니다.");
                            window.location.reload(); // 성공 시 페이지 새로고침
                        }} else {{
                            const error = await response.json();
                            alert(`퀘스트 등록 실패: ${{error.detail || response.statusText}}`);
                        }}
                    }} catch (error) {{
                        alert('요청 중 오류 발생: ' + error.message);
                    }}
                }});
            </script>
        </body>
    </html>
    """
    
    # 이제 이스케이프된 HTML을 바로 반환합니다. 
    # {{ }}로 이스케이프되어 f-string이 내부 중괄호를 무시하고 table_rows만 주입합니다.
    return HTMLResponse(content=html_content)


# uvicorn src.main:app --reload
