'''
FastAPI 인스턴스를 생성하고, /, /plot/user, /users/ 등 모든 API 엔드포인트를 정의
get_db() 함수를 통해 DB 세션을 각 요청에 주입하고, /users/ 라우트에서는 crud.py 함수를 호출하여 DB 작업을 수행
'''
# fast api 백엔드를 위한 import
from fastapi import FastAPI, Depends, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from src import crud, schemas, database
from . import crud, schemas, model
# Db를 위한 import
from .database import SessionLocal, init_db, Quest
from . import crud, schemas
#  AI 예측 및 시간 관리를 위한 임포트 추가
from sklearn.preprocessing import OneHotEncoder 
import pandas as pd 
from datetime import datetime
# 시각화를 위한 import
from .habit_analysis import plot_user_progress,plot_success_rate_by_category, plot_focus_area, plot_growth_trend

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

# 유저 ID를 가져오는 의존성 함수
def get_current_user_id(request: Request):
    user_id = request.cookies.get("user_id")
    if user_id:
        return int(user_id)
    # 로그인 안 되어 있으면 None 반환
    return None

# -----로그인 관련-----
# 로그인 페이지
@app.get("/login", response_class=HTMLResponse)
def login_page():
    return """
    <html>
        <head>
            <title>로그인</title>
            <style>
                body { font-family:'Segoe UI', sans-serif; text-align:center; background-color:#f0f2f5; margin-top:80px; }
                .login-box { background:white; padding:30px; width:380px; margin:auto; border-radius:12px; box-shadow:0 3px 10px rgba(0,0,0,0.1); }
                input { width:85%; padding:10px; margin-top:10px; border-radius:8px; border:1px solid #ccc; }
                button { margin-top:15px; padding:10px 20px; background-color:#030928; color:white; border:none; border-radius:8px; cursor:pointer; }
                button:hover { background-color:#02071e; }
            </style>
        </head>
        <body>
            <div class="login-box">
                <h2>AI Quest Tracker</h2>
                <form action="/login" method="post">
                    <input type="text" name="nickname" placeholder="닉네임을 입력하세요" required><br>
                    <button type="submit">시작하기 </button>
                </form>
            </div>
        </body>
    </html>
    """

# 로그인/회원가입 처리
@app.post("/login")
async def login_user(request: Request, db: Session = Depends(get_db)):
    form = await request.form()
    nickname = form.get("nickname")

    # 닉네임으로 사용자 검색
    user = crud.get_user_by_name(db, name=nickname)
    if not user:
        # 없으면 새로 생성 (schemas.UserCreate의 default값 사용)
        new_user = schemas.UserCreate(name=nickname)
        user = crud.create_user(db=db, user=new_user)
        # 새로 생성된 경우, 온보딩 페이지로 리디렉션
        redirect_url = "/onboarding"
    else:
        # 기존 사용자인 경우, 온보딩을 이미 완료했는지 확인
        if user.consistency_score == 3 and user.risk_aversion_score == 3:
            # 기본값(3)이면 온보딩 페이지로 리디렉션
            redirect_url = "/onboarding"
        else:
            redirect_url = "/"

    # 로그인 쿠키 설정 후 리디렉션
    response = RedirectResponse(url=redirect_url, status_code=303)
    response.set_cookie(key="user_id", value=str(user.id), httponly=True, max_age=86400)
    response.set_cookie(key="user_name", value=user.name, httponly=False, max_age=86400) # HTML에서 사용 가능하도록 httponly 해제
    return response

# 로그아웃
@app.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("user_id")
    response.delete_cookie("user_name")
    return response


# 온보딩 질문 페이지 (성향 점수 수집)
@app.get("/onboarding", response_class=HTMLResponse)
def onboarding_page(user_id: int = Depends(get_current_user_id)):
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)

    return f"""
    <html>
        <head>
            <title>초기 성향 분석</title>
            <style>
                body {{ font-family:'Segoe UI', sans-serif; text-align:center; background-color:#f0f2f5; margin-top:60px; }}
                .onboarding-box {{ background:white; padding:30px; width:450px; margin:auto; border-radius:12px; box-shadow:0 3px 10px rgba(0,0,0,0.1); text-align:left; }}
                h2 {{ text-align:center; color:#030928; }}
                label {{ display:block; margin-top:20px; font-weight:bold; color:#333; }}
                .radio-group {{ margin-top:10px; }}
                input[type="radio"] {{ margin-right:5px; }}
                button {{ width:100%; margin-top:30px; padding:12px; background-color:#030928; color:white; border:none; border-radius:8px; cursor:pointer; font-size:16px; }}
                button:hover {{ background-color:#02071e; }}
            </style>
        </head>
        <body>
            <div class="onboarding-box">
                <h2>🚀 AI 피드백을 위한 초기 성향 분석</h2>
                <p style="text-align:center; color:#666;">AI 예측의 정확도를 높이기 위해 두 가지 질문에 답변해주세요. (1: 전혀 아님, 5: 매우 그렇다)</p>
                <form action="/onboarding" method="post">
                    <input type="hidden" name="user_id" value="{user_id}">
                    
                    <label>1. 저는 한번 시작한 일은 꾸준히 해내는 편입니다. (일관성)</label>
                    <div class="radio-group">
                        {''.join([f'<input type="radio" name="consistency_score" value="{i}" required>{i}' for i in range(1, 6)])}
                    </div>

                    <label>2. 저는 기존 목표보다 약간 어려운 목표에 도전하는 것을 선호합니다. (도전 선호도)</label>
                    <div class="radio-group">
                        {''.join([f'<input type="radio" name="risk_aversion_score" value="{i}" required>{i}' for i in range(1, 6)])}
                    </div>
                    
                    <button type="submit">AI 피드백 시작하기</button>
                </form>
            </div>
        </body>
    </html>
    """

# 온보딩 질문 답변 처리 (성향 점수 DB 업데이트)
@app.post("/onboarding")
async def process_onboarding(
    user_id: int = Form(...),
    consistency_score: int = Form(...),
    risk_aversion_score: int = Form(...),
    db: Session = Depends(get_db)
):
    # Pydantic 스키마를 사용하여 유효성 검사 및 데이터 준비
    try:
        scores = schemas.UserUpdateScores(
            consistency_score=consistency_score,
            risk_aversion_score=risk_aversion_score
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"유효성 검사 오류: {e}")

    # crud 함수를 사용하여 DB 업데이트
    crud.update_user_scores(db, user_id, scores)
    
    # 메인 페이지로 리디렉션
    response = RedirectResponse(url="/", status_code=303)
    return response

# -----메인 페이지------
@app.get("/", response_class=HTMLResponse)
# 💡 FIX: db 의존성 주입 (FastAPI의 Depends 사용)
def root(request: Request, db: Session = Depends(get_db)): 
    # 1. 로그인 확인
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)
    
    user_id_int = int(user_id)
    # 💡 FIX: db 객체를 사용하여 crud 함수 호출
    user = crud.get_user(db, user_id_int) 

    if not user:
        # 쿠키는 있지만 DB에 없는 경우 (오류), 로그아웃 페이지로
        return RedirectResponse(url="/logout", status_code=303)

    # 2. 온보딩 완료 확인 (성향 점수가 기본값(3)인지 확인)
    if user.consistency_score == 3 and user.risk_aversion_score == 3:
        return RedirectResponse(url="/onboarding", status_code=303)

    # 3. 데이터 로드 (원래 사용자 코드 유지)
    user_name = user.name
    # 💡 FIX: db 객체를 사용하여 crud 함수 호출
    quests = crud.get_quests_by_user(db, user_id=user_id_int)

    # 로그인 및 온보딩 완료 시, 기존 메인 화면 렌더링 (원래 HTML 디자인 유지)
    return HTMLResponse(f"""
    <html>
    <head>
        <title>AI Quest Tracker</title>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                background-color: #f9fafc;
                margin: 0;
                padding: 0;
                text-align: center;
                color: #222;
            }}
            header {{
                background: linear-gradient(120deg, #02071e, #030928);
                color: white;
                padding: 40px 0;
                box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            }}
            h1 {{ font-size: 2.2em; margin: 0; }}
            /* 💡 사용자 이름 표시를 위한 desc 스타일 수정 */
            p.desc {{ font-size: 1.1em; color: #ddd; margin-top: 10px; }}

            .container {{
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 20px;
                margin: 40px auto;
                max-width: 900px;
            }}

            .card {{
                background: white;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                width: 260px;
                padding: 25px;
                transition: transform 0.2s ease;
            }}
            .card:hover {{
                transform: translateY(-5px);
            }}
            .card h2 {{
                margin-bottom: 10px;
                color: #02071e;
            }}
            .card p {{
                color: #555;
                font-size: 0.95em;
                margin-bottom: 15px;
            }}
            .card a {{
                display: inline-block;
                text-decoration: none;
                background-color: #030928;
                color: white;
                padding: 10px 18px;
                border-radius: 6px;
                transition: background-color 0.2s;
            }}
            .card a:hover {{
                background-color: #02071e;
            }}
            footer {{
                margin-top: 50px;
                font-size: 0.9em;
                color: #888;
            }}
            footer a {{
                color: #007bff;
                text-decoration: none;
            }}
            footer a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>🚀 AI Quest Tracker</h1>

            <p class="desc">{user_name}님 환영합니다! 습관을 쌓고, AI로 성장해보세요 | <a href="/logout" style="color: #ffcccc;">로그아웃</a></p>
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
                <a href="/plot/dashboard">시각화 보기</a>
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
    """)

# -----시각화 관련 라우트 (habit_analyis), 데이터 시각화 페이지-----

# 데이터 허브 페이지
@app.get("/plot/dashboard", response_class=HTMLResponse)
def plot_dashboard(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse("/login", status_code=302)

    return """
    <html>
    <head>
        <title>📊 데이터 시각화</title>
        <style>
            body {
                font-family: 'Segoe UI', sans-serif;
                background-color: #f9fafc;
                margin: 0;
                padding: 0;
                color: #222;
                text-align: center;
            }
            header {
                background: linear-gradient(120deg, #02071e, #030928);
                color: white;
                padding: 40px 0;
                box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            }
            h1 { margin: 0; font-size: 2em; }
            p.desc { color: #ddd; margin-top: 5px; }

            .container {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 25px;
                margin: 50px auto;
                max-width: 900px;
            }
            .card {
                background: white;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                width: 250px;
                padding: 25px;
                transition: transform 0.2s ease;
            }
            .card:hover { transform: translateY(-5px); }
            .card h2 { color: #02071e; margin-bottom: 10px; }
            .card p { color: #555; font-size: 0.95em; margin-bottom: 15px; }
            .card a {
                display: inline-block;
                text-decoration: none;
                background-color: #030928;
                color: white;
                padding: 10px 16px;
                border-radius: 6px;
                transition: background-color 0.2s;
            }
            .card a:hover { background-color: #02071e; }

            footer { margin-top: 40px; color: #888; font-size: 0.9em; }
            a.home { color: #007bff; text-decoration: none; }
            a.home:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <header>
            <h1>데이터 시각화 및 분석</h1>
            <p class="desc">나의 성취와 패턴을 다양한 시각화로 확인하세요</p>
        </header>

        <div class="container">
            <div class="card">
                <h2>개인 퀘스트 현황</h2>
                <p>완료율과 미완료율을 비율로 시각화합니다.</p>
                <a href="/plot/user">보기</a>
            </div>

            <div class="card">
                <h2>카테고리별 분석</h2>
                <p>AI 예측 성공률을 카테고리별로 비교합니다.</p>
                <a href="/plot/quest">보기</a>
            </div>

            <div class="card">
                <h2>성장 추세</h2>
                <p>시간이 지남에 따라 완료된 퀘스트 수를 확인하세요.</p>
                <a href="/plot/trend">보기</a>
            </div>

            <div class="card">
                <h2>집중 분야 분석</h2>
                <p>내가 가장 몰입하는 카테고리를 시각화합니다.</p>
                <a href="/plot/focus">보기</a>
            </div>
        </div>

        <footer>
            <a href="/plot/dashboard">📊 대시보드로 돌아가기</a> |
            <a class="home" href="/">🏠 홈으로 돌아가기</a>
        </footer>
    </body>
    </html>
    """

# 헬퍼 함수: 데이터 없음 메시지 HTML 생성
def _no_data_html(message: str = "데이터가 없습니다. 퀘스트를 먼저 추가하세요!") -> str:
    """데이터가 없을 때 표시할 중앙 정렬된 HTML 응답을 생성합니다."""
    return f"""
    <html>
        <body style="text-align:center; font-family:'Segoe UI'; padding-top: 50px;">
            <h3 style="color: #555;">{message}</h3>
            <br><a href="/plot/dashboard">대시보드로 돌아가기</a>
            <br><a href="/">🏠 홈으로</a>
        </body>
    </html>
    """

# 공통 스타일 템플릿
def _styled_plot_page(title: str, desc: str, emoji: str, img_base64: str) -> str:
    return f"""
    <html>
    <head>
        <title>{emoji} {title}</title>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                background-color: #f4f6f9;
                color: #222;
                margin: 0;
                padding: 0;
                text-align: center;
            }}
            header {{
                background: linear-gradient(120deg, #02071e, #030928);
                color: white;
                padding: 30px 0;
                margin-bottom: 30px;
                box-shadow: 0 3px 6px rgba(0,0,0,0.2);
            }}
            h1 {{ margin: 0; font-size: 1.8em; }}
            p.desc {{ color: #ccc; margin-top: 8px; font-size: 1em; }}
            .card {{
                background: white;
                width: 80%;
                max-width: 700px;
                margin: 0 auto;
                border-radius: 12px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                padding: 20px;
                text-align: center;
            }}
            img {{
                width: 90%;
                max-width: 650px;
                border-radius: 8px;
                margin-top: 15px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.15);
            }}
            footer {{
                margin-top: 30px;
                color: #777;
                font-size: 0.9em;
            }}
            a {{
                color: #007bff;
                text-decoration: none;
            }}
            a:hover {{ text-decoration: underline; }}
        </style>
    </head>
    <body>
        <header>
            <h1>{emoji} {title}</h1>
            <p class="desc">{desc}</p>
        </header>

        <div class="card">
            <img src="data:image/png;base64,{img_base64}" alt="{title}" />
        </div>

        <footer>
            <a href="/plot/dashboard">📊 대시보드로 돌아가기</a> |
            <a href="/">🏠 홈으로</a>
        </footer>
    </body>
    </html>
    """

# 개인 퀘스트 진행 현황 시각화
@app.get("/plot/user", response_class=HTMLResponse)
def plot_user(request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse("/login", status_code=302)

    img_base64 = plot_user_progress(db, int(user_id))
    if not img_base64:
        return HTMLResponse(_no_data_html("데이터가 없습니다. 퀘스트를 먼저 추가하세요!"))

    return _styled_plot_page(
        title="내 퀘스트 진행 현황",
        desc="완료율과 미완료율의 비율을 시각적으로 확인하세요.",
        emoji="📊",
        img_base64=img_base64
    )

# 카테고리별 성공률 시각화
@app.get("/plot/quest", response_class=HTMLResponse)
def plot_quest(request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse("/login", status_code=302)

    img_base64 = plot_success_rate_by_category(db, int(user_id))
    if not img_base64:
        return HTMLResponse(_no_data_html("데이터가 없습니다. 퀘스트를 먼저 추가하세요!"))

    return _styled_plot_page(
        title="카테고리별 평균 성공률",
        desc="AI가 예측한 카테고리별 성공률을 비교해보세요.",
        emoji="🎯",
        img_base64=img_base64
    )


# 성장 추세 시각화 
@app.get("/plot/trend", response_class=HTMLResponse)
def plot_trend(request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse("/login", status_code=302)

    img_base64 = plot_growth_trend(db, int(user_id))
    if not img_base64:
        return HTMLResponse(_no_data_html("충분한 완료 기록 데이터가 없습니다. 퀘스트를 더 완료하세요!"))

    return _styled_plot_page(
        title="시간 경과에 따른 성장 추세",
        desc="시간에 따라 누적 완료 퀘스트 수의 변화를 확인하세요.",
        emoji="📈",
        img_base64=img_base64
    )

# 집중 분야 분석 시각화
@app.get("/plot/focus", response_class=HTMLResponse)
def plot_focus(request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse("/login", status_code=302)

    img_base64 = plot_focus_area(db, int(user_id))
    if not img_base64:
        return HTMLResponse(_no_data_html("데이터가 없습니다. 퀘스트를 먼저 추가하세요!"))

    return _styled_plot_page(
        title="집중 분야 분석",
        desc="내가 가장 몰입하는 카테고리를 시각화합니다.",
        emoji="💡",
        img_base64=img_base64
    )

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

# -----퀘스트 관련----- 
# 퀘스트 생성 추가
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
                "category": quest.category,
                "duration": quest.duration,
                "difficulty": quest.difficulty,
                "motivation": quest.motivation,
                "success_rate": predicted_rate,
            }
        )
        return db_quest

    except Exception as e:
        print(f"[ERROR] 퀘스트 생성 실패: {e}")
        raise HTTPException(status_code=400, detail="퀘스트 생성 중 오류가 발생했습니다.")


# 특정 사용자 퀘스트 조회 추가
@app.get("/users/{user_id}/quests/", response_model=list[schemas.Quest])
def get_user_quests(user_id: int, db: Session = Depends(get_db)):
    quests = crud.get_quests(db=db, user_id=user_id)
    if not quests and crud.get_user(db, user_id) is None:
        raise HTTPException(status_code=404, detail="User not found")
    return quests

# 퀘스트 목록 UI 엔드포인트
@app.get("/quests/list", response_class=HTMLResponse)
def quests_list(request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)
    
    user_id_int = int(user_id)
    user = crud.get_user(db, user_id_int) 

    if not user:
        return RedirectResponse(url="/logout", status_code=303)

    user_name = user.name
    quests = crud.get_quests_by_user(db, user_id=user_id_int)
    
    # 기본 퀘스트 목록 출력
    quest_list_html = ""
    if quests:
        for quest in quests:
            status_text = " 완료" if quest.completed else " 미완료"
            status_class = "completed" if quest.completed else "pending"
            
            ai_info_text = "AI 추천" if quest.ai_recommended else "직접 등록"
            success_rate_percent = f"{quest.success_rate * 100:.1f}%" if quest.success_rate is not None else "N/A"
            
            quest_list_html += f"""
            <div class="quest-item {status_class}">
                <div class="quest-info">
                    <span class="quest-name">{quest.name}</span>
                    <span class="quest-meta">
                        | 카테고리: {quest.category} 
                        | 난이도: {quest.difficulty or 'N/A'} 
                        | 목표: {quest.duration or 'N/A'}일 
                        | AI 성공률: {success_rate_percent}
                    </span>
                    <p class="motivation-text">[{ai_info_text}] {quest.motivation or '동기 부여 문구 없음'}</p>
                </div>
                <div class="quest-actions">
                    <span class="status-badge">{status_text}</span>
                    <button class="toggle-btn" data-item-id="{quest.id}" data-completed="{quest.completed}">상태 변경</button> 
                    <button class="delete-btn" data-item-id="{quest.id}">삭제</button>
                </div>
            </div>
            """
    else:
        quest_list_html = "<p class='no-quest'>아직 등록된 퀘스트가 없습니다. 위에서 새로운 퀘스트를 추가해 보세요!</p>"

    # 완료된 퀘스트 목록 분리
    completed_quests = [q for q in quests if q.completed]
    active_quests = [q for q in quests if not q.completed]

    # 진행 중인 퀘스트 HTML
    active_quest_html = ""
    if active_quests:
        for quest in active_quests:
            success_rate_percent = f"{quest.success_rate * 100:.1f}%"
            active_quest_html += f"""
            <div class="quest-item pending">
                <div class="quest-info">
                    <span class="quest-name">{quest.name}</span>
                    <span class="quest-meta">
                        | 카테고리: {quest.category} | 난이도: {quest.difficulty or 'N/A'} | 목표: {quest.duration or 'N/A'}일
                    </span>
                    <p class="motivation-text">[{ 'AI 추천' if quest.ai_recommended else '직접 등록'}] {quest.motivation or ''}</p>
                </div>
                <div class="quest-actions">
                    <span class="status-badge">진행 중</span>
                    <button class="toggle-btn" data-item-id="{quest.id}">완료로 변경</button> 
                    <button class="delete-btn" data-item-id="{quest.id}">삭제</button>
                </div>
            </div>
            """
    else:
        active_quest_html = "<p class='no-quest'>현재 진행 중인 퀘스트가 없습니다.</p>"

    # 완료된 퀘스트 HTML
    completed_quest_html = ""
    if completed_quests:
        completed_quests.sort(key=lambda q: q.completed_at or q.created_at, reverse=True)
        for quest in completed_quests[:5]:  # 최근 5개까지만 보여주기
            days = (quest.completed_at - quest.created_at).days if quest.completed_at else "-"
            completed_quest_html += f"""
            <div class="quest-item completed">
                <div class="quest-info">
                    <span class="quest-name">{quest.name}</span>
                    <span class="quest-meta">
                        | 카테고리: {quest.category} | 난이도: {quest.difficulty or 'N/A'} | 기간: {days}일 | 성공률: {quest.success_rate * 100:.1f}%
                    </span>
                    <p class="motivation-text">"{quest.motivation or '동기 없음'}"</p>
                </div>
                <div class="quest-actions">
                    <span class="status-badge">✅ 완료</span>
                    <button class="delete-btn" data-item-id="{quest.id}">삭제</button>
                </div>
            </div>
            """
    else:
        completed_quest_html = "<p class='no-quest'>아직 완료된 퀘스트가 없습니다.</p>"


    # 최종 HTML 렌더링
    html = f"""
    <html>
    <head>
        <title>퀘스트 관리 - AI Quest Tracker</title>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; background-color: #f9fafc; margin: 0; padding: 0; text-align: center; color: #222; }}
            header {{ background: linear-gradient(120deg, #02071e, #030928); color: white; padding: 30px 0; box-shadow: 0 3px 6px rgba(0,0,0,0.1); }}
            h1 {{ font-size: 2em; margin: 0; }}
            .desc {{ font-size: 1em; color: #ddd; margin-top: 5px; }}
            .back-link {{ color: #a0a0ff; text-decoration: none; font-weight: bold; margin-top: 10px; display: inline-block; }}

            .content-container {{
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 30px;
                margin: 40px auto;
                max-width: 800px;
            }}

            .add-quest-card {{
                background: white;
                border-radius: 12px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                padding: 30px;
                width: 100%;
                text-align: left;
            }}
            .add-quest-card h2 {{ color: #02071e; margin-top: 0; border-bottom: 2px solid #eee; padding-bottom: 10px; margin-bottom: 20px; }}
            .add-quest-card label {{ display: block; margin-top: 15px; font-weight: bold; font-size: 0.95em; color: #444; }}
            .add-quest-card input, .add-quest-card select, .add-quest-card textarea {{ 
                width: 100%; 
                padding: 10px; 
                margin-top: 5px; 
                border-radius: 6px; 
                border: 1px solid #ccc; 
                box-sizing: border-box; 
                font-size: 1em;
            }}
            .add-quest-card button {{ 
                width: 100%; 
                margin-top: 25px; 
                padding: 12px; 
                background-color: #030928; 
                color: white; 
                border: none; 
                border-radius: 6px; 
                cursor: pointer; 
                font-size: 1.1em;
                transition: background-color 0.2s;
            }}
            .add-quest-card button:hover {{ background-color: #02071e; }}
            
            .quest-list-section {{
                width: 100%;
                text-align: left;
            }}
            .quest-item {{ 
                background: white; 
                border: 1px solid #eee; 
                padding: 15px; 
                margin-bottom: 15px; 
                border-radius: 8px; 
                display: flex; 
                justify-content: space-between; 
                align-items: center; 
                box-shadow: 0 1px 3px rgba(0,0,0,0.05); 
                transition: border-left 0.3s ease;
            }}
            .quest-item.completed {{ border-left: 5px solid #28a745; }}
            .quest-item.pending {{ border-left: 5px solid #ffc107; }}
            .quest-info {{ flex-grow: 1; }}
            .quest-name {{ font-weight: bold; font-size: 1.1em; color: #02071e; margin-right: 10px; }}
            .quest-meta {{ font-size: 0.85em; color: #777; }}
            .motivation-text {{ font-size: 0.9em; color: #555; margin-top: 5px; font-style: italic; }}

            .status-badge {{ 
                padding: 4px 8px; 
                border-radius: 12px; 
                font-size: 0.8em; 
                font-weight: bold; 
                margin-right: 10px;
                display: inline-block;
            }}
            .quest-item.completed .status-badge {{ background-color: #e6f6ec; color: #28a745; }}
            .quest-item.pending .status-badge {{ background-color: #fff4e0; color: #ffc107; }}
            
            .toggle-btn, .delete-btn {{ 
                background-color: #030928; 
                color: white; 
                border: none; 
                padding: 8px 12px; 
                border-radius: 6px; 
                cursor: pointer; 
                margin-left: 5px; 
                font-size: 0.9em; 
                transition: background-color 0.2s;
            }}
            .toggle-btn:hover {{ background-color: #02071e; }}
            .delete-btn {{ background-color: #dc3545; }}
            .delete-btn:hover {{ background-color: #c82333; }}
            
            .no-quest {{ text-align:center; padding:30px; color:#777; font-style: italic; }}

            .quest-list-section h2 {{
                color: #02071e;
                border-left: 6px solid #030928;
                padding-left: 10px;
            }}

            .quest-item.completed {{
                background: #f7fdf8;
                border-left: 5px solid #28a745;
            }}
            .quest-item.pending {{
                background: #fffdf4;
                border-left: 5px solid #ffc107;
            }}
            .quest-item:hover {{
                transform: translateY(-3px);
                transition: transform 0.2s;
            }}
        </style>
    </head>
    <body>
        <header>
            <h1>퀘스트 관리</h1>
            <p class="desc">현재 {user_name}님의 진행 중인 퀘스트 목록입니다.</p>
            <a href="/" class="back-link">← 메인 화면으로 돌아가기</a>
        </header>

        <div class="content-container">
            
            <div class="add-quest-card">
                <h2>새로운 퀘스트 등록</h2>
                <form id="quest-form">
                    <input type="hidden" name="user_id" value="{user_id_int}">
                    
                    <label for="name">퀘스트 이름/설명 (필수)</label>
                    <input type="text" id="name" name="name" placeholder="예: 매일 30분 운동하기" required>

                    <label for="category">카테고리 (필수)</label>
                    <select id="category" name="category" required>
                        <option value="health">Health</option>
                        <option value="study">Study</option>
                        <option value="exercise">Exercise</option>
                        <option value="reading">Reading</option>
                        <option value="work">Work</option>
                        <option value="hobby">Hobby</option>
                        <option value="general">General</option>
                    </select>
                    
                    <label for="duration">기간 (일, 선택)</label>
                    <input type="number" id="duration" name="duration" min="1" placeholder="예: 7 (7일 목표)">
                    
                    <label for="difficulty">난이도 (1~5, 선택)</label>
                    <input type="number" id="difficulty" name="difficulty" min="1" max="5" placeholder="예: 3">
                    
                    <label for="motivation">동기 부여 문구 (선택)</label>
                    <textarea id="motivation" name="motivation" rows="2" placeholder="이 퀘스트를 달성해야 하는 이유를 적어주세요."></textarea>
                    
                    <button type="submit">AI 성공률 예측 및 퀘스트 추가</button>
                </form>
            </div>
            
            <div class="quest-list-section">
                <h2>📝 현재 진행 중인 퀘스트 목록</h2>
                {quest_list_html}
            </div>

            <div class="quest-list-section">
                <h2>🏁 완료된 퀘스트 아카이브</h2>
                <p style="color:#777; font-size:0.9em;">최근 완료한 퀘스트들을 모아봤어요!</p>
                {completed_quest_html}
            </div>
    </div>
        </div>

        <script>
        // 퀘스트 추가 처리 (POST /quests/)
        document.getElementById('quest-form').addEventListener('submit', async (e) => {{
            e.preventDefault();
            const form = e.target;
            const data = Object.fromEntries(new FormData(form).entries());
            
            // 데이터 타입 변환
            data.user_id = parseInt(data.user_id);
            data.name = data.name || data.category;
            data.duration = data.duration ? parseInt(data.duration) : null;
            data.difficulty = data.difficulty ? parseInt(data.difficulty) : null;
            data.motivation = data.motivation || null;

            const res = await fetch("/quests/", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify(data),
                credentials: "include"
            }});

            if (res.ok) {{
                alert("퀘스트가 성공적으로 등록되었습니다. AI 예측 성공률을 확인해 보세요!");
                location.reload();
            }} else {{
                const errorData = await res.json();
                alert("퀘스트 추가 실패: " + (errorData.detail || "서버 오류"));
            }}
        }});

        // 퀘스트 상태 토글 처리 (PATCH /quests/{{quest_id}}/toggle)
        document.querySelectorAll('.toggle-btn').forEach(button => {{
            button.addEventListener('click', async (e) => {{
                const itemId = e.currentTarget.getAttribute('data-item-id');
                
                if (!itemId) {{
                    console.error("Error: item ID is missing for toggle.");
                    alert("퀘스트 ID를 찾을 수 없습니다.");
                    return;
                }}

                const res = await fetch(`/quests/${{itemId}}/toggle`, {{
                    method: "PATCH",
                    credentials: "include"
                }});

                if (res.ok) location.reload();
                else alert("상태 변경 실패");
            }});
        }});

        //퀘스트 삭제 처리 (DELETE /quests/{{quest_id}})
        document.querySelectorAll('.delete-btn').forEach(button => {{
            button.addEventListener('click', async (e) => {{
                if (!confirm("정말로 이 퀘스트를 삭제하시겠습니까?")) return;
                
                const itemId = e.currentTarget.getAttribute('data-item-id');
                
                if (!itemId) {{
                    console.error("Error: item ID is missing for delete.");
                    alert("퀘스트 ID를 찾을 수 없습니다.");
                    return;
                }}

                const res = await fetch(`/quests/${{itemId}}`, {{
                    method: "DELETE",
                    credentials: "include"
                }});

                if (res.ok) location.reload();
                else alert("삭제 실패");
            }});
        }});
    </script>
    </body>
    </html>
    """
    return HTMLResponse(html)

# 퀘스트 완료 토글 (PATCH)
@app.patch("/quests/{quest_id}/toggle")
def toggle_quest(quest_id: int, request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다")

    quest = crud.get_quest_by_user(db, quest_id, int(user_id))
    if not quest:
        raise HTTPException(status_code=404, detail="Quest not found or not yours")

    quest.completed = not quest.completed
    db.commit()
    db.refresh(quest)
    return {"id": quest.id, "completed": quest.completed}

# 퀘스트 삭제 (DELETE)
@app.delete("/quests/{quest_id}")
def delete_quest(quest_id: int, request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="로그인이 필요합니다")

    quest = crud.get_quest_by_user(db, quest_id, int(user_id))
    if not quest:
        raise HTTPException(status_code=404, detail="Quest not found or not yours")

    db.delete(quest)
    db.commit()
    return {"detail": "Deleted"}

#-----recommend 페이지-----
# AI 퀘스트 추천 페이지
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
    user_id_str = request.cookies.get("user_id")
    if not user_id_str:
        # 로그인 쿠키가 없으면 로그인 페이지로 리디렉션 (로그인 강제)
        return RedirectResponse(url="/login", status_code=303)
    try:
        user_id = int(user_id_str)
    except ValueError:
        # user_id 값이 유효한 정수가 아닌 경우 (쿠키 변조 등)
        return RedirectResponse(url="/login", status_code=303)

    form = await request.form()
    quest_name = form.get("quest_name")
    duration = int(form.get("duration"))
    difficulty = int(form.get("difficulty"))
    
    
    # 현재 로그인 기능이 없으므로 user_id=1로 가정
    success_rate = model.predict_success_rate(user_id, quest_name, duration, difficulty)
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
