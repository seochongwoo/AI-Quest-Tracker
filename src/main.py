'''
FastAPI 인스턴스를 생성하고, /, /plot/user, /users/ 등 모든 API 엔드포인트를 정의
get_db() 함수를 통해 DB 세션을 각 요청에 주입하고, /users/ 라우트에서는 crud.py 함수를 호출하여 DB 작업을 수행
'''
# fast api 백엔드를 위한 import
from fastapi import FastAPI, Depends, HTTPException, Request, Form, Query, Body
from typing import Annotated
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.orm import Session
from src import crud, schemas, database
from . import crud, schemas, model
from pydantic import BaseModel
# Db를 위한 import
from .database import SessionLocal, init_db, QuestHistory, Quest
from . import crud, schemas
#  AI 예측 및 시간 관리를 위한 임포트 추가
from sklearn.preprocessing import OneHotEncoder 
import pandas as pd 
from datetime import datetime, timezone
# 시각화를 위한 import
from .habit_analysis import plot_user_progress,plot_success_rate_by_category, plot_focus_area, plot_growth_trend
# ai_recoomend를 위한 import
from .ai_recommend import generate_ai_recommendation
from dotenv import load_dotenv
load_dotenv()

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
                <form method="post" action="/login">
                    <label for="nickname">닉네임:</label>
                    <input type="text" id="nickname" name="nickname" required>

                    <label for="email">이메일:</label>
                    <input type="email" id="email" name="email" required>

                    <button type="submit">로그인 / 회원가입</button>
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
    email = form.get("email")

    # 입력값 검증
    if not nickname or not email:
        return HTMLResponse("<h3>닉네임과 이메일을 모두 입력해주세요.</h3>", status_code=400)

    # 이메일로 먼저 사용자 검색
    user = crud.get_user_by_email(db, email=email)
    if not user:
        # 닉네임으로 중복 체크 (같은 닉네임이 이미 있으면 에러)
        name_conflict = crud.get_user_by_name(db, name=nickname)
        if name_conflict:
            return HTMLResponse("<h3>이미 존재하는 닉네임입니다. 다른 닉네임을 사용해주세요.</h3>", status_code=400)

        # 새로운 사용자 생성
        new_user = schemas.UserCreate(name=nickname, email=email)
        user = crud.create_user(db=db, user=new_user)
        redirect_url = "/onboarding"
    else:
        # 기존 사용자인 경우 온보딩 여부 확인
        if user.consistency_score == 3 and user.risk_aversion_score == 3:
            redirect_url = "/onboarding"
        else:
            redirect_url = "/"

    # 쿠키 설정
    response = RedirectResponse(url=redirect_url, status_code=303)
    response.set_cookie(key="user_id", value=str(user.id), httponly=True, max_age=86400)
    response.set_cookie(key="user_name", value=user.name, httponly=False, max_age=86400)
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


# 특정 사용자 퀘스트 조회
@app.get("/quests/list", response_class=HTMLResponse)
def quests_list(request: Request, db: Session = Depends(get_db)):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)

    user_id_int = int(user_id)
    user = crud.get_user(db, user_id_int)
    if not user:
        return RedirectResponse(url="/logout", status_code=303)

    quests = crud.get_quests_by_user(db, user_id=user_id_int)
    active_quests = [q for q in quests if not q.completed]
    completed_quests = [q for q in quests if q.completed]

    total = len(quests)
    completed = len(completed_quests)
    completion_rate = (completed / total * 100) if total > 0 else 0

    if total == 0:
        ai_message = "🚀 새로운 퀘스트로 첫 도전을 시작해보세요!"
    elif completion_rate >= 80:
        ai_message = "🔥 거의 완벽해요! 이제 더 어려운 도전도 괜찮을 것 같아요."
    elif completion_rate >= 50:
        ai_message = "💪 꾸준함이 보이네요. 남은 퀘스트도 완수해봐요!"
    else:
        ai_message = "🌱 오늘 하나만이라도 도전해볼까요?"

    # QuestHistory에서 최신 progress 가져오기
    def get_latest_progress(q):
        last = (
            db.query(QuestHistory)
            .filter(QuestHistory.quest_id == q.id)
            .order_by(QuestHistory.timestamp.desc())
            .first()
        )
        if last:
            return round(last.progress , 1)
        return 0.0

    def render_quest_card(q):
        progress = get_latest_progress(q)
        rate = f"{q.success_rate * 100:.1f}%" if q.success_rate else "-"
        ai_tag = "🤖 AI 추천" if q.ai_recommended else "직접 등록"
        duration = q.duration or 1
        diff = q.difficulty or "-"
        motivation = q.motivation or "동기 없음"
        category_emoji = {
            "health": "💪", "study": "📚", "reading": "📖",
            "work": "💼", "hobby": "🎨", "exercise": "🏋️‍♂️"
        }.get(q.category, "🎯")

        if q.completed:
            days = (q.completed_at - q.created_at).days if q.completed_at else "-"
            status = f"<span class='status completed'>✅ 완료 ({days}일)</span>"
            card_class = "completed"
        else:
            status = f"<span class='status active'>🕓 진행 중 ({progress:.0f}%)</span>"
            card_class = "active"

        return f"""
        <div class="quest-card {card_class}" data-quest-id="{q.id}" data-duration="{duration}" data-progress="{progress}">
            <div class="emoji">{category_emoji}</div>
            <div class="info">
                <h3>{q.name}</h3>
                <p>{ai_tag} | 성공률: {rate} | 난이도: {diff} | 목표: {duration}일</p>
                <p class="motivation">"{motivation}"</p>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress}%;"></div>
                </div>
            </div>
            <div class="actions">
                {status}
                <button class="toggle-btn" data-item-id="{q.id}">
                    {'🔁 미완료로 변경' if q.completed else '✅ 완료로 변경'}
                </button>
                <button class="delete-btn" data-item-id="{q.id}">🗑 삭제</button>
            </div>
        </div>
        """

    active_html = "".join(render_quest_card(q) for q in active_quests) or "<p class='no-quest'>현재 진행 중인 퀘스트가 없습니다.</p>"
    completed_html = "".join(render_quest_card(q) for q in completed_quests) or "<p class='no-quest'>아직 완료된 퀘스트가 없습니다.</p>"

    html = f"""
    <html>
    <head>
        <title>나의 퀘스트</title>
        <style>
            body {{
                font-family: 'Segoe UI', sans-serif;
                background: #f8f9fc;
                margin: 0;
                color: #222;
            }}
            header {{
                background: linear-gradient(135deg, #02071e, #030928);
                color: white;
                padding: 30px 0;
                text-align: center;
                box-shadow: 0 3px 6px rgba(0,0,0,0.15);
            }}
            .stats {{
                display: flex;
                justify-content: center;
                gap: 25px;
                margin-top: 10px;
                font-size: 0.95em;
                color: #ddd;
            }}
            .ai-feedback {{
                background: #fff9e6;
                border-left: 5px solid #ffd43b;
                color: #555;
                margin: 25px auto;
                max-width: 700px;
                padding: 15px;
                border-radius: 8px;
                font-style: italic;
                text-align: center;
            }}
            .content {{
                max-width: 800px;
                margin: 30px auto;
                padding: 0 20px;
            }}
            h2 {{
                border-left: 6px solid #030928;
                padding-left: 10px;
                color: #030928;
                margin-top: 40px;
            }}
            .quest-card {{
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                margin: 15px 0;
                display: flex;
                align-items: center;
                padding: 15px;
                transition: transform 0.2s;
            }}
            .quest-card:hover {{ transform: translateY(-3px); cursor: pointer; }}
            .quest-card.active {{ border-left: 5px solid #007bff; }}
            .quest-card.completed {{ border-left: 5px solid #28a745; background: #f7fdf8; }}
            .emoji {{ font-size: 2em; width: 50px; text-align: center; }}
            .info h3 {{ margin: 0; color: #111; font-size: 1.1em; }}
            .progress-bar {{
                width: 100%; background: #eee; border-radius: 6px;
                height: 8px; margin-top: 8px; overflow: hidden;
            }}
            .progress-fill {{
                height: 100%; background: #007bff; width: 0%;
                transition: width 0.3s ease-in-out;
            }}
            .status.active {{ color: #007bff; font-weight: bold; }}
            .status.completed {{ color: #28a745; }}
            button {{
                background: #02071e; color: white; border: none;
                border-radius: 6px; padding: 7px 10px; font-size: 0.85em;
                cursor: pointer; margin-left: 5px;
            }}
            .delete-btn {{ background: #dc3545; }}
            .no-quest {{ text-align: center; color: #777; font-style: italic; }}

            /* 추가 폼 */
            .add-form {{
                background: white;
                border-radius: 12px;
                box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                padding: 25px;
                margin-top: 40px;
            }}
            .add-form input, .add-form select, .add-form textarea {{
                width: 100%;
                margin: 8px 0;
                padding: 10px;
                border: 1px solid #ccc;
                border-radius: 6px;
            }}
            .add-form button {{
                background: #030928;
                color: white;
                border: none;
                width: 100%;
                padding: 12px;
                border-radius: 6px;
                font-size: 1em;
                cursor: pointer;
            }}

            /* 진행률 모달 */
            .modal {{
                position: fixed; top: 0; left: 0;
                width: 100%; height: 100%;
                background: rgba(0,0,0,0.5);
                display: flex; justify-content: center; align-items: center;
            }}
            .modal.hidden {{ display: none; }}
            .modal-content {{
                background: white; padding: 20px; border-radius: 10px;
                width: 320px; text-align: center;
            }}
            #progress-grid {{
                display: grid; grid-template-columns: repeat(auto-fill, 30px);
                gap: 5px; justify-content: center; margin-top: 15px;
            }}
            .progress-cell {{
                width: 30px; height: 30px; background: #eee;
                border-radius: 5px; cursor: pointer;
            }}
            .progress-cell.checked {{ background: #007bff; }}
        </style>
    </head>
    <body>
        <header>
            <h1>🌟 {user.name}님의 퀘스트 보드</h1>
            <div class="stats">
                <span>총 퀘스트: {total}</span>
                <span>완료: {completed}</span>
                <span>달성률: {completion_rate:.1f}%</span>
            </div>
        </header>

        <div class="ai-feedback">{ai_message}</div>
        <div class="content">
            <h2>🟢 진행 중</h2>{active_html}
            <h2>🏁 완료</h2>{completed_html}

            <div class="add-form">
                <h2>➕ 새로운 퀘스트 추가</h2>
                <form id="quest-form">
                    <input type="hidden" name="user_id" value="{user_id_int}">
                    <input type="text" name="name" placeholder="예: 매일 30분 운동하기" required>
                    <select name="category">
                        <option value="exercise">운동</option>
                        <option value="study">공부</option>
                        <option value="reading">독서</option>
                        <option value="work">일</option>
                        <option value="hobby">취미</option>
                        <option value="health">건강</option>
                        <option value="general">일반</option>
                    </select>
                    <input type="number" name="duration" placeholder="기간 (일)" min="1">
                    <input type="number" name="difficulty" placeholder="난이도 (1~5)" min="1" max="5">
                    <textarea name="motivation" placeholder="동기 부여 문구 (선택)"></textarea>
                    <button type="submit">AI 성공률 예측 및 추가</button>
                </form>
            </div>
        </div>

        <div id="progress-modal" class="modal hidden">
            <div class="modal-content">
                <h3 id="modal-quest-name"></h3>
                <div id="progress-grid"></div>
                <button id="close-modal">닫기</button>
            </div>
        </div>

        <script>
        // 퀘스트 추가
        document.getElementById('quest-form').addEventListener('submit', async (e) => {{
            e.preventDefault();
            const data = Object.fromEntries(new FormData(e.target).entries());
            data.user_id = parseInt(data.user_id);
            data.duration = data.duration ? parseInt(data.duration) : 1;
            data.difficulty = data.difficulty ? parseInt(data.difficulty) : null;
            const res = await fetch("/quests/", {{
                method: "POST",
                headers: {{ "Content-Type": "application/json" }},
                body: JSON.stringify(data),
                credentials: "include"
            }});
            if (res.ok) location.reload();
            else alert("퀘스트 추가 실패");
        }});

        // 진행률 모달
        const modal = document.getElementById("progress-modal");
        const grid = document.getElementById("progress-grid");
        const modalName = document.getElementById("modal-quest-name");
        const closeModal = document.getElementById("close-modal");

        document.querySelectorAll('.quest-card.active').forEach(card => {{
            card.addEventListener('click', (e) => {{
                if (e.target.classList.contains('toggle-btn') || e.target.classList.contains('delete-btn')) return;

                const questId = card.dataset.questId;
                const duration = parseInt(card.dataset.duration);
                const progress = parseFloat(card.dataset.progress);
                modalName.innerText = card.querySelector('h3').innerText + " 진행 관리";

                grid.innerHTML = '';
                const checkedCells = Math.round((progress / 100) * duration);

                for (let i = 1; i <= duration; i++) {{
                    const cell = document.createElement('div');
                    cell.classList.add('progress-cell');
                    if (i <= checkedCells) cell.classList.add('checked');

                    cell.addEventListener('click', async () => {{
                        cell.classList.toggle('checked');
                        const done = grid.querySelectorAll('.checked').length;
                        const newProgress = parseFloat(((done / duration) * 100).toFixed(1));
                        console.log("duration:", duration, "done:", done, "newProgress:", newProgress)

                        const res = await fetch(`/quests/${{questId}}/progress`, {{
                            method: "PATCH",
                            headers: {{ "Content-Type": "application/json" }},
                            credentials: "include",
                            body: JSON.stringify({{ progress: newProgress }})
                        }});

                        if (res.ok) {{
                            card.querySelector('.progress-fill').style.width = newProgress + "%";
                            card.dataset.progress = newProgress;
                            card.querySelector('.status').innerText = `🕓 진행 중 (${{newProgress}}%)`;
                        }} else {{
                            alert("진행률 업데이트 실패");
                        }}
                    }});

                    grid.appendChild(cell);
                }}
                modal.classList.remove('hidden');
            }});
        }});
        closeModal.addEventListener('click', () => modal.classList.add('hidden'));

        // 완료 토글
        document.querySelectorAll('.toggle-btn').forEach(btn => {{
            btn.addEventListener('click', async e => {{
                e.stopPropagation();
                const id = e.target.dataset.itemId;
                await fetch(`/quests/${{id}}/toggle`, {{ method: "PATCH", credentials: "include" }});
                location.reload();
            }});
        }});

        // 삭제
        document.querySelectorAll('.delete-btn').forEach(btn => {{
            btn.addEventListener('click', async e => {{
                e.stopPropagation();
                if (!confirm("정말 삭제하시겠습니까?")) return;
                const id = e.target.dataset.itemId;
                await fetch(`/quests/${{id}}`, {{ method: "DELETE", credentials: "include" }});
                location.reload();
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

    # 현재 상태 반전
    previous_state = quest.completed
    quest.completed = not quest.completed

    # UTC 일관성 유지
    if quest.created_at.tzinfo is None:
        quest.created_at = quest.created_at.replace(tzinfo=timezone.utc)

    # 완료로 변경된 경우
    if quest.completed:
        quest.completed_at = datetime.now(timezone.utc)
        duration_days = max((quest.completed_at - quest.created_at).days, 1)

        # 기존 completed 로그가 있으면 새로 추가하지 않음
        last_log = (
            db.query(QuestHistory)
            .filter(QuestHistory.quest_id == quest.id)
            .order_by(QuestHistory.timestamp.desc())
            .first()
        )

        if not last_log or last_log.action != "completed":
            history_entry = QuestHistory(
                quest_id=quest.id,
                user_id=quest.user_id,
                action="completed",
                progress=1.0,
                completed_at=quest.completed_at,
                duration_days=duration_days,
                timestamp=datetime.now(timezone.utc),
            )
            db.add(history_entry)

    # 미완료로 되돌린 경우
    else:
        quest.completed_at = None

        # 최근 로그가 이미 reopened이면 중복 추가 안 함
        last_log = (
            db.query(QuestHistory)
            .filter(QuestHistory.quest_id == quest.id)
            .order_by(QuestHistory.timestamp.desc())
            .first()
        )

        if not last_log or last_log.action != "reopened":
            history_entry = QuestHistory(
                quest_id=quest.id,
                user_id=quest.user_id,
                action="reopened",
                progress=0.0,
                timestamp=datetime.now(timezone.utc),
            )
            db.add(history_entry)

    db.commit()
    db.refresh(quest)

    return {
        "id": quest.id,
        "completed": quest.completed,
        "completed_at": quest.completed_at,
    }

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

# 진행률 표시
class ProgressUpdate(BaseModel):
    progress: float
 
@app.patch("/quests/{quest_id}/progress")
async def update_progress(
    quest_id: int,
    body: ProgressUpdate,
    db: Session = Depends(get_db)
):
    progress = round(body.progress, 1)

    quest = db.query(Quest).filter(Quest.id == quest_id).first()
    if not quest:
        raise HTTPException(status_code=404, detail="Quest not found")

    quest.progress = progress
    db.commit()
    db.refresh(quest)

    # 기록 남기기
    last = (
        db.query(QuestHistory)
        .filter(QuestHistory.quest_id == quest_id)
        .order_by(QuestHistory.timestamp.desc())
        .first()
    )
    if not last or abs(last.progress - quest.progress) >= 0.1:
        db.add(
            QuestHistory(
                quest_id=quest.id,
                user_id=quest.user_id,
                action="progress_update",
                progress=quest.progress,
                timestamp=datetime.now(timezone.utc),
            )
        )
        db.commit()

    return {"id": quest.id, "progress": quest.progress}


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
async def recommend_result(
    request: Request,
    quest_name: Annotated[str, Form()],
    duration: Annotated[int, Form()],
    difficulty: Annotated[int, Form()]
):
    user_id_str = request.cookies.get("user_id")
    if not user_id_str:
        return RedirectResponse(url="/login", status_code=303)

    try:
        user_id = int(user_id_str)
    except ValueError:
        return RedirectResponse(url="/login", status_code=303)

    # 1. 성공 확률 예측
    success_rate = model.predict_success_rate(user_id, quest_name, duration, difficulty)
    percent = round(success_rate * 100, 1)

    # 2. 사용자 프로필 데이터 로드 (새로운 함수 호출)
    user_profile = crud.get_user_profile_for_ai(user_id) 
    
    # 3. Gemini AI 조언 생성 (변경된 인자 전체 전달)
    ai_tip = generate_ai_recommendation(
        quest_name=quest_name,
        duration=duration,
        difficulty=difficulty,
        # 사용자 프로필 데이터 전달
        consistency_score=user_profile["consistency_score"],
        risk_aversion_score=user_profile["risk_aversion_score"],
        total_quests=user_profile["total_quests"],
        completed_quests=user_profile["completed_quests"],
        preferred_category=user_profile["preferred_category"]
    )
    
    # 4. 성공률 메시지 및 색상 설정 (기존 로직 유지)
    if percent >= 70:
        color = "#28a745"
    elif percent >= 50:
        color = "#ffc107"
    else:
        color = "#dc3545"

    if percent >= 80:
        message = "🔥 도전해볼 만한 목표예요!"
    elif percent >= 60:
        message = "💪 충분히 가능성이 있습니다!"
    elif percent >= 40:
        message = "⚖️ 조금 어렵지만 해볼 수 있어요."
    else:
        message = "💀 난이도가 높습니다. 단계를 낮춰보세요."

    # 5. 결과 페이지 렌더링 (HTML 부분은 변경 없음)
    return f"""
    <html>
        <head>
            <title>AI 추천 결과</title>
            <style>
                body {{ font-family:'Segoe UI', sans-serif; text-align:center; background-color:#f8f9fa; margin-top:60px; }}
                .result-box {{ background:white; width:420px; margin:0 auto; border-radius:12px; padding:25px; box-shadow:0 4px 10px rgba(0,0,0,0.1); }}
                .gauge-bar {{ height:25px; border-radius:10px; background-color:#e9ecef; overflow:hidden; margin-top:15px; }}
                .gauge-fill {{ height:100%; background-color:{color}; width:{percent}%; text-align:right; color:white; font-weight:bold; padding-right:8px; border-radius:10px; transition:width 0.6s ease-in-out; }}
                .ai-tip {{ background:#f1f3f5; border-left:4px solid #0078d4; padding:12px; margin-top:20px; border-radius:8px; text-align:left; color:#333; }}
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

                <div class="ai-tip">
                    <b>💬 AI 코치의 조언</b><br>
                    {ai_tip}
                </div>

                <br>
                <a href="/recommend">🔁 다시 예측하기</a> | <a href="/">🏠 홈으로</a>
            </div>
        </body>
    </html>
    """

# uvicorn src.main:app --reload
