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
from fastapi.templating import Jinja2Templates
import os
# Db를 위한 import
from .database import SessionLocal, init_db, QuestHistory, Quest
from . import crud, schemas
#  A시간 관리를 위한 임포트 추가
from datetime import datetime, timezone
# 시각화를 위한 import
from .habit_analysis import plot_user_progress,plot_success_rate_by_category, plot_focus_area, plot_growth_trend
# ai_recoomend를 위한 import
from .ai_recommend import generate_ai_recommendation
from dotenv import load_dotenv
load_dotenv()

### 서버 시작 시 자동으로 train.py 호출 
from contextlib import asynccontextmanager
import subprocess
import threading

@asynccontextmanager
async def lifespan(app: FastAPI):
    def run_training():
        subprocess.run(["python","-m", "src.train"], check=False)

    # 서버 시작 시
    threading.Thread(target=run_training, daemon=True).start()
    print("✅ 서버 시작: 모델 학습 시작")

    yield


app = FastAPI(title="AI Quest Tracker API", lifespan=lifespan)
MODEL_PATH = "model/model.pkl"
# templates로 html 코드 분리
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
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
#  db 의존성 주입 (FastAPI의 Depends 사용)
def root(request: Request, db: Session = Depends(get_db)): 
    # 1. 로그인 확인
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse(url="/login", status_code=303)
    
    user_id_int = int(user_id)
    # db 객체를 사용하여 crud 함수 호출
    user = crud.get_user(db, user_id_int) 

    if not user:
        # 쿠키는 있지만 DB에 없는 경우 (오류), 로그아웃 페이지로
        return RedirectResponse(url="/logout", status_code=303)

    # 2. 온보딩 완료 확인 (성향 점수가 기본값(3)인지 확인)
    if user.consistency_score == 3 and user.risk_aversion_score == 3:
        return RedirectResponse(url="/onboarding", status_code=303)

    # 3. 데이터 로드 (원래 사용자 코드 유지)
    user_name = user.name
    # db 객체를 사용하여 crud 함수 호출
    quests = crud.get_quests_by_user(db, user_id=user_id_int)

    # 로그인 및 온보딩 완료 시, 기존 메인 화면 렌더링 
    return templates.TemplateResponse(
        "home.html",
        {
            "request": request,
            "user_name": user.name,
        },
    )

# -----시각화 관련 라우트 (habit_analyis), 데이터 시각화 페이지-----

# 데이터 허브 페이지
@app.get("/plot/dashboard", response_class=HTMLResponse)
async def plot_dashboard(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return RedirectResponse("/login")
    return templates.TemplateResponse("plot_dashboard.html", {"request": request})


# 공통 헬퍼
def render_no_data(request: Request, message: str = "데이터가 없습니다. 퀘스트를 먼저 추가하세요!"):
    """데이터 없을 때 표시할 페이지"""
    return templates.TemplateResponse("plot_page.html", {
        "request": request,
        "title": "데이터 없음",
        "desc": "아직 분석할 데이터가 부족해요",
        "emoji": "면",
        "message": message
    })

def render_plot_page(request: Request, title: str, desc: str, emoji: str, img_base64: str):
    """모든 시각화 페이지 공통 템플릿"""
    return templates.TemplateResponse("plot_page.html", {
        "request": request,
        "title": title,
        "desc": desc,
        "emoji": emoji,
        "img_base64": img_base64
    })

def get_user_id(request: Request) -> int | None:
    """쿠키에서 user_id 가져오기 + 검증"""
    user_id_str = request.cookies.get("user_id")
    if not user_id_str:
        return None
    try:
        return int(user_id_str)
    except ValueError:
        return None

# 각 시각화 페이지
PLOT_ROUTES = [
    {
        "path": "/plot/user",
        "title": "내 퀘스트 진행 현황",
        "desc": "완료 vs 미완료 비율을 한눈에!",
        "emoji": "파이",
        "func": "plot_user_progress",
        "no_data_msg": "퀘스트를 추가하면 바로 분석됩니다!"
    },
    {
        "path": "/plot/quest",
        "title": "카테고리별 성공률",
        "desc": "AI 예측이 얼마나 정확한지 확인하세요",
        "emoji": "대상",
        "func": "plot_success_rate_by_category",
        "no_data_msg": "카테고리별 데이터가 쌓이면 분석 가능!"
    },
    {
        "path": "/plot/trend",
        "title": "성장 추세 그래프",
        "desc": "내가 얼마나 꾸준히 성장했는지 확인",
        "emoji": "그래프",
        "func": "plot_growth_trend",
        "no_data_msg": "완료된 퀘스트가 3개 이상 필요해요!"
    },
    {
        "path": "/plot/focus",
        "title": "집중 분야 분석",
        "desc": "내가 가장 열정적인 분야는?",
        "emoji": "전구",
        "func": "plot_focus_area",
        "no_data_msg": "다양한 카테고리 퀘스트를 시도해보세요!"
    }
]

# 자동으로 라우트 생성 (코드 80% 감소!)
for route in PLOT_ROUTES:
    @app.get(route["path"], response_class=HTMLResponse)
    async def create_plot_route(
        request: Request,
        db: Session = Depends(get_db),
        r=route  # 클로저 캡처 방지
    ):
        user_id = get_user_id(request)
        if not user_id:
            return RedirectResponse("/login")

        # 동적 함수 호출
        plot_func = globals().get(r["func"])
        if not plot_func:
            return render_no_data("시각화 기능을 찾을 수 없습니다.")

        img_base64 = plot_func(db, user_id)
        
        if not img_base64:
            return render_no_data(r["no_data_msg"])

        return render_plot_page(
            request=request,
            title=r["title"],
            desc=r["desc"],
            emoji=r["emoji"],
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

    streak = user.streak_days or 0

    if total == 0:
        ai_message = "🚀 새로운 퀘스트로 첫 도전을 시작해보세요!"
    elif streak == 0:
        ai_message = "오늘 다시 시작해볼까요? 꾸준함이 힘이에요!"
    elif streak < 3:
        ai_message = f"{streak}일 연속 도전 중이에요! 작은 습관이 큰 변화를 만들어요."
    elif streak < 7:
        ai_message = f"{streak}일째 성장 중이에요! 이 페이스라면 멀지 않았어요."
    elif streak < 30:
        ai_message = f"{streak}일 연속! 놀라운 꾸준함이에요!"
    else:
        ai_message = f"🌟 {streak}일 연속 달성! 전설적인 성취예요."

    # 완료율 보조 메시지 (보완용)
    if completion_rate >= 80:
        ai_message += " 🎯 거의 완벽해요! 새로운 도전도 괜찮겠어요."
    elif completion_rate >= 50:
        ai_message += " 💪 절반 이상 완수했어요. 끝까지 가봅시다!"
    else:
        ai_message += " 🚀 오늘은 하나만이라도 도전해볼까요?"

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
    completed_html = "".join(render_quest_card(q) for q in completed_quests) or "<p class='no-quest'>완료된 퀘스트가 없습니다.</p>"

    return templates.TemplateResponse(
        "quests_list.html",
        {
            "request": request,
            "user": user,
            "total": total,
            "completed": completed,
            "completion_rate": completion_rate,
            "streak": streak,
            "ai_message": ai_message,
            "active_html": active_html,
            "completed_html": completed_html, 
        },
    )

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

        # streak day 로직 추가
        streak = crud.calculate_streak_days(db, int(user_id))
        user = crud.get_user(db, int(user_id))
        if user:
            user.streak_days = streak
            db.commit()

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

        # 미완료로 바뀐 경우 streak 다시 계산
        streak = crud.calculate_streak_days(db, int(user_id))
        user = crud.get_user(db, int(user_id))
        if user:
            user.streak_days = streak
            db.commit()


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
async def recommend_page(request: Request):
    return templates.TemplateResponse("recommend.html", {"request": request})

@app.post("/recommend/result", response_class=HTMLResponse)
async def recommend_result(
    request: Request,
    quest_name: str = Form(...),
    duration: int = Form(...),
    difficulty: int = Form(...)
):
    user_id_str = request.cookies.get("user_id")
    if not user_id_str:
        return RedirectResponse("/login")
    try:
        user_id = int(user_id_str)
    except:
        return RedirectResponse("/login")

    # AI 예측
    success_rate = model.predict_success_rate(user_id, quest_name, duration, difficulty)
    percent = round(success_rate * 100, 1)

    user_profile = crud.get_user_profile_for_ai(user_id)
    ai_tip = generate_ai_recommendation(
        quest_name=quest_name,
        duration=duration,
        difficulty=difficulty,
        **user_profile
    )

    # 색상 및 메시지
    if percent >= 70:
        color = "#28a745"
        message = "도전해볼 만한 목표예요!"
    elif percent >= 50:
        color = "#ffc107"
        message = "충분히 가능성이 있습니다!"
    else:
        color = "#dc3545"
        message = "조금 어렵지만 해볼 수 있어요!"

    return templates.TemplateResponse("recommend_result.html", {
        "request": request,
        "quest_name": quest_name,
        "percent": percent,
        "color": color,
        "message": message,
        "ai_tip": ai_tip
    })

# uvicorn src.main:app --reload
