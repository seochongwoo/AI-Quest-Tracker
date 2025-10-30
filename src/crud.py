'''
DB 관리 및 데이터 구조 정의 (백본)
database.py의 모델과 schemas.py의 형식을 사용하여 실제 DB와의 상호작용(생성, 읽기, 업데이트, 삭제)을 위한 함수
'''
from sqlalchemy.orm import Session
from .database import User, Quest, QuestHistory, SessionLocal
from .schemas import UserCreate, QuestCreate, UserUpdateScores
from . import model
from datetime import datetime, timezone

# User CRUD 함수
def get_user(db: Session, user_id: int):
    """ID로 사용자 정보를 조회합니다."""
    return db.query(User).filter(User.id == user_id).first()

def get_users(db: Session, skip: int = 0, limit: int = 100):
    """사용자 목록을 조회합니다."""
    return db.query(User).offset(skip).limit(limit).all()

def create_user(db: Session, user: UserCreate):
    """새로운 사용자를 생성합니다."""
    # DB 모델 인스턴스 생성
    db_user = User(name=user.name, email=user.email)
    
    # DB에 추가, 커밋
    db.add(db_user)
    db.commit()
    db.refresh(db_user) # ID와 같은 자동 생성된 값을 로드
    return db_user

# Quest CRUD 함수
def create_user_quest(db: Session, quest: QuestCreate):

    predicted_rate = model.predict_success_rate(
        user_id=quest.user_id,
        quest_name=quest.name,
        duration=quest.duration,
        difficulty=quest.difficulty
    )

    quest_data = quest.model_dump()
    quest_data['success_rate'] = predicted_rate

    db_quest = Quest(**quest_data)
    db.add(db_quest)
    db.commit()
    db.refresh(db_quest)
    
    history_entry = QuestHistory(
        quest_id=db_quest.id,
        user_id=db_quest.user_id,
        action="created",
        progress=0.0,
        started_at=db_quest.created_at,
        timestamp=datetime.now(timezone.utc)
    )
    db.add(history_entry)
    db.commit()

    return db_quest

# 퀘스트 목록 조회
def get_quests(db: Session, skip: int = 0, limit: int = 100):
    return db.query(Quest).order_by(Quest.id.desc()).offset(skip).limit(limit).all()

# 특정 사용자 ID의 퀘스트 목록을 최신 순으로 조회 (토글/삭제용)
def get_quest_by_user(db: Session, quest_id: int, user_id: int):
    return db.query(Quest).filter(
        Quest.id == quest_id,
        Quest.user_id == user_id
    ).first()

# 특정 유저의 모든 퀘스트 목록 (대시보드/목록용)
def get_quests_by_user(db: Session, user_id: int, skip: int = 0, limit: int = 100):
    return (
        db.query(Quest)
        .filter(Quest.user_id == user_id)
        .offset(skip)
        .limit(limit)
        .all()
    )

# ID로 퀘스트 조회
def get_quest(db: Session, quest_id: int):
    return db.query(Quest).filter(Quest.id == quest_id).first()

# 퀘스트 완료로 변경 
def mark_quest_complete(db: Session, quest_id: int):
    db_quest = db.query(Quest).filter(Quest.id == quest_id).first()
    if db_quest:
        db_quest.completed = True
        db.commit()
        db.refresh(db_quest)
        return db_quest
    return None

# 새로운 퀘스트 생성 및 DB 저장
def create_quest(db: Session, quest_data: dict):

    db_quest = Quest(**quest_data)
    db.add(db_quest)
    db.commit()
    db.refresh(db_quest)
    
    history_entry = QuestHistory(
        quest_id=db_quest.id,
        user_id=db_quest.user_id,
        action="created",
        progress=0.0,
        started_at=datetime.now(timezone.utc),
        timestamp=datetime.now(timezone.utc)
    )
    db.add(history_entry)
    db.commit()

    return db_quest

# 간단한 로그인 기능
def get_user_by_name(db: Session, name: str):
    return db.query(User).filter(User.name == name).first()

def create_simple_user(db: Session, name: str):
    user = User(name=name, email=f"{name}@auto.local")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

# 이름으로 유저 찾기 함수
def get_user_by_name(db: Session, name: str):
    return db.query(User).filter(User.name == name).first()

# 이메일로 사용자 조회
def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

# 성향 점수 업데이트 함수
def update_user_scores(db: Session, user_id: int, scores: UserUpdateScores):
    db_user = db.query(User).filter(User.id == user_id).first()
    if db_user:
        db_user.consistency_score = scores.consistency_score
        db_user.risk_aversion_score = scores.risk_aversion_score
        db.commit()
        db.refresh(db_user)
    return db_user

# ai 조언 생성에 필요한 사용자의 성향 및 통계 데이터를 조회
def get_user_profile_for_ai(user_id: int):
    db = SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    db.close()
    
    if not user:
        # 사용자가 없을 경우 기본값 반환
        return {
            "consistency_score": 3,
            "risk_aversion_score": 3,
            "total_quests": 10,
            "completed_quests": 5,
            "preferred_category": None
        }

    return {
        "consistency_score": user.consistency_score if hasattr(user, 'consistency_score') else 3,
        "risk_aversion_score": user.risk_aversion_score if hasattr(user, 'risk_aversion_score') else 3,
        "total_quests": user.total_quests if hasattr(user, 'total_quests') else 10,
        "completed_quests": user.completed_quests if hasattr(user, 'completed_quests') else 5,
        "preferred_category": user.preferred_category if hasattr(user, 'preferred_category') else None
    }