'''
DB 관리 및 데이터 구조 정의 (백본)
database.py의 모델과 schemas.py의 형식을 사용하여 실제 DB와의 상호작용(생성, 읽기, 업데이트, 삭제)을 위한 함수
'''
from sqlalchemy.orm import Session
from .database import User, Quest 
from .schemas import UserCreate, QuestCreate 
from . import database, schemas
from . import model

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
    """특정 사용자(user_id)를 위한 새로운 퀘스트를 생성합니다."""

    predicted_rate = model.predict_success_rate(
        user_id=quest.user_id,
        quest_name=quest.name,
        duration=quest.duration,
        difficulty=quest.difficulty
    )
    
    # Pydantic 모델 데이터를 딕셔너리로 변환
    quest_data = quest.model_dump()
    
    # 딕셔너리에 예측된 성공 확률 추가 
    quest_data['success_rate'] = predicted_rate
    
    # DB 모델 인스턴스 생성
    db_quest = Quest(**quest_data) 
    
    db.add(db_quest)
    db.commit()
    db.refresh(db_quest)
    return db_quest

def get_quests(db: Session, skip: int = 0, limit: int = 100):
    """모든 퀘스트 목록을 조회합니다."""
    return db.query(Quest).order_by(Quest.id.desc()).offset(skip).limit(limit).all()

def get_quest(db: Session, quest_id: int):
    """ID로 특정 퀘스트 조회"""
    return db.query(Quest).filter(Quest.id == quest_id).first()

def mark_quest_complete(db: Session, quest_id: int):
    """퀘스트를 완료 상태로 변경합니다."""
    db_quest = db.query(Quest).filter(Quest.id == quest_id).first()
    if db_quest:
        db_quest.completed = True
        db.commit()
        db.refresh(db_quest)
        return db_quest
    return None

def create_quest(db: Session, quest_data: dict):
    """
    새로운 퀘스트 생성 및 DB 저장
    """
    db_quest = database.Quest(**quest_data)
    db.add(db_quest)
    db.commit()
    db.refresh(db_quest)
    return db_quest