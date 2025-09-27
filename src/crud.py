'''
DB 관리 및 데이터 구조 정의 (백본)
database.py의 모델과 schemas.py의 형식을 사용하여 실제 DB와의 상호작용(생성, 읽기, 업데이트, 삭제)을 위한 함수
'''
from sqlalchemy.orm import Session
from .database import User, Quest # DB 모델
from .schemas import UserCreate, QuestCreate # Pydantic 스키마

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
    # **kwargs로 Pydantic 모델 데이터를 DB 모델에 전달합니다.
    db_quest = Quest(**quest.model_dump()) 
    
    db.add(db_quest)
    db.commit()
    db.refresh(db_quest)
    return db_quest

def get_quests(db: Session, user_id: int):
    """특정 사용자의 퀘스트 목록을 조회합니다."""
    return db.query(Quest).filter(Quest.user_id == user_id).all()

def mark_quest_complete(db: Session, quest_id: int):
    """퀘스트를 완료 상태로 변경합니다."""
    db_quest = db.query(Quest).filter(Quest.id == quest_id).first()
    if db_quest:
        db_quest.completed = True
        db.commit()
        db.refresh(db_quest)
        return db_quest
    return None