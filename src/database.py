'''
DB 관리 및 데이터 구조 정의 (백본)
SQLAlchemy를 사용하여 SQLite 파일(db.sqlite3)과 연결하는 엔진과 세션을 생성
'''
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey, Float, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

DATABASE_URL = "sqlite:///./db.sqlite3"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User 테이블
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    is_active = Column(Boolean, default=True)
    # 성향 점수
    consistency_score = Column(Integer, default=3)
    risk_aversion_score = Column(Integer, default=3)

    quests = relationship("Quest", back_populates="user", cascade="all, delete")

# Quest 테이블
class Quest(Base):
    __tablename__ = "quests"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String, index=True)
    category = Column(String, default="general")  # 카테고리 추가
    duration = Column(Integer)
    difficulty = Column(Integer)
    motivation = Column(Text, nullable=True)  # 사용자가 입력하는 목표/동기
    completed = Column(Boolean, default=False)
    ai_recommended = Column(Boolean, default=False)
    success_rate = Column(Float, default=0.5)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="quests")

# DB 생성
def init_db():
    Base.metadata.create_all(bind=engine)