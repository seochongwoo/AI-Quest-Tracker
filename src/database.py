'''
DB 관리 및 데이터 구조 정의 (백본)
SQLAlchemy를 사용하여 SQLite 파일(db.sqlite3)과 연결하는 엔진과 세션을 생성
'''
from sqlalchemy import create_engine, Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = "sqlite:///./db.sqlite3"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# User 테이블
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)

    is_active = Column(Boolean, default=True) # 활성화 필드 추가
    
    quests = relationship("Quest", back_populates="user") # 관계 이름 수정

# Quest 테이블 추가
class Quest(Base):
    __tablename__ = "quests"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String, index=True)
    duration = Column(Integer)
    difficulty = Column(Integer)
    completed = Column(Boolean, default=False)

    user = relationship("User", back_populates="quests")

# DB 생성
def init_db():
    Base.metadata.create_all(bind=engine)