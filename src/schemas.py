'''
DB 관리 및 데이터 구조 정의 (백본)
FastAPI와 Pydantic을 사용하여 API로 들어오고 나가는 데이터의 형식과 유효성 정의
DB 객체를 API 형식으로 변환할 때 사용
'''
from pydantic import BaseModel, Field
from datetime import date
from typing import Optional

# 1. User 관련 스키마
class UserBase(BaseModel):
    """사용자 생성 및 업데이트에 필요한 기본 정보"""
    name: str = Field(..., max_length=50)
    email: str = Field(..., max_length=100)

class UserCreate(UserBase):
    """사용자 생성 스키마 (UserBase와 동일)"""
    pass

class User(UserBase):
    """사용자 데이터 반환 스키마"""
    id: int
    is_active: bool = True
    
    class Config:
        # SQLAlchemy 모델을 Pydantic 모델로 변환할 수 있도록 설정
        from_attributes = True 

# 2. Quest 관련 스키마
class QuestBase(BaseModel):
    """퀘스트 생성 및 업데이트에 필요한 기본 정보"""
    user_id: int
    name: str = Field(..., max_length=100)
    duration: Optional[int] = Field(None, description="퀘스트 예상 소요 일수")
    difficulty: Optional[int] = Field(None, description="난이도 (1-5)")

class QuestCreate(QuestBase):
    """퀘스트 생성 스키마"""
    pass

class Quest(QuestBase):
    """퀘스트 데이터 반환 스키마"""
    id: int
    completed: bool = False

    success_rate: float = 0.5
    
    class Config:
        from_attributes = True

# 3. Quest Record 스키마