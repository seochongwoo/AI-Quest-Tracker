'''
DB 관리 및 데이터 구조 정의 (백본)
FastAPI와 Pydantic을 사용하여 API로 들어오고 나가는 데이터의 형식과 유효성 정의
DB 객체를 API 형식으로 변환할 때 사용
'''
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

# ---------- User ----------
class UserBase(BaseModel):
    name: str
    email: str

class UserCreate(UserBase):
    email: Optional[str] = Field("dummy@user.com", description="Simple login을 위한 더미 이메일")
    
    # 온보딩 질문에서 받을 성향 점수
    consistency_score: Optional[int] = Field(3, ge=1, le=5)
    risk_aversion_score: Optional[int] = Field(3, ge=1, le=5)

class User(UserBase):
    id: int
    is_active: bool = True

    consistency_score: int
    risk_aversion_score: int

    class Config:
        from_attributes = True

# 성향 점수 업데이트를 위한 스키마
class UserUpdateScores(BaseModel):
    consistency_score: int = Field(..., ge=1, le=5)
    risk_aversion_score: int = Field(..., ge=1, le=5)

# ---------- Quest ----------
class QuestBase(BaseModel):
    user_id: int
    name: str
    category: Optional[str] = "general"
    duration: Optional[int] = Field(None, ge=1, description="예상 일수 (1 이상)")
    difficulty: Optional[int] = Field(None, ge=1, le=5, description="난이도 (1~5)")
    motivation: Optional[str] = None

class QuestCreate(QuestBase):
    pass

class Quest(QuestBase):
    id: int
    completed: bool
    ai_recommended: bool
    success_rate: float
    created_at: datetime
    completed_at: Optional[datetime]

    class Config:
        from_attributes = True