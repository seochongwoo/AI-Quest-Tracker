'''
DB 관리 및 데이터 구조 정의 (백본)
머신 러닝 성공확률 상승을 위한 더미데이터 생성 
'''
import random
from math import exp
import numpy as np
from datetime import datetime, timedelta, timezone
from .database import SessionLocal, User, Quest, init_db
from .model import get_user_success_rate

# category별 기본 성공률 (현실적 편향)
CATEGORY_BASE = {
    "reading": 0.75,   # 책읽기: 비교적 잘 지키는 편
    "study":   0.65,   # 공부: 중간
    "exercise":0.45,   # 운동: 낮은 편
    "work":    0.6,
    "hobby":   0.55,
    "health":  0.6,
    # 기타는 평균값 사용
}

# difficulty별 평균 설정 
DIFFICULTY_DIST = {
    1: (0.9, 0.10),   # 쉬움: 평균 88%, sd 5%
    2: (0.75, 0.10),
    3: (0.55,  0.12),
    4: (0.35,  0.13),
    5: (0.18,  0.12),   # 매우 어려움: 평균 20%, sd 8%
}

def seed_users(db, num_users=5):
    users = []
    user_bias_map = {} 
    for i in range(1, num_users + 1):
        # 유저별 편향 값을 랜덤하게 설정
        bias = np.random.normal(loc=0.0, scale=0.1) 
        user = User(name=f"user{i}", email=f"user{i}@example.com")
        db.add(user)
        users.append(user)
        user_bias_map[i] = bias 

    db.commit()
    for u in users:
        db.refresh(u)

    return users, user_bias_map

def get_completion_probability(difficulty: int, duration: int) -> float:
    """
    난이도(1~5)에 따라 성공 확률을 계산
    난이도 1: 0.85 (가장 쉬움)
    난이도 5: 0.35 (가장 어려움)
    """
    base_rate = 0.8
    diff_penalty = (difficulty - 1) * 0.1
    duration_penalty = (duration - 1) * 0.03
    prob = max(0.1, base_rate - diff_penalty - duration_penalty)
    return round(prob, 3)

def calculate_success_rate(
        db, 
        user_id: int, 
        duration: int, 
        difficulty: int, 
        category: str = None,  
        user_bias_map: dict = None):
    """
    실제 모델과 유사한 논리로 성공 확률 계산
    - 사용자 과거 성공률
    - 난이도/기간 보정
    """
    # 1 사용자 기반 성공률 (새 사용자면 0.5)
    user_rate = get_user_success_rate(user_id)
    if user_rate is None:
        user_rate = 0.5

    # 2 카테고리 기본값
    cat_base = CATEGORY_BASE.get(category, 0.6)  # 알려진 카테고리 없으면 0.6

    # 3 difficulty 기반 분포 샘플링
    difficulty = int(max(1, min(5, difficulty)))
    dist_mean, dist_sd = DIFFICULTY_DIST.get(difficulty, (0.6, 0.1))

    # 카테고리 베이스와 difficulty 분포를 섞어 "raw_mean" 산출
    # (카테고리 성향이 난이도 체감에 영향을 줌)
    raw_mean = 0.6 * dist_mean + 0.4 * cat_base

    # 난이도 레벨이 높을수록 분산을 약간 증가시켜 현실성 추가
    sd = dist_sd * (1.0 + (difficulty - 3) * 0.1)

    # 4 정규분포에서 샘플링 (클램프 전)
    sampled = np.random.normal(loc=raw_mean, scale=sd)

    # 5 duration 영향(길면 성공률 하락) — 기간이 길수록 소폭 페널티
    duration_penalty = 0.0
    if duration is not None and duration > 1:
        # duration이 클수록 더 불확실해지므로 페널티를 준다 (단계적으로)
        duration_penalty = -0.01 * min(duration - 1, 30)  # 최대 -0.30
    # 편향된 값 가져오기 
    user_bias = user_bias_map.get(user_id, 0.0)     

    # 6 user_rate 보정 (개인화)
    noise = random.uniform(-0.04, 0.04)
    final = 0.1 * user_rate + 0.6 * sampled + 0.1 * user_bias + 0.15 * (1 + duration_penalty) + 0.05 * noise

    # 7 안전 범위로 클램프 및 반환
    final = float(max(0.05, min(0.95, final)))
    return round(final, 3)

def seed_quests(db, users,user_bias_map,  num_quests_per_user=10):
    categories = ["health", "study", "exercise", "reading", "work", "hobby"]
    for user in users:
        for i in range(num_quests_per_user):
            category = random.choice(categories)
            name = f"퀘스트_{i+1}_{category}"
            duration = random.randint(1, 30)
            difficulty = random.randint(1, 5)
            motivation = f"이 목표는 {category} 관련이다"

            success_rate = calculate_success_rate(db, user.id, duration, difficulty,user_bias_map=user_bias_map)
            completed = random.random() < success_rate

            created = datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30))
            completed_at = created + timedelta(days=random.randint(0, duration)) if completed else None

            quest = Quest(
                user_id=user.id,
                name=name,
                category=category,
                duration=duration,
                difficulty=difficulty,
                motivation=motivation,
                completed=completed,
                completed_at=completed_at,
                success_rate=success_rate,  
            )
            db.add(quest)
    db.commit()

def run_seed():
    init_db()
    db = SessionLocal()
    users, user_bias_map = seed_users(db, num_users=5)
    seed_quests(db, users, user_bias_map, num_quests_per_user=15)
    db.close()
    print("✅ 더미 데이터 삽입 완료.")

if __name__ == "__main__":
    run_seed()

# python -m src.seed
