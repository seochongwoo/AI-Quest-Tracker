"""
DB 관리 및 데이터 구조 정의 (백본)
현실적 분포 기반의 더미데이터 생성 + ML 학습 품질 개선
"""

import random
from datetime import datetime, timedelta, timezone
import numpy as np
from .database import SessionLocal, User, Quest, init_db
from .model import get_user_success_rate


# 카테고리별 현실적 성공률 평균
# 실제 조사 기반 (운동/공부 지속률, 독서율, 취미 지속 등)
CATEGORY_BASE = {
    "reading": 0.70,    # 책읽기: 비교적 잘 지킴
    "study":   0.55,    # 공부: 의욕 높지만 지속 낮음
    "exercise":0.40,    # 운동: 의욕은 높으나 지속 어려움
    "work":    0.60,    # 일 관련 목표: 중간 이상
    "hobby":   0.50,    # 취미: 꾸준히 하는 사람 적음
    "health":  0.65,    # 건강 관련: 비교적 높음
}

# 난이도별 분포 
DIFFICULTY_DIST = {
    1: (0.90, 0.10),
    2: (0.75, 0.12),
    3: (0.55, 0.13),
    4: (0.30, 0.14),
    5: (0.12, 0.15),
}

# 유저 수/퀘스트 수 확장
NUM_USERS = 20
QUESTS_PER_USER = 30


def seed_users(db, num_users=NUM_USERS):
    """랜덤한 편향값을 가진 유저 더미 생성"""
    users = []
    user_bias_map = {}

    for i in range(1, num_users + 1):
        bias = np.random.normal(loc=0.0, scale=0.15)  # 개인별 편향 (조금 더 강하게)
        user = User(name=f"user{i}", email=f"user{i}@example.com")
        db.add(user)
        users.append(user)
        user_bias_map[i] = bias

    db.commit()
    for u in users:
        db.refresh(u)

    return users, user_bias_map


def calculate_success_rate(db, user_id, duration, difficulty, category, user_bias_map):
    """현실적 성공률 계산 (category + difficulty + duration + 개인차 반영)"""
    # 1. 사용자 과거 성공률
    user_rate = get_user_success_rate(user_id)
    if user_rate is None or np.isnan(user_rate):
        user_rate = 0.5

    # 2. 카테고리 평균
    cat_base = CATEGORY_BASE.get(category, 0.55)

    # 3. 난이도 분포에서 샘플링
    difficulty = int(max(1, min(5, difficulty)))
    dist_mean, dist_sd = DIFFICULTY_DIST[difficulty]

    # 4. raw 평균 계산 (카테고리 영향 40%, 난이도 영향 60%)
    raw_mean = 0.6 * dist_mean + 0.4 * cat_base

    # 5. 기간 패널티 (길면 더 어렵다)
    duration_penalty = 0.006 * min(duration, 60)  # 최대 -0.3
    raw_mean -= duration_penalty

    # 6. 정규분포 샘플링
    sampled = np.random.normal(loc=raw_mean, scale=dist_sd)

    # 7. 개인 편향
    user_bias = user_bias_map.get(user_id, 0.0)

    # 8. 최종 조합 (difficulty 효과를 강화)
    final = (
        0.15 * user_rate +
        0.7 * sampled +
        0.12 * user_bias +
        0.03 * random.uniform(-0.05, 0.05)
    )

    # 9. 안전 범위 클램프
    return float(np.clip(final, 0.05, 0.95))


def seed_quests(db, users, user_bias_map, num_quests_per_user=QUESTS_PER_USER):
    """유저별 퀘스트 생성"""
    categories = list(CATEGORY_BASE.keys())

    for user in users:
        for i in range(num_quests_per_user):
            category = random.choice(categories)
            name = f"퀘스트_{i+1}_{category}"
            duration = random.randint(1, 30)
            difficulty = random.randint(1, 5)
            motivation = f"이 목표는 {category} 관련이다"

            success_rate = calculate_success_rate(db, user.id, duration, difficulty, category, user_bias_map)
            completed = random.random() < success_rate

            created = datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30))
            completed_at = created + timedelta(days=random.randint(1, duration)) if completed else None

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
    """DB 초기화 및 시드 실행"""
    init_db()
    db = SessionLocal()
    users, user_bias_map = seed_users(db)
    seed_quests(db, users, user_bias_map)
    db.close()
    print("✅ 더미 데이터 삽입 완료.")


if __name__ == "__main__":
    run_seed()

# python -m src.seed

