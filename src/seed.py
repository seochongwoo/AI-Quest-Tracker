'''
DB 관리 및 데이터 구조 정의 (백본)
머신 러닝 성공확률 상승을 위한 더미데이터 생성 
'''
import random
from datetime import datetime, timedelta, timezone
from .database import SessionLocal, User, Quest, init_db
from .model import adjust_by_difficulty_duration, get_user_success_rate

def seed_users(db, num_users=5):
    users = []
    for i in range(1, num_users + 1):
        user = User(name=f"user{i}", email=f"user{i}@example.com")
        db.add(user)
        users.append(user)
    db.commit()
    for u in users:
        db.refresh(u)
    return users

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

def calculate_success_rate(db, user_id: int, duration: int, difficulty: int):
    """
    실제 모델과 유사한 논리로 성공 확률 계산
    - 사용자 과거 성공률
    - 난이도/기간 보정
    """
    base_rate = 0.6  # 기본 베이스 확률
    user_rate = get_user_success_rate(user_id)
    adjust_factor = adjust_by_difficulty_duration(difficulty, duration)

    # 가중치 적용
    final_rate = (
        0.5 * base_rate + 
        0.3 * user_rate + 
        0.2 * adjust_factor
    )
    return round(float(max(0.05, min(final_rate, 0.95))), 3)


def seed_quests(db, users, num_quests_per_user=10):
    categories = ["health", "study", "exercise", "reading", "work", "hobby"]
    for user in users:
        for i in range(num_quests_per_user):
            category = random.choice(categories)
            name = f"퀘스트_{i+1}_{category}"
            duration = random.randint(1, 10)
            difficulty = random.randint(1, 5)
            motivation = f"이 목표는 {category} 관련이다"

            # 실제 AI와 동일한 방식으로 success_rate 계산
            success_rate = calculate_success_rate(db, user.id, duration, difficulty)

            # 성공 여부 결정
            completed = random.random() < success_rate

            # 생성일 / 완료일
            created = datetime.now(timezone.utc) - timedelta(days=random.randint(0, 30))
            completed_at = created + timedelta(days=random.randint(0, duration)) if completed else None

            # DB 객체 생성
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
    users = seed_users(db, num_users=5)
    seed_quests(db, users, num_quests_per_user=15)
    db.close()
    print("✅ 더미 데이터 삽입 완료.")

if __name__ == "__main__":
    run_seed()

# python -m src.seed