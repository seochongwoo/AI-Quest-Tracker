'''
DB 관리 및 데이터 구조 정의 (백본)
머신 러닝 성공확률 상승을 위한 더미데이터 생성 
'''
import random
from datetime import datetime, timedelta
from .database import SessionLocal, User, Quest, init_db

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

def seed_quests(db, users, num_quests_per_user=10):
    categories = ["health", "study", "exercise", "reading", "work", "hobby"]
    for user in users:
        for i in range(num_quests_per_user):
            name = f"퀘스트_{i+1}"
            category = random.choice(categories)
            # duration 1~10일
            duration = random.randint(1, 10)
            # difficulty 1~5
            difficulty = random.randint(1, 5)
            # 동기(motivation) 간단한 문장
            motivation = f"이 목표는 {category} 관련이다"
            # completed 확률적으로 정하기
            completed = random.random() < 0.5  # 약 절반 확률로 완료
            # 완료된 경우 completed_at 설정
            if completed:
                # 생성일로부터 랜덤 시점에 완료됨
                created = datetime.utcnow() - timedelta(days=random.randint(0, 30))
                completed_at = created + timedelta(days=random.randint(0, duration))
            else:
                completed_at = None

            quest = Quest(
                user_id=user.id,
                name=name,
                category=category,
                duration=duration,
                difficulty=difficulty,
                motivation=motivation,
                completed=completed,
                completed_at=completed_at,
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
