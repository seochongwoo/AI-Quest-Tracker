# 파일: habit_analysis.py

import pandas as pd
import matplotlib.pyplot as plt

# 1. CSV 불러오기
data = pd.read_csv("data/sample_quests.csv")

# 2. 데이터 확인
print("===== 전체 데이터 =====")
print(data)
print("\n===== 기본 통계 =====")
print(data.describe())

# 3. 사용자별 완료한 퀘스트 수
user_completed = data.groupby('user_id')['completed'].sum()
print("\n===== 사용자별 완료 퀘스트 수 =====")
print(user_completed)

# 4. 시각화: 사용자별 완료 퀘스트
plt.figure(figsize=(6,4))
user_completed.plot(kind='bar', color='skyblue')
plt.title("사용자별 완료한 퀘스트 수")
plt.xlabel("user_id")
plt.ylabel("완료 퀘스트 수")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# 5. 시각화: 퀘스트별 완료율
quest_completed = data.groupby('quest')['completed'].mean()
plt.figure(figsize=(8,4))
quest_completed.plot(kind='bar', color='lightgreen')
plt.title("퀘스트별 완료율")
plt.xlabel("퀘스트")
plt.ylabel("완료율")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
