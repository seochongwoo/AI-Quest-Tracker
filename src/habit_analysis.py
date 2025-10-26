'''
데이터 분석, 시각화 및 ML
API 서버와 별개로 데이터를 로드하고 분석 결과를 터미널에 출력
'''

import io
import base64
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func
from .database import Quest, QuestHistory

# GUI 없는 'Agg' 백엔드 사용 설정 (오류방지)
matplotlib.use('Agg')

# Matplotlib 한글 폰트 설정 
try:
    plt.rcParams['font.family'] = 'Malgun Gothic'
except:
    plt.rcParams['font.family'] = 'DejaVu Sans' 

plt.rcParams['axes.unicode_minus'] = False 

def plot_user_progress(db: Session, user_id: int):
    """개인 퀘스트 진행 현황을 시각화"""
    quests = db.query(Quest).filter(Quest.user_id == user_id).all()
    if not quests:
        return None

    df = pd.DataFrame([{
        "name": q.name,
        "completed": q.completed,
        "category": q.category,
        "success_rate": q.success_rate
    } for q in quests])

    # 완료 상태 비율 시각화
    fig, ax = plt.subplots(figsize=(5, 4))
    df['completed'].value_counts().plot(
        kind='pie', autopct='%1.0f%%', colors=['lightcoral', 'lightgreen'],
        labels=['미완료', '완료'], startangle=90, ax=ax
    )
    ax.set_title("📊 완료/미완료 비율")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def plot_success_rate_by_category(db: Session, user_id: int):
    """카테고리별 성공률 시각화"""
    quests = db.query(Quest).filter(Quest.user_id == user_id).all()
    if not quests:
        return None

    df = pd.DataFrame([{
        "category": q.category,
        "success_rate": q.success_rate
    } for q in quests])

    fig, ax = plt.subplots(figsize=(6, 4))
    df.groupby('category')['success_rate'].mean().plot(
        kind='bar', color='skyblue', ax=ax
    )
    ax.set_title("🎯 카테고리별 평균 성공률")
    ax.set_xlabel("카테고리")
    ax.set_ylabel("AI 예측 성공률")
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def plot_growth_trend(db: Session, user_id: int):
    """사용자의 누적 퀘스트 완료 추세를 시각화합니다."""
    # 'completed' 액션이 발생한 날짜별 카운트 조회
    trend_data = db.query(
        func.date(QuestHistory.timestamp).label('date'),
        func.count(QuestHistory.id).label('completed_count')
    ).filter(
        QuestHistory.user_id == user_id,
        QuestHistory.action == "completed"
    ).group_by(
        func.date(QuestHistory.timestamp)
    ).order_by(
        'date'
    ).all()

    if not trend_data:
        return None

    df = pd.DataFrame(trend_data, columns=['date', 'completed_count'])
    df['date'] = pd.to_datetime(df['date'])
    
    # 누적 합계 계산
    df['cumulative_completed'] = df['completed_count'].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['date'], df['cumulative_completed'], marker='o', linestyle='-', color='#030928')
    ax.set_title('시간 경과에 따른 누적 퀘스트 완료 수', fontsize=16, pad=20)
    ax.set_xlabel('날짜', fontsize=12)
    ax.set_ylabel('누적 완료 수', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45)
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def plot_focus_area(db: Session, user_id: int):
    """사용자가 등록한 퀘스트의 카테고리 분포를 도넛 차트로 시각화합니다."""
    # 사용자의 퀘스트 카테고리별 카운트 조회
    category_counts = db.query(
        Quest.category, 
        func.count(Quest.id).label('count')
    ).filter(
        Quest.user_id == user_id
    ).group_by(
        Quest.category
    ).all()

    if not category_counts:
        return None
        
    df = pd.DataFrame(category_counts, columns=['category', 'count'])
    
    # 시각화 데이터 준비
    counts = df['count']
    labels = df['category']

    colors = ['#030928', '#28a745', '#ffc107', '#007bff', '#dc3545', '#6f42c1', '#20c997', '#e83e8c']
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 도넛 차트 생성
    wedges, texts, autotexts = ax.pie(
        counts, 
        autopct='%1.1f%%', 
        startangle=90, 
        colors=colors[:len(counts)],
        wedgeprops=dict(width=0.4, edgecolor='w') 
    )
    
    # 범례 및 제목 설정
    ax.legend(wedges, labels, title="카테고리", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title('나의 퀘스트 카테고리 분포 (집중 분야)', fontsize=16, pad=20)
    
    # 차트를 원형으로 유지
    ax.axis('equal') 
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_base64