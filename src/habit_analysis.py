'''
데이터 분석, 시각화 및 ML
API 서버와 별개로 데이터를 로드하고 분석 결과를 터미널에 출력
'''

import io
import base64
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func
from .database import Quest, QuestHistory

# GUI 없는 백엔드
matplotlib.use('Agg')

# 전역 스타일 설정
plt.style.use('default')
plt.rcParams.update({
    'font.family': 'Malgun Gothic',
    'axes.unicode_minus': False,
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#f8f9fc',
    'axes.edgecolor': '#e2e8f0',
    'axes.grid': True,
    'grid.color': '#e2e8f0',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'text.color': '#2d3748',
    'axes.labelcolor': '#4a5568',
    'xtick.color': '#718096',
    'ytick.color': '#718096',
    'font.size': 11,
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.facecolor': 'white',
    'legend.edgecolor': '#e2e8f0',
})

# 1. 내 퀘스트 현황 
def plot_user_progress(db: Session, user_id: int):
    quests = db.query(Quest).filter(Quest.user_id == user_id).all()
    if not quests:
           return None

    completed = sum(1 for q in quests if q.completed)
    total = len(quests)
    pending = total - completed

    if total == 0:
        return None

    # 데이터
    sizes = [completed, pending]
    labels = ['완료', '진행 중']
    colors = ['#10b981', '#f59e0b']  # 초록, 주황

    # 2D 도넛 + 3D 효과
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111)

    # 도넛 차트
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct=lambda pct: f'{int(pct/100*total)}개\n({pct:.1f}%)',
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.4, edgecolor='white', linewidth=4),
        textprops={'fontsize': 14, 'fontweight': 'bold', 'color': 'white'},
        labeldistance=1.15
    )

    # 3D 효과: 그림자 원
    shadow = plt.Circle((0, 0), 0.7, color='black', alpha=0.15, transform=ax.transData)
    ax.add_patch(shadow)

    # 중앙 원
    centre_circle = plt.Circle((0, 0), 0.35, color='white', ec='#e2e8f0', lw=3)
    ax.add_patch(centre_circle)

    # 중앙 텍스트 (빛나는 효과)
    ax.text(0, 0.1, f'{completed}', ha='center', va='center', fontsize=48, fontweight='bold', color='#10b981')
    ax.text(0, -0.1, f'/{total}', ha='center', va='center', fontsize=28, color='#666')

    # 제목
    ax.set_title('내 퀘스트 현황', fontsize=26, fontweight='bold', pad=40, color='#1a202c')

    # 3D 느낌 주는 기울기
    for wedge in wedges:
        wedge.set_edgecolor('white')
        wedge.set_linewidth(5)

    ax.axis('equal')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=180, bbox_inches='tight', facecolor='#f8f9fc')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

# 2. 카테고리별 성공률
def plot_success_rate_by_category(db: Session, user_id: int):
    quests = db.query(Quest).filter(Quest.user_id == user_id).all()
    if not quests:
        return None

    df = pd.DataFrame([
        {"category": q.category or "기타", "rate": q.success_rate or 0}
        for q in quests
    ])
    if df.empty:
        return None

    avg_rates = df.groupby('category')['rate'].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(avg_rates)), avg_rates.values, 
                  color='#667eea', edgecolor='#5a67d8', linewidth=2.5, alpha=0.9)

    # 그라데이션
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.coolwarm(i / len(bars)))

    ax.set_title('카테고리별 AI 예측 성공률', fontsize=20, fontweight='bold', pad=25, color='#1a202c')
    ax.set_ylabel('성공률', fontsize=12)
    ax.set_xticks(range(len(avg_rates)))
    ax.set_xticklabels(avg_rates.index, rotation=20, ha='right', fontsize=10)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))

    # 값 라벨
    for i, v in enumerate(avg_rates.values):
        ax.text(i, v + 0.03, f'{v:.0%}', ha='center', fontweight='bold', fontsize=11, color='#2d3748')

    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='#f8f9fc')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


# 3. 성장 추세 

def plot_growth_trend(db: Session, user_id: int):
    data = db.query(
        func.date(QuestHistory.timestamp).label('date'),
        func.count(QuestHistory.id).label('count')
    ).filter(
        QuestHistory.user_id == user_id,
        QuestHistory.action == "completed"
    ).group_by('date').order_by('date').all()

    if not data or len(data) < 2:
        return None

    df = pd.DataFrame(data, columns=['date', 'count'])
    df['date'] = pd.to_datetime(df['date'])
    df['cumulative'] = df['count'].cumsum()

    fig, ax = plt.subplots(figsize=(11, 6.5))

    # 빛나는 효과
    ax.plot(df['date'], df['cumulative'], color='#a0aec0', linewidth=12, alpha=0.4)
    line = ax.plot(df['date'], df['cumulative'], color='#667eea', linewidth=4,
                   marker='o', markersize=9, markerfacecolor='#5a67d8', markeredgecolor='white', markeredgewidth=2)

    ax.set_title('나의 성장 곡선', fontsize=22, fontweight='bold', pad=30, color='#1a202c')
    ax.set_xlabel('날짜', fontsize=12)
    ax.set_ylabel('누적 완료 수', fontsize=12)
    ax.grid(True, alpha=0.4)

    # 포인트 라벨
    for i, (date, cum) in enumerate(zip(df['date'], df['cumulative'])):
        if i == 0 or i == len(df)-1 or i % 3 == 0:
            ax.text(date, cum + 0.8, str(cum), ha='center', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='white', edgecolor='#667eea', alpha=0.9))

    plt.xticks(rotation=30)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='#f8f9fc')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


# 4. 집중 분야 
def plot_focus_area(db: Session, user_id: int):
    counts = db.query(
        Quest.category, func.count(Quest.id).label('cnt')
    ).filter(Quest.user_id == user_id).group_by(Quest.category).all()

    if not counts:
        return None

    labels, values = zip(*[(c or "기타", n) for c, n in counts])
    colors = ['#667eea', '#f093fb', '#a8edea', '#fed6e3', '#ff9a9e', '#a18cd1', '#fad0c4']

    fig, ax = plt.subplots(figsize=(10, 10), facecolor='none')
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct='%1.1f%%', startangle=90,
        colors=colors[:len(values)], wedgeprops=dict(width=0.4, edgecolor='white', linewidth=4),
        textprops={'color': 'black', 'weight': 'bold', 'fontsize': 12},
        labeldistance=1.1
    )

    # 글로우
    for w in wedges:
        w.set_edgecolor('#ffffff')
        w.set_linewidth(5)
        w.set_alpha(0.95)

    total = sum(values)
    ax.text(0, 0, f'{total}\n총 퀘스트', ha='center', va='center', fontsize=20,
            fontweight='bold', color='#667eea',
            bbox=dict(boxstyle="circle,pad=0.8", facecolor='white', alpha=0.9))

    ax.set_title('집중 분야 TOP', fontsize=24, fontweight='bold', pad=50, color='#1a202c')
    ax.axis('equal')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', transparent=True)
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)
    return img_base64