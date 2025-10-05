'''
데이터 분석, 시각화 및 ML
Pandas를 사용해 사용자별 완료 수나 퀘스트별 완료율 등을 계산
Base64 문자열로 인코딩하여 main.py의 HTML 응답으로 직접 전달하는 함수
'''

import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sqlalchemy.orm import Session
from .database import SessionLocal, Quest, init_db

DATA_PATH = "data/sample_quests.csv"

def load_data_from_db():
    """DB의 Quest 테이블 데이터를 Pandas DataFrame으로 로드합니다."""
    db: Session = SessionLocal()
    try:
        # DB에서 모든 퀘스트를 조회
        quests = db.query(Quest).all()
        
        # 퀘스트 객체 리스트를 DataFrame으로 변환
        # 참고: DB 컬럼명('name', 'duration')을 기존 ML 코드의 컬럼명('quest', 'days')에 맞게 매핑합니다.
        data_list = []
        for q in quests:
            data_list.append({
                'user_id': q.user_id,
                'quest': q.name,        # Quest.name -> DataFrame['quest']
                'days': q.duration,     # Quest.duration -> DataFrame['days']
                'difficulty': q.difficulty,
                'completed': q.completed
            })
        
        df = pd.DataFrame(data_list)
        
        # completed 필드를 boolean에서 int(0 또는 1)로 변환
        if not df.empty:
            df['completed'] = df['completed'].astype(int)
            
        return df

    except Exception as e:
        print(f"DB에서 데이터를 로드하는 중 오류 발생: {e}")
        return pd.DataFrame()
        
    finally:
        db.close()


def load_data():
    """모델 학습 및 그래프 생성에 필요한 데이터를 로드합니다. (DB 우선)"""
    df = load_data_from_db()
    
    # DB에서 데이터를 로드하지 못했거나 데이터가 비어있을 경우에만 CSV 로드 시도
    if df.empty and pd.io.common.file_exists(DATA_PATH):
        print("DB에 데이터가 없거나 오류가 발생했습니다. CSV 파일에서 데이터를 로드합니다.")
        return pd.read_csv(DATA_PATH)
    
    return df

def get_user_completed():
    data = load_data()
    return data.groupby("user_id")["completed"].sum().to_dict()

def get_quest_completion_rate():
    data = load_data()
    return data.groupby("quest")["completed"].mean().round(2).to_dict()


# 그래프 생성 함수
def plot_user_completed():
    data = load_data()
    user_completed = data.groupby('user_id')['completed'].sum()
    
    plt.figure(figsize=(6,4))
    user_completed.plot(kind='bar', color='skyblue')
    plt.title("사용자별 완료한 퀘스트 수")
    plt.xlabel("user_id")
    plt.ylabel("완료 퀘스트 수")
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    # 이미지 메모리에 저장
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    # base64로 인코딩 후 HTML img src로 사용 가능
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def plot_quest_completion_rate():
    data = load_data()
    quest_completed = data.groupby('quest')['completed'].mean()
    
    plt.figure(figsize=(8,4))
    quest_completed.plot(kind='bar', color='lightgreen')
    plt.title("퀘스트별 완료율")
    plt.xlabel("퀘스트")
    plt.ylabel("완료율")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

# 한글 폰트 설정 
from matplotlib import rc

rc('font', family='Malgun Gothic')  
plt.rcParams['axes.unicode_minus'] = False  

DATA_PATH = "data/sample_quests.csv"