import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

DATA_PATH = "data/sample_quests.csv"

def load_data():
    return pd.read_csv(DATA_PATH)

def get_user_completed():
    data = load_data()
    return data.groupby("user_id")["completed"].sum().to_dict()

def get_quest_completion_rate():
    data = load_data()
    return data.groupby("quest")["completed"].mean().round(2).to_dict()

# =====================
# 그래프 생성 함수
# =====================
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