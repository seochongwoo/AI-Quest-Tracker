![header](https://capsule-render.vercel.app/api?type=waving&color=0:02071e,80:030928&height=300&section=header&text=AI%20Quest%20Tracker&fontSize=70&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Track%20your%20habits%20and%20get%20AI-powered%20feedback!&descAlignY=51&descAlign=50)

# 🌟 AI Quest Tracker

- **AI Quest Tracker**는 오픈소스 habit tracker를 기반으로, **머신러닝을 활용해 퀘스트(습관) 성공 확률을 예측**하고, **맞춤형 퀘스트를 추천**하며, 간단한 **AI 피드백**을 제공하는 프로젝트입니다. 
- 사용자는 자신이 원하는 퀘스트를 추가하고, 실행 결과를 기록하며, AI로부터 동기부여와 피드백을 받을 수 있습니다.
- [Habitica](https://habitica.com/)와 같은 habit tracker에서 영감을 받았으며, **데이터 기반 개인화**를 주요 목표로 합니다.

---

##  Table of Contents
1. [Getting Started](#getting-started)  
2. [Features](#features)  
   1. [샘플 데이터](#샘플-데이터)  
   2. [모델 학습](#모델-학습)  
   3. [API 실행](#api-실행)  
   4. [예측 결과](#예측-결과)  
3. [Demo](#demo)  
4. [API Docs](#api-docs)  
5. [기술 스택](#기술-스택)  
6. [Reference](#reference)  
7. [License](#license)  

---

##  Getting Started

### Requirements
- Python 3.9+
- pip

### Installation
```bash
# 저장소 클론
git clone https://github.com/username/AI-Quest-Tracker.git
cd AI-Quest-Tracker

# 패키지 설치
pip install -r requirements.txt
```

### Running
```bash
# 1. AI 모델 학습 (최초 1회 필수)
# model/model.pkl 파일을 생성합니다.
python -m src.train

# 2. FastAPI 실행 (서버 실행)
uvicorn src.main:app --reload
```

- 실행 후: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 접속하면 Swagger UI에서 API 확인 가능 ✅

---

##  Features

###  샘플 데이터
`data/sample_quests.csv`
```csv
user_id,quest,days,completed
1,"아침 7시 기상",3,1
1,"물 2L 마시기",7,0
2,"하루 30분 운동",5,1
2,"영어 단어 10개 외우기",7,0
3,"저녁 10시 취침",7,1
```

###  모델 학습
`src/train.py`  
- `scikit-learn`으로 간단한 로지스틱 회귀 모델 학습  
- 학습된 모델을 `model/model.pkl`로 저장  

```python
joblib.dump(model, "model/model.pkl")
```

###  API 실행
`src/main.py`  
- FastAPI 서버 구동  
- `/predict` 엔드포인트 제공  

```http
GET /predict?duration=3&difficulty=2
```

### 예측 결과
```json
{
  "duration": 3,
  "difficulty": 2,
  "success_prob": 0.74
}
```

---

##  Demo

예시:
- 데이터 학습 화면
- FastAPI Swagger 실행 화면
- 예측 결과 API 호출 화면  

---

##  API Docs

---

##  기술 스택
- **Backend**: Python, FastAPI  
- **ML**: scikit-learn, joblib  
- **DB (옵션)**: SQLite 
- **Visualization**: matplotlib, Plotly  

---

##  Reference
- [Habitica](https://habitica.com/)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)  
- [FastAPI](https://fastapi.tiangolo.com/)  

---

##  License
This project is licensed under the MIT License.
