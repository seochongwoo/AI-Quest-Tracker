![header](https://capsule-render.vercel.app/api?type=waving&color=0:02071e,80:030928&height=300&section=header&text=AI%20Quest%20Tracker&fontSize=70&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Track%20your%20habits%20and%20get%20AI-powered%20feedback!&descAlignY=51&descAlign=50)

# 🌟 AI Quest Tracker

- **AI Quest Tracker**는 오픈소스 habit tracker에서 영감을 받아, **머신러닝을 활용해 퀘스트(습관) 성공 확률을 예측**하고, **맞춤형 퀘스트를 추천**하며, 간단한 **AI 피드백**을 제공하는 프로젝트입니다. 
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

# 3. 퀘스트 완료 혹은 삭제 후 train.py 다시 실행!
python -m src.train
```

- 실행 후: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 접속하면 Swagger UI에서 API 확인 가능 ✅

---

##  Features

###  샘플 데이터
`seed.py`로 더미데이터 생성
- 데이터베이스에 저장되는 레코드 예시
```csv
user_id, name, category, duration, difficulty, completed, ...
1,"퀘스트_1_health","health",7,3,1
1,"퀘스트_2_study","study",3,5,0
2,"퀘스트_1_exercise","exercise",10,2,1
```

###  모델 학습
`src/train.py`  
- 랜덤 포레스트 기반 분류 모델 학습 및 보정(CalibratedClassifierCV) 적용
- 퀘스트 이름(name)을 SentenceTransformer로 임베딩하여 모델 피처에 사용
-  사용자별 완료율(user_success_rate), 기간(days), 난이도(difficulty) 등을 피처로 활용하여 성공 여부(completed) 예측
- 학습된 모델과 임베딩 객체를 포함한 튜플을 model/model.pkl로 저장
```python
# train.py에서 모델과 임베더 객체를 함께 저장합니다.
dump((model, embedder), MODEL_PATH)
```


###  API 실행
`src/main.py`  
- FastAPI 서버 구동 시 model.py를 통해 model/model.pkl에서 학습된 모델을 로드합니다.
- /quests/list: 사용자별 퀘스트 목록 및 상태 토글/삭제 기능 제공
- /recommend/result: 사용자의 로그인 ID를 기반으로 맞춤형 성공률 예측
- /quests/ (POST): 퀘스트 생성 시 AI 모델을 통해 성공률을 예측하여 DB에 저장

### 예측 결과
```python
# model.py의 predict_success_rate 함수 호출
predicted_rate = predict_success_rate(
    user_id=user_id,
    quest_name="매일 30분 운동",  # 예측할 퀘스트 이름
    duration=7,                  # 예상 기간 (일)
    difficulty=4                 # 난이도 (1~5)
)
# 반환 값은 0.0 ~ 1.0 사이의 성공 확률
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
- **ML**: scikit-learn (RandomForest, CalibratedClassifierCV), joblib, SentenceTransformer 
- **DB**: SQLAlchemy
- **Visualization**: matplotlib, Plotly, HTML/CSS Gauge Bar

---

##  Reference
- [Habitica](https://habitica.com/)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)  
- [FastAPI](https://fastapi.tiangolo.com/)  
- Sentence Transformers (텍스트 임베딩)
---

##  License
This project is licensed under the MIT License.
