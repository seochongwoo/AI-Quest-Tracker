![header](https://capsule-render.vercel.app/api?type=waving&color=0:02071e,80:030928&height=300&section=header&text=AI%20Quest%20Tracker&fontSize=70&fontColor=fff&animation=fadeIn&fontAlignY=38&desc=Track%20your%20habits%20and%20get%20AI-powered%20feedback!&descAlignY=51&descAlign=50)

# 🌟 AI Quest Tracker

- **AI Quest Tracker**는 오픈소스 habit tracker에서 영감을 받아, **머신러닝을 활용해 퀘스트(습관) 성공 확률을 예측**하고, **맞춤형 퀘스트를 추천**하며, 간단한 **AI 피드백**을 제공하는 프로젝트입니다. 
- 사용자는 자신이 원하는 퀘스트를 추가하고, 실행 결과를 기록하며, AI로부터 동기부여와 피드백을 받을 수 있습니다.
- 사용기간이 길어지고 더 많은 데이터를 쌓을 수록 더 정확한 예측 결과를 내놓습니다.
- [Habitica](https://habitica.com/)와 같은 habit tracker에서 영감을 받았으며, **데이터 기반 개인화**를 주요 목표로 합니다.

---

##  Table of Contents
1. [Getting Started](#getting-started)  
2. [Features](#features)  
   1. [샘플 데이터](#샘플-데이터)  
   2. [모델 학습](#모델-학습)
   3. [주요 엔드포인트](#주요-엔드포인트)  
   4. [API 실행](#api-실행)  
   5. [예측 결과](#예측-결과)  
3. [Demo](#demo)  
4. [API Docs](#api-docs)  
5. [기술 스택](#기술-스택)  
6. [Reference](#reference)  
7. [License](#license)  

---

##  Getting Started

### API key 설정 방법
1. [Google AI Studio](https://aistudio.google.com/)에서 Gemini API Key를 발급받습니다.
2. 프로젝트 루트 경로에 `.env` 파일을 생성하고 아래 내용을 추가합니다.
3. GEMINI_API_KEY="YOUR_API_KEY"

### Requirements
- Python 3.9+
- pip

### Installation
```bash
# 저장소 클론
git clone https://github.com/seochongwoo/AI-Quest-Tracker.git
cd AI-Quest-Tracker
# 패키지 설치
pip install -r requirements.txt
```

### Running
```bash
# FastAPI 실행 (서버 실행)
uvicorn src.main:app --reload
```

- 실행 후: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) 접속하면 Swagger UI에서 API 확인 가능 ✅
- 주의: 초기에 모델의 예측 결과와 AI 코치의 조언이 서로 다를 수 있습니다!
---

##  Features

###  샘플 데이터
`seed.py`로 더미데이터 
- AI를 통한 현실적인 더미데이터 생성
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

### 주요 엔드포인트
- /quests/{quest_id}: 퀘스트 상세 조회 및 업데이트 (예: 상태 토글/삭제)
- /plot/dashboard: 사용자별 퀘스트 시각화 제공
- /recommend/result: 사용자의 로그인 ID를 기반으로 Gemini를 통한 맞춤형 성공률 예측 및 조언
- /calendar: 사용자의 성취를 달력 형태로 제공

### 예측 결과
```python
# model.py의 predict_success_rate 함수 호출
predicted_rate = predict_success_rate(
    user_id=user_id,
    quest_name="매일 30분 운동",      # 예측할 퀘스트 이름
    duration=7,                      # 예상 기간 (일)
    difficulty=4                     # 난이도 (1~5)
    category: Optional[str] = None,  # 카테고리 설정
    motivation: Optional[str] = None # 동기부여 문구
)-> float
# 반환 값은 0.0 ~ 1.0 사이의 성공 확률
```

---

##  Demo

로그인 화면
<img width="1031" height="490" alt="image" src="https://github.com/user-attachments/assets/f8b923a5-6b08-4033-a961-2fca7ba22b80" />

메인 페이지
<img width="1039" height="848" alt="image" src="https://github.com/user-attachments/assets/22f57859-7d5d-46ef-9667-8df56889c695" />


예시:
- /quests 예시
  메인  페이지
  <img width="1037" height="1074" alt="image" src="https://github.com/user-attachments/assets/ccf9a969-1d88-4aea-b589-28ef9edee08c" />
  특정 퀘스트의 성공을 여러번 하면 성공률이 높아짐을 확인할 수 있습니다.
  <img width="799" height="937" alt="image" src="https://github.com/user-attachments/assets/097afa2a-52c5-4da8-be3e-22050adc4c7d" />
  목표 일수에 따라 퀘스트를 얼만큼 수행했는지 체크할 수 있습니다.
  <img width="1016" height="1102" alt="image" src="https://github.com/user-attachments/assets/bc50755c-0aa8-4a6a-bbcc-25821f02c2f6" />
  
- /plot/dashboard 예시
  메인 페이지
  <img width="1038" height="809" alt="image" src="https://github.com/user-attachments/assets/efbc1bbb-2f9a-4c5a-86f5-369b039b23e9" />
  개인 퀘스트 현황
  <img width="1022" height="1101" alt="image" src="https://github.com/user-attachments/assets/295807b4-2548-483a-9e23-a51902c0150f" />
  카테고리별 성공률
  <img width="1020" height="700" alt="image" src="https://github.com/user-attachments/assets/5de64fb0-8902-4b06-9826-0d5796e5466a" />
  성장 추세 그래프
  <img width="1021" height="747" alt="image" src="https://github.com/user-attachments/assets/ca1d809e-1d39-41e5-a0d4-fe109db46d3d" />
  집중 분야 분석
  <img width="1019" height="1100" alt="image" src="https://github.com/user-attachments/assets/c7899c40-ed31-41e9-b688-a5d66dd002b4" />
  test1 계정으로는 운동 카테고리를 많이 수행했기 때문에 exercise의 성공률이 다른 카테고리 보다 높은 것을 확인할 수 있습니다.
  
- /recommend 예시
  메인 페이지
  <img width="1022" height="1116" alt="image" src="https://github.com/user-attachments/assets/904eb9c9-63fb-4f27-b574-72ce7c6d8d71" />
  gemini API를 활용한 예측 화면
  <img width="1019" height="1113" alt="image" src="https://github.com/user-attachments/assets/23339f51-9968-450b-9703-552b54f1ec96" />
  

- /calendar 예시
  메인 페이지
  <img width="1033" height="991" alt="image" src="https://github.com/user-attachments/assets/3dc2f898-8e35-42c9-99d5-7d35223f4a14" />


---

##  API Docs

---

##  기술 스택
- **Backend**: Python, FastAPI  
- **ML**: scikit-learn (RandomForest, CalibratedClassifierCV), joblib, SentenceTransformer 
- **DB**: SQLAlchemy (with SQLite backend)
- **Visualization**: matplotlib, Plotly, HTML/CSS Gauge Bar

---

##  Reference
- [Habitica](https://habitica.com/)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)  
- [FastAPI](https://fastapi.tiangolo.com/)  
- Sentence Transformers (텍스트 임베딩)
- Gemini API 
---

##  License
This project is licensed under the MIT License.
