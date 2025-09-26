import pickle
from utils import load_data

MODEL_PATH = "model/model.pkl"

def train_model():
    data = load_data()
    # 여기는 단순 예시: 완료율 평균을 "모델"처럼 저장
    avg_completion = data["completed"].mean()

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(avg_completion, f)
    print(f"모델이 저장되었습니다: {MODEL_PATH}")

if __name__ == "__main__":
    train_model()
