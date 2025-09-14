from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Quest Tracker API 준비 완료!"}
