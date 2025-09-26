from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from .utils import plot_user_completed, plot_quest_completion_rate

app = FastAPI(title="AI Quest Tracker API")

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <html>
        <head><title>AI Quest Tracker</title></head>
        <body>
            <h1>AI Quest Tracker API 준비 완료!</h1>
            <p><a href="/plot/user"><button>사용자별 완료 퀘스트 그래프</button></a></p>
            <p><a href="/plot/quest"><button>퀘스트별 완료율 그래프</button></a></p>
        </body>
    </html>
    """

@app.get("/plot/user", response_class=HTMLResponse)
def user_plot():
    img_base64 = plot_user_completed()
    return f'<html><body><h2>사용자별 완료 퀘스트</h2><img src="data:image/png;base64,{img_base64}"/></body></html>'

@app.get("/plot/quest", response_class=HTMLResponse)
def quest_plot():
    img_base64 = plot_quest_completion_rate()
    return f'<html><body><h2>퀘스트별 완료율</h2><img src="data:image/png;base64,{img_base64}"/></body></html>'



# uvicorn src.main:app --reload