from fastapi import FastAPI
import requests
import torch
from transformers import pipeline

app = FastAPI()

# Load mô hình phân tích cảm xúc (có thể thay thế bằng BERT, LSTM, VADER, v.v.)
sentiment_analysis = pipeline("sentiment-analysis")

# Hàm lấy bình luận từ Facebook
def get_facebook_comments(post_id, access_token):
    url = f"https://graph.facebook.com/v12.0/{post_id}/comments?access_token={access_token}"
    response = requests.get(url)
    data = response.json()
    comments = [c["message"] for c in data.get("data", [])]
    return comments

# API nhận ID bài viết Facebook và trả về kết quả phân tích
@app.get("/analyze")
def analyze_comments(post_id: str, access_token: str):
    comments = get_facebook_comments(post_id, access_token)
    results = [{"comment": c, "sentiment": sentiment_analysis(c)[0]} for c in comments]
    return {"data": results}
