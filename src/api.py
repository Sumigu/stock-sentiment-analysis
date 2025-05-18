from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import glob
from typing import List, Dict, Any
import sys
import time

# 현재 디렉토리를 모듈 검색 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 감성 분석기 임포트
from sentiment_analyzer import SentimentAnalyzer

app = FastAPI(title="주식 뉴스 감성 분석 API")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 폴더 경로 (절대 경로로 지정)
DATA_DIR = "/home/yyusu/capstone/data"
MODEL_PATH = "/home/yyusu/capstone/models/sentiment_model.pt"

# 감성 분석기 초기화
sentiment_analyzer = None

@app.on_event("startup")
async def startup_event():
    global sentiment_analyzer
    # 감성 분석기 초기화
    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else None
    sentiment_analyzer = SentimentAnalyzer(model_path)
    print("Sentiment analyzer initialized")

@app.get("/")
def read_root():
    return {"message": "주식 뉴스 감성 분석 API에 오신 것을 환영합니다!"}

@app.get("/companies")
def get_companies():
    """시가총액 상위 20개 기업 목록을 반환합니다."""
    # CSV 파일 목록 가져오기 (절대 경로 사용)
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_*_main_news_10pages.csv"))
    
    # 파일이 없으면 다른 위치도 시도
    if not csv_files:
        # 현재 디렉토리에서 시도
        csv_files = glob.glob("*_*_main_news_10pages.csv")
    
    if not csv_files:
        # 상위 디렉토리의 data 폴더에서 시도
        csv_files = glob.glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "*_*_main_news_10pages.csv"))
    
    print(f"Found {len(csv_files)} CSV files: {csv_files}")
    
    companies = []
    for file in csv_files:
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 2:
            code = parts[0]
            name = parts[1]
            companies.append({"code": code, "name": name})
    
    return companies

@app.get("/news/{company_code}")
def get_company_news(company_code: str):
    """특정 기업의 뉴스와 감성 분석 결과를 반환합니다."""
    # 해당 기업의 CSV 파일 찾기 (절대 경로 사용)
    csv_files = glob.glob(os.path.join(DATA_DIR, f"{company_code}_*_main_news_10pages.csv"))
    
    if not csv_files:
        # 현재 디렉토리에서 시도
        csv_files = glob.glob(f"{company_code}_*_main_news_10pages.csv")
    
    if not csv_files:
        # 상위 디렉토리의 data 폴더에서 시도
        csv_files = glob.glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", f"{company_code}_*_main_news_10pages.csv"))
    
    if not csv_files:
        raise HTTPException(status_code=404, detail=f"기업 코드 {company_code}에 대한 뉴스 데이터를 찾을 수 없습니다.")
    
    # 뉴스 데이터 로드
    news_df = pd.read_csv(csv_files[0])
    
    # 감성 분석
    start_time = time.time()
    news_df = sentiment_analyzer.analyze_dataframe(news_df)
    print(f"Sentiment analysis completed in {time.time() - start_time:.2f} seconds")
    
    # 결과 반환
    return news_df.to_dict(orient="records")

@app.get("/news")
def get_all_news():
    """모든 기업의 뉴스와 감성 분석 결과를 반환합니다."""
    # 모든 CSV 파일 찾기 (절대 경로 사용)
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_*_main_news_10pages.csv"))
    
    if not csv_files:
        # 현재 디렉토리에서 시도
        csv_files = glob.glob("*_*_main_news_10pages.csv")
    
    if not csv_files:
        # 상위 디렉토리의 data 폴더에서 시도
        csv_files = glob.glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "*_*_main_news_10pages.csv"))
    
    if not csv_files:
        raise HTTPException(status_code=404, detail="뉴스 데이터를 찾을 수 없습니다.")
    
    # 모든 뉴스 데이터 결합
    all_news = []
    for file in csv_files:
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 2:
            code = parts[0]
            name = parts[1]
            
            df = pd.read_csv(file)
            df['company_code'] = code
            df['company_name'] = name
            
            # 감성 분석
            df = sentiment_analyzer.analyze_dataframe(df)
            
            all_news.append(df)
    
    # 데이터프레임 결합
    if all_news:
        combined_df = pd.concat(all_news, ignore_index=True)
        return combined_df.to_dict(orient="records")
    else:
        return []

@app.get("/sentiment_summary")
def get_sentiment_summary():
    """기업별 감성 분석 요약을 반환합니다."""
    # 모든 CSV 파일 찾기 (절대 경로 사용)
    csv_files = glob.glob(os.path.join(DATA_DIR, "*_*_main_news_10pages.csv"))
    
    if not csv_files:
        # 현재 디렉토리에서 시도
        csv_files = glob.glob("*_*_main_news_10pages.csv")
    
    if not csv_files:
        # 상위 디렉토리의 data 폴더에서 시도
        csv_files = glob.glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "*_*_main_news_10pages.csv"))
    
    if not csv_files:
        raise HTTPException(status_code=404, detail="뉴스 데이터를 찾을 수 없습니다.")
    
    # 기업별 감성 분석 요약
    summary = []
    for file in csv_files:
        filename = os.path.basename(file)
        parts = filename.split('_')
        if len(parts) >= 2:
            code = parts[0]
            name = parts[1]
            
            df = pd.read_csv(file)
            df = sentiment_analyzer.analyze_dataframe(df)
            
            # 감성 비율 계산
            total = len(df)
            positive = len(df[df['sentiment'] == '긍정'])
            neutral = len(df[df['sentiment'] == '중립'])
            negative = len(df[df['sentiment'] == '부정'])
            
            summary.append({
                "company_code": code,
                "company_name": name,
                "total_news": total,
                "positive": positive,
                "neutral": neutral,
                "negative": negative,
                "positive_ratio": positive / total if total > 0 else 0,
                "neutral_ratio": neutral / total if total > 0 else 0,
                "negative_ratio": negative / total if total > 0 else 0
            })
    
    return summary