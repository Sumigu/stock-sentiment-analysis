import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os

# API 엔드포인트 설정
API_URL = "http://localhost:8000"

# 페이지 설정
st.set_page_config(
    page_title="주식 뉴스 감성 분석",
    page_icon="📈",
    layout="wide"
)

# 제목
st.title("📊 시가총액 상위 기업 뉴스 감성 분석")
st.markdown("네이버 증권 뉴스 제목을 분석하여 기업별 감성 지수를 시각화합니다.")

# 데이터 로드 함수
@st.cache_data(ttl=300)
def load_companies():
    try:
        response = requests.get(f"{API_URL}/companies")
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"API 서버에 연결할 수 없습니다: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_company_news(company_code):
    try:
        response = requests.get(f"{API_URL}/news/{company_code}")
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"뉴스 데이터를 불러오는데 실패했습니다: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def load_sentiment_summary():
    try:
        response = requests.get(f"{API_URL}/sentiment_summary")
        return pd.DataFrame(response.json())
    except Exception as e:
        st.error(f"감성 분석 요약을 불러오는데 실패했습니다: {e}")
        return pd.DataFrame()

# 기업 데이터 로드
with st.spinner("기업 데이터를 불러오는 중..."):
    companies_df = load_companies()

if not companies_df.empty:
    # 사이드바에 기업 선택 옵션 추가
    st.sidebar.header("기업 선택")
    
    # 기업 이름 컬럼 확인
    name_col = 'name' if 'name' in companies_df.columns else 'company_name'
    code_col = 'code' if 'code' in companies_df.columns else 'company_code'
    
    if name_col in companies_df.columns:
        selected_company = st.sidebar.selectbox(
            "분석할 기업을 선택하세요:",
            options=companies_df[name_col].tolist(),
            index=0
        )
        
        # 선택된 기업의 코드 가져오기
        if code_col in companies_df.columns:
            selected_code = companies_df[companies_df[name_col] == selected_company][code_col].iloc[0]
            
            # 탭 생성
            tab1, tab2 = st.tabs(["기업별 분석", "종합 분석"])
            
            with tab1:
                st.header(f"{selected_company} 뉴스 감성 분석")
                
                # 뉴스 데이터 로드
                with st.spinner("뉴스를 불러오는 중..."):
                    news_df = load_company_news(selected_code)
                
                if news_df.empty:
                    st.warning("뉴스 데이터가 없습니다.")
                else:
                    # 감성 분석 결과 시각화
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 감성 분포 파이 차트
                        if 'sentiment' in news_df.columns:
                            sentiment_counts = news_df['sentiment'].value_counts()
                            fig = px.pie(
                                values=sentiment_counts.values, 
                                names=sentiment_counts.index,
                                title="뉴스 감성 분포",
                                color=sentiment_counts.index,
                                color_discrete_map={'긍정': 'green', '중립': 'gray', '부정': 'red'}
                            )
                            st.plotly_chart(fig)
                        else:
                            st.warning("감성 분석 결과가 없습니다.")
                    
                    with col2:
                        # 감성 확률 분포 히스토그램
                        if 'sentiment_prob' in news_df.columns and 'sentiment' in news_df.columns:
                            fig = px.histogram(
                                news_df, 
                                x='sentiment_prob',
                                color='sentiment',
                                title="감성 확률 분포",
                                labels={'sentiment_prob': '확률', 'count': '뉴스 수'},
                                color_discrete_map={'긍정': 'green', '중립': 'gray', '부정': 'red'}
                            )
                            st.plotly_chart(fig)
                        else:
                            st.warning("감성 확률 데이터가 없습니다.")
                    
                    # 뉴스 목록 표시
                    st.subheader("최근 뉴스 목록")
                    
                    # 감성별 색상 설정
                    def highlight_sentiment(val):
                        if val == '긍정':
                            return 'background-color: rgba(0, 255, 0, 0.2)'
                        elif val == '부정':
                            return 'background-color: rgba(255, 0, 0, 0.2)'
                        return ''
                    
                    # 표시할 컬럼 선택
                    display_cols = ['제목', 'sentiment', 'sentiment_prob', '날짜', '언론사']
                    display_cols = [col for col in display_cols if col in news_df.columns]
                    
                    # 스타일이 적용된 데이터프레임 표시
                    if 'sentiment' in display_cols:
                        styled_df = news_df[display_cols].style.applymap(
                            highlight_sentiment, subset=['sentiment']
                        )
                        st.dataframe(styled_df, use_container_width=True)
                    else:
                        st.dataframe(news_df[display_cols], use_container_width=True)
            
            with tab2:
                st.header("전체 기업 뉴스 감성 분석")
                
                # 감성 분석 요약 로드
                with st.spinner("감성 분석 요약을 불러오는 중..."):
                    summary_df = load_sentiment_summary()
                
                if summary_df.empty:
                    st.warning("감성 분석 요약 데이터가 없습니다.")
                else:
                    # 기업별 감성 비율 시각화
                    st.subheader("기업별 뉴스 감성 비율")
                    
                    # 데이터 준비
                    plot_df = summary_df.copy()
                    plot_df['긍정'] = plot_df['positive_ratio'] * 100
                    plot_df['중립'] = plot_df['neutral_ratio'] * 100
                    plot_df['부정'] = plot_df['negative_ratio'] * 100
                    
                    # 긍정 비율 기준으로 정렬
                    plot_df = plot_df.sort_values('긍정', ascending=False)
                    
                    # 기업명 컬럼 확인
                    company_name_col = 'company_name' if 'company_name' in plot_df.columns else 'name'
                    
                    if company_name_col in plot_df.columns:
                        # 막대 그래프 생성
                        fig = px.bar(
                            plot_df,
                            x=company_name_col,
                            y=['긍정', '중립', '부정'],
                            title="기업별 뉴스 감성 비율",
                            labels={'value': '비율 (%)', company_name_col: '기업명', 'variable': '감성'},
                            color_discrete_map={'긍정': 'green', '중립': 'gray', '부정': 'red'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 감성 지수 계산 (긍정 - 부정)
                        st.subheader("기업별 감성 지수 (긍정 - 부정)")
                        plot_df['감성지수'] = plot_df['긍정'] - plot_df['부정']
                        plot_df = plot_df.sort_values('감성지수', ascending=False)
                        
                        # 수평 막대 그래프 생성
                        fig = px.bar(
                            plot_df,
                            y=company_name_col,
                            x='감성지수',
                            title="기업별 감성 지수 (긍정 - 부정)",
                            labels={'감성지수': '감성 지수', company_name_col: '기업명'},
                            color='감성지수',
                            color_continuous_scale=['red', 'yellow', 'green']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 뉴스 수와 감성 관계
                        st.subheader("뉴스 수와 감성 관계")
                        fig = px.scatter(
                            plot_df,
                            x='total_news',
                            y='감성지수',
                            size='total_news',
                            color='감성지수',
                            hover_name=company_name_col,
                            title="뉴스 수와 감성 지수의 관계",
                            labels={'total_news': '총 뉴스 수', '감성지수': '감성 지수'},
                            color_continuous_scale=['red', 'yellow', 'green']
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 데이터 테이블 표시
                        st.subheader("기업별 감성 분석 요약")
                        display_cols = [company_name_col, 'total_news', 'positive', 'neutral', 'negative', '긍정', '중립', '부정', '감성지수']
                        display_cols = [col for col in display_cols if col in plot_df.columns]
                        st.dataframe(plot_df[display_cols].round(2), use_container_width=True)
                    else:
                        st.error(f"기업명 컬럼({company_name_col})을 찾을 수 없습니다.")
        else:
            st.error(f"기업 코드 컬럼({code_col})을 찾을 수 없습니다.")
    else:
        st.error(f"기업 이름 컬럼({name_col})을 찾을 수 없습니다.")
else:
    st.error("기업 데이터를 불러올 수 없습니다. API 서버가 실행 중인지 확인해주세요.")

# 푸터
st.markdown("---")
st.markdown("© 2025 주식 뉴스 감성 분석 프로젝트")