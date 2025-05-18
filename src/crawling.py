import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

# 종목코드와 기업명 리스트
stocks = [
    ('005930', '삼성전자'),
    ('000660', 'SK하이닉스'),
    ('373220', 'LG에너지솔루션'),
    ('207940', '삼성바이오로직스'),
    ('012450', '한화에어로스페이스'),
    ('005380', '현대차'),
    ('005935', '삼성전자우'),
    ('329180', 'HD현대중공업'),
    ('105560', 'KB금융'),
    ('068270', '셀트리온'),
    ('000270', '기아'),
    ('035420', 'NAVER'),
    ('055550', '신한지주'),
    ('042660', '한화오션'),
    ('012330', '현대모비스'),
    ('138040', '메리츠금융지주'),
    ('005490', 'POSCO홀딩스'),
    ('028260', '삼성물산'),
    ('009540', 'HD한국조선해양'),
    ('196170', '알테오젠')
]

for code, name in stocks:
    print(f"▶ {name}({code}) 뉴스 수집 중...")
    HEADERS = {
        'Referer': f'https://finance.naver.com/item/news.naver?code={code}',
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    news_df = pd.DataFrame(columns=['날짜', '제목', '언론사', 'URL'])

    for page in range(1, 11):
        NEWS_URL = f"https://finance.naver.com/item/news_news.naver?code={code}&page={page}"
        response = requests.get(NEWS_URL, headers=HEADERS)
        soup = BeautifulSoup(response.text, 'html.parser')

        # 연관기사 목록 제거용 제목 수집
        exclude_titles = set()
        for table in soup.find_all('table', {'class': 'type5'}):
            caption = table.find('caption')
            if caption and '연관기사 목록' in caption.text:
                for td in table.find_all('td', {'class': 'title'}):
                    a_tag = td.find('a')
                    if a_tag:
                        exclude_titles.add(a_tag.text.strip())

        # 뉴스 테이블에서 유효한 뉴스만 추출
        news_table = soup.find('table', {'class': 'type5'})
        if not news_table:
            continue
        news_rows = news_table.find_all('tr')
        for row in news_rows:
            tr_class = row.get('class')
            if (tr_class is None) or \
               ('first' in tr_class) or \
               ('last' in tr_class) or \
               ('relation_tit' in tr_class):
                title_cell = row.find('td', {'class': 'title'})
                if not title_cell:
                    continue
                title_tag = title_cell.find('a')
                if not title_tag:
                    continue
                title = title_tag.text.strip()
                if title in exclude_titles:
                    continue
                url = 'https://finance.naver.com' + title_tag['href']
                date_cell = row.find('td', {'class': 'date'})
                date = date_cell.text.strip() if date_cell else ''
                press_cell = row.find('td', {'class': 'info'})
                press = press_cell.text.strip() if press_cell else ''
                news_df = news_df._append({
                    '날짜': date,
                    '제목': title,
                    '언론사': press,
                    'URL': url
                }, ignore_index=True)
        time.sleep(1)

    print(f" - 수집 완료: {len(news_df)}개 기사")
    filename = f"{code}_{name}_main_news_10pages.csv"
    news_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f" - 저장 완료: {filename}\n")

print("✅ 모든 종목 뉴스 수집이 완료되었습니다.")