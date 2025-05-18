
```
wsl  환경에서 만듬.
웹 실행 코드 탭 두개로 실행.

추가할만한 기능능
장마감시의 뉴스 자동 크롤링 기능, 현재 주가도 함께 보여주기
```
```bash
python -m uvicorn src.api:app --reload
python -m streamlit run src/app.py
```