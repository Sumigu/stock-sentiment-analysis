```bash
pip install requests pandas beautifulsoup4 lxml
```
```
크롤링 시총상위 20위 이상으로 20개정도?
```
```bash
python -m uvicorn src.api:app --reload
python -m streamlit run src/app.py
```