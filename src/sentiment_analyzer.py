import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class SentimentAnalyzer:
    def __init__(self, model_path=None):
        # GPU 사용 가능 여부 확인
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        try:
            # Hugging Face 모델 사용 (KR-FinBert)
            print("Loading KR-FinBert model...")
            self.tokenizer = AutoTokenizer.from_pretrained("snunlp/KR-FinBert-SC")
            self.model = AutoModelForSequenceClassification.from_pretrained("snunlp/KR-FinBert-SC").to(self.device)
            self.model.eval()  # 평가 모드로 설정
            print("Successfully loaded pre-trained KR-FinBert model")
            self.model_type = "finbert"
        except Exception as e:
            print(f"Error loading KR-FinBert: {e}")
            print("Falling back to rule-based analysis...")
            self.model_type = "rule-based"
    
    def predict(self, text, max_len=64):
        """
        텍스트의 감성을 예측합니다.
        
        Args:
            text (str): 분석할 텍스트
            max_len (int): 최대 토큰 길이
            
        Returns:
            dict: 감성 분석 결과 (label, sentiment, probability)
        """
        # 금융 특화 키워드 체크 (강한 신호가 있는 경우 우선 적용)
        strong_positive = ['급등', '신기록', '사상 최대', '대폭 상승', '호실적', '급증', '돌파', '최고치', '급반등']
        strong_negative = ['급락', '폭락', '대폭 하락', '적자 전환', '부진', '최저치', '위기', '충격', '폭락세']
        
        # 강한 키워드가 있으면 해당 감성으로 바로 결정
        for keyword in strong_positive:
            if keyword in text:
                return {
                    'label': 2,
                    'sentiment': '긍정',
                    'probability': 0.95,
                    'all_probabilities': {'부정': 0.02, '중립': 0.03, '긍정': 0.95}
                }
        
        for keyword in strong_negative:
            if keyword in text:
                return {
                    'label': 0,
                    'sentiment': '부정',
                    'probability': 0.95,
                    'all_probabilities': {'부정': 0.95, '중립': 0.03, '긍정': 0.02}
                }
        
        if self.model_type == "finbert":
            try:
                # FinBert 모델을 사용한 감성 분석
                inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
                
                # 원래 확률 저장
                original_probabilities = probabilities.copy()
                
                # 중립 임계값 조정 (중립 비율 줄이기)
                neg_prob, neu_prob, pos_prob = probabilities
                
                # 중립 감성 비율 줄이기 위한 임계값 조정
                # 중립 확률이 0.5 이하면 긍정/부정 중 더 높은 쪽으로 기울이기
                if neu_prob <= 0.5:
                    if pos_prob > neg_prob:
                        sentiment_label = 2  # 긍정
                    else:
                        sentiment_label = 0  # 부정
                # 중립 확률이 0.5~0.7 사이인 경우에도 긍정/부정 확률이 0.3 이상이면 해당 방향으로 기울이기
                elif neu_prob > 0.5 and neu_prob < 0.7:
                    if pos_prob > neg_prob and pos_prob >= 0.3:
                        sentiment_label = 2  # 긍정
                        # 확률 재조정
                        probabilities = np.array([neg_prob * 0.6, neu_prob * 0.7, pos_prob * 1.4])
                        probabilities = probabilities / np.sum(probabilities)  # 정규화
                    elif neg_prob > pos_prob and neg_prob >= 0.3:
                        sentiment_label = 0  # 부정
                        # 확률 재조정
                        probabilities = np.array([neg_prob * 1.4, neu_prob * 0.7, pos_prob * 0.6])
                        probabilities = probabilities / np.sum(probabilities)  # 정규화
                    else:
                        sentiment_label = 1  # 중립
                else:
                    sentiment_label = np.argmax(probabilities)
                
                # 추가 키워드 기반 보정
                positive_keywords = ['상승', '증가', '성장', '개선', '확대', '호조', '순항', '강세', '반등', '회복', '흑자']
                negative_keywords = ['하락', '감소', '축소', '악화', '부진', '위축', '우려', '약세', '후퇴', '쇼크', '적자']
                
                # 키워드 매칭 수 확인
                pos_matches = sum(1 for word in positive_keywords if word in text)
                neg_matches = sum(1 for word in negative_keywords if word in text)
                
                # 키워드 매칭 결과에 따른 추가 보정
                if sentiment_label == 1:  # 중립으로 분류된 경우만 추가 보정
                    if pos_matches > neg_matches and pos_matches >= 1:
                        sentiment_label = 2  # 긍정으로 변경
                        # 확률 재조정 (키워드 매칭 수에 따라 확률 증가)
                        confidence = min(0.6 + pos_matches * 0.1, 0.9)
                        probabilities = np.array([0.1, 1.0 - confidence, confidence])
                    elif neg_matches > pos_matches and neg_matches >= 1:
                        sentiment_label = 0  # 부정으로 변경
                        # 확률 재조정 (키워드 매칭 수에 따라 확률 증가)
                        confidence = min(0.6 + neg_matches * 0.1, 0.9)
                        probabilities = np.array([confidence, 1.0 - confidence, 0.1])
                
                # 특수 문자 기반 추가 보정
                if sentiment_label == 1:  # 여전히 중립인 경우
                    if '↑' in text or '⇧' in text or '↗' in text:
                        sentiment_label = 2  # 긍정으로 변경
                        probabilities = np.array([0.1, 0.3, 0.6])
                    elif '↓' in text or '⇩' in text or '↘' in text:
                        sentiment_label = 0  # 부정으로 변경
                        probabilities = np.array([0.6, 0.3, 0.1])
                
                # 부정어 처리
                negation_words = ['않', '안', '못', '없', '불', '노', '거부']
                has_negation = any(word in text for word in negation_words)
                
                # 부정어가 있고 긍정/부정으로 분류된 경우, 감성 반전 고려
                if has_negation and (sentiment_label == 0 or sentiment_label == 2):
                    # 원래 확률과 현재 확률의 차이가 크지 않은 경우에만 반전 적용
                    if abs(probabilities[sentiment_label] - original_probabilities[sentiment_label]) < 0.3:
                        # 감성 반전 (긍정 <-> 부정)
                        if sentiment_label == 0:
                            sentiment_label = 2
                            probabilities = np.array([0.2, 0.3, 0.5])
                        else:
                            sentiment_label = 0
                            probabilities = np.array([0.5, 0.3, 0.2])
                
                sentiment_prob = probabilities[sentiment_label]
                
                # 감성 레이블 매핑 (0: 부정, 1: 중립, 2: 긍정)
                sentiment_map = {0: '부정', 1: '중립', 2: '긍정'}
                
                return {
                    'label': int(sentiment_label),
                    'sentiment': sentiment_map[sentiment_label],
                    'probability': float(sentiment_prob),
                    'all_probabilities': {sentiment_map[i]: float(p) for i, p in enumerate(probabilities)}
                }
            except Exception as e:
                print(f"Error in FinBert prediction: {e}")
                # 오류 발생 시 규칙 기반 분석으로 대체
                return self._rule_based_predict(text)
        else:
            # 규칙 기반 감성 분석 사용
            return self._rule_based_predict(text)
    
    def _rule_based_predict(self, text):
        """
        규칙 기반 감성 분석을 수행합니다.
        """
        # 긍정/부정 키워드 기반 간단한 감성 분석
        positive_words = ['상승', '호조', '증가', '성장', '개선', '좋은', '성공', '흑자', '최대', '신기록', 
                          '급등', '돌파', '강세', '호실적', '기대', '순항', '호황', '반등', '회복']
        negative_words = ['하락', '부진', '감소', '하향', '악화', '나쁜', '실패', '적자', '최저', '위기', 
                          '급락', '추락', '약세', '저조', '우려', '충격', '불황', '후퇴', '쇼크']
        
        pos_count = sum(1 for word in positive_words if word in text)
        neg_count = sum(1 for word in negative_words if word in text)
        
        # 특수 문자 고려
        if '↑' in text or '⇧' in text or '↗' in text:
            pos_count += 1
        if '↓' in text or '⇩' in text or '↘' in text:
            neg_count += 1
        
        if pos_count > neg_count:
            sentiment = 2  # 긍정
            prob = 0.7 + (pos_count - neg_count) * 0.05
            prob = min(prob, 0.95)
        elif neg_count > pos_count:
            sentiment = 0  # 부정
            prob = 0.7 + (neg_count - pos_count) * 0.05
            prob = min(prob, 0.95)
        else:
            # 키워드가 없거나 동일한 경우 추가 분석
            if '?' in text:
                sentiment = 0  # 부정적 질문으로 간주
                prob = 0.6
            elif '!' in text:
                sentiment = 2  # 긍정적 강조로 간주
                prob = 0.6
            else:
                # 중립 비율 줄이기 위해 문장 길이 기반 판단
                if len(text) < 15:  # 짧은 문장은 긍정으로 기울이기
                    sentiment = 2
                    prob = 0.55
                else:
                    sentiment = 1  # 중립
                    prob = 0.8
        
        # 부정어 처리
        negation_words = ['않', '안', '못', '없', '불', '노', '거부']
        if any(word in text for word in negation_words) and (sentiment == 0 or sentiment == 2):
            # 감성 반전 (긍정 <-> 부정)
            sentiment = 2 if sentiment == 0 else 0
            prob = max(0.55, prob - 0.1)  # 확률 약간 감소
        
        sentiment_map = {0: '부정', 1: '중립', 2: '긍정'}
        return {
            'label': sentiment,
            'sentiment': sentiment_map[sentiment],
            'probability': float(prob),
            'all_probabilities': {
                '부정': float(0.8 if sentiment == 0 else 0.1),
                '중립': float(0.8 if sentiment == 1 else 0.1),
                '긍정': float(0.8 if sentiment == 2 else 0.1)
            }
        }
    
    def analyze_dataframe(self, df, text_column='제목'):
        """
        데이터프레임의 텍스트 컬럼을 분석합니다.
        
        Args:
            df (DataFrame): 분석할 데이터프레임
            text_column (str): 분석할 텍스트가 있는 컬럼명
            
        Returns:
            DataFrame: 감성 분석 결과가 추가된 데이터프레임
        """
        results = []
        
        for i, text in enumerate(df[text_column]):
            try:
                if i % 10 == 0:  # 진행 상황 로깅 (10개마다)
                    print(f"Analyzing text {i+1}/{len(df)}: {text[:30]}...")
                result = self.predict(text)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing text: {text}")
                print(f"Error: {e}")
                # 오류 발생 시 규칙 기반 분석 사용
                results.append(self._rule_based_predict(text))
        
        # 결과를 데이터프레임에 추가
        df['sentiment_label'] = [r['label'] for r in results]
        df['sentiment'] = [r['sentiment'] for r in results]
        df['sentiment_prob'] = [r['probability'] for r in results]
        
        # 감성 분포 출력
        sentiment_counts = df['sentiment'].value_counts()
        total = len(df)
        print("\n감성 분석 결과 분포:")
        for sentiment, count in sentiment_counts.items():
            percentage = (count / total) * 100
            print(f"{sentiment}: {count}개 ({percentage:.1f}%)")
        
        return df