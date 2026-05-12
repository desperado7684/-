import yfinance as yf
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# 검색과 분석을 위한 핵심 유니버스 세팅
ASSETS = {
    # 한국 핵심 종목
    "^KS11": {"name": "코스피 종합지수", "type": "국내지수"},
    "005930.KS": {"name": "삼성전자", "type": "국내주식"},
    "000660.KS": {"name": "SK하이닉스", "type": "국내주식"},
    "005380.KS": {"name": "현대차", "type": "국내주식"},
    "086790.KS": {"name": "하나금융지주", "type": "국내주식"},
    "035420.KS": {"name": "NAVER", "type": "국내주식"},
    
    # 글로벌 빅테크 및 지수
    "^SOX": {"name": "필라델피아 반도체", "type": "해외지수"},
    "^IXIC": {"name": "나스닥 종합", "type": "해외지수"},
    "MSFT": {"name": "마이크로소프트", "type": "해외주식"},
    "META": {"name": "메타 (페이스북)", "type": "해외주식"},
    "NVDA": {"name": "엔비디아", "type": "해외주식"},
    "TSLA": {"name": "테슬라", "type": "해외주식"},
    "AAPL": {"name": "애플", "type": "해외주식"},
    
    # 거시경제 & 리스크 지표
    "KRW=X": {"name": "원/달러 환율", "type": "거시/환율"},
    "^TNX": {"name": "미 국채 10년물", "type": "거시/금리"},
    "^VIX": {"name": "VIX 공포지수", "type": "리스크"},
    "BTC-USD": {"name": "비트코인", "type": "가상자산"}
}

def analyze_and_comment(rsi, macd, bb, change, asset_type):
    """지표를 분석하여 토스 스타일의 친절한 해설 생성"""
    if asset_type in ["거시/환율", "거시/금리", "리스크"]:
        if change > 1.0: return "시장 불안감이 커지고 있어요. 보수적인 투자가 필요한 시점이에요. 🚨"
        elif change < -1.0: return "불안 요소가 줄어들고 있어요. 증시에는 긍정적인 신호예요. 🌤️"
        else: return "큰 변동 없이 안정적인 흐름을 보이고 있어요. ⚖️"

    if rsi >= 70 or bb >= 95:
        return "단기적으로 많이 올랐어요. 과열 상태일 수 있으니 추격 매수는 조심하세요. 🔥"
    elif rsi <= 30 or bb <= 5:
        return "최근 많이 떨어졌어요. 기술적 반등(저점 매수)을 노려볼 만한 구간이에요. 💡"
    elif macd > 0 and change > 0:
        return "상승 추세를 탔어요! 모멘텀이 좋아서 더 오를 가능성이 있어요. 🚀"
    elif macd < 0 and change < 0:
        return "하락 추세가 이어지고 있어요. 바닥을 다질 때까지 조금 더 지켜보세요. 📉"
    else:
        return "뚜렷한 방향성 없이 에너지를 모으는 중이에요. 분할 매수하기 좋은 시기예요. 🧩"

def collect():
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    market_data = []

    for sym, info in ASSETS.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="3mo")
            
            if len(hist) > 30:
                close = hist['Close']
                cur = close.iloc[-1]
                prev = close.iloc[-2]
                chg_pct = ((cur - prev) / prev) * 100
                
                # 1. RSI 계산
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                current_rsi = round(rsi.iloc[-1], 1) if not np.isnan(rsi.iloc[-1]) else 50
                
                # 2. MACD 계산
                exp1 = close.ewm(span=12, adjust=False).mean()
                exp2 = close.ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                macd_hist = round((macd - signal).iloc[-1], 2)

                # 3. Bollinger Bands %B 계산
                sma = close.rolling(window=20).mean()
                std = close.rolling(window=20).std()
                upper_bb = sma + (std * 2)
                lower_bb = sma - (std * 2)
                bb_pct = round(((cur - lower_bb.iloc[-1]) / (upper_bb.iloc[-1] - lower_bb.iloc[-1])) * 100, 1)
                
                # 해설 생성
                comment = analyze_and_comment(current_rsi, macd_hist, bb_pct, chg_pct, info["type"])
                
                market_data.append({
                    "symbol": sym,
                    "name": info["name"],
                    "type": info["type"],
                    "price": round(cur, 2) if cur > 100 else round(cur, 4),
                    "change": round(chg_pct, 2),
                    "tech": {"rsi": current_rsi, "macd": macd_hist, "bb": bb_pct},
                    "comment": comment
                })
        except Exception as e:
            print(f"Error {sym}: {e}")

    # 시장 온도 계산 (매크로 제외한 자산 중 상승 비율)
    risk_assets = [d for d in market_data if d["type"] not in ["거시/환율", "거시/금리", "리스크"]]
    bull_count = sum(1 for d in risk_assets if d["change"] > 0)
    market_temp = round((bull_count / len(risk_assets)) * 100) if risk_assets else 50

    output = {
        "kst": now.strftime("%Y년 %m월 %d일 %H:%M 기준"),
        "market_temp": market_temp,
        "data": market_data
    }
    
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__": collect()
