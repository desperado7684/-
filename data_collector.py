import yfinance as yf
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# 카테고리별 정밀 추적 자산 (거시경제, 글로벌 지수, 핵심 포트폴리오)
ASSETS = {
    # 1. Macro & Yields (거시/금리/환율)
    "KRW=X": {"name": "USD/KRW 환율", "cat": "macro", "invert": True}, # 환율 상승은 KOSPI에 악재
    "^TNX": {"name": "미 국채 10년물", "cat": "macro", "invert": True},
    "^VIX": {"name": "VIX 공포지수", "cat": "macro", "invert": True},
    "DX-Y.NYB": {"name": "달러 인덱스", "cat": "macro", "invert": True},
    
    # 2. Global Indices (글로벌 지수)
    "^SOX": {"name": "필라델피아 반도체", "cat": "index", "invert": False},
    "^IXIC": {"name": "NASDAQ", "cat": "index", "invert": False},
    "^GSPC": {"name": "S&P 500", "cat": "index", "invert": False},
    "^N225": {"name": "닛케이 225", "cat": "index", "invert": False},
    
    # 3. Core Holdings & Indicators (핵심 자산 및 지표)
    "MSFT": {"name": "Microsoft", "cat": "core", "invert": False},
    "META": {"name": "Meta Platforms", "cat": "core", "invert": False},
    "BTC-USD": {"name": "Bitcoin", "cat": "core", "invert": False},
    "000660.KS": {"name": "SK하이닉스", "cat": "kr_core", "invert": False},
    "005930.KS": {"name": "삼성전자", "cat": "kr_core", "invert": False},
    "005380.KS": {"name": "현대차", "cat": "kr_core", "invert": False},
    "^KS11": {"name": "KOSPI", "cat": "kr_core", "invert": False}
}

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def collect_data():
    kst = pytz.timezone('Asia/Seoul')
    now_kst = datetime.now(kst)
    market_data = {}
    
    # 예측 엔진을 위한 팩터 변수
    factors = {"bull_score": 0, "bear_score": 0, "total_weight": 0}

    for symbol, info in ASSETS.items():
        try:
            ticker = yf.Ticker(symbol)
            # 기술적 분석을 위해 1개월 데이터 수집
            hist = ticker.history(period="1mo")
            
            if len(hist) >= 15: # RSI 계산 최소 요건 충족
                close_prices = hist['Close']
                current_price = close_prices.iloc[-1]
                prev_price = close_prices.iloc[-2]
                change_pct = ((current_price - prev_price) / prev_price) * 100
                
                # 변동성(최근 5일 표준편차) 및 RSI 계산
                volatility = close_prices.tail(5).pct_change().std() * 100
                rsi_series = calculate_rsi(close_prices)
                current_rsi = rsi_series.iloc[-1]
                
                market_data[symbol] = {
                    "name": info["name"],
                    "cat": info["cat"],
                    "price": round(current_price, 2),
                    "change": round(change_pct, 2),
                    "rsi": round(current_rsi, 1) if not np.isnan(current_rsi) else 50,
                    "volatility": round(volatility, 2) if not np.isnan(volatility) else 0,
                    "trend": "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral",
                    "ok": True
                }
                
                # [KOSPI 예측 엔진 로직] 
                # Macro와 글로벌 Index만 가중치 점수에 반영
                if info["cat"] in ["macro", "index"]:
                    # 역상관(invert) 자산은 방향을 반대로 계산
                    direction_multiplier = -1 if info["invert"] else 1
                    # 단순 등락이 아닌 변동성 대비 모멘텀(Momentum / Volatility)을 가중치로 사용
                    weight = 1.5 if symbol in ["^SOX", "KRW=X"] else 1.0 # 반도체와 환율에 초과 가중치
                    impact = (change_pct * direction_multiplier * weight) / (volatility if volatility > 0 else 1)
                    
                    if impact > 0:
                        factors["bull_score"] += impact
                    else:
                        factors["bear_score"] += abs(impact)
                    factors["total_weight"] += weight

        except Exception as e:
            print(f"Fetch failed for {symbol}: {e}")
            market_data[symbol] = {"name": info["name"], "ok": False}

    # 최종 예측 점수 산출 (Z-Score 기반 정규화 모델 모사)
    raw_signal = (factors["bull_score"] - factors["bear_score"]) / (factors["total_weight"] if factors["total_weight"] > 0 else 1)
    
    final_output = {
        "timestamp": now_kst.strftime("%Y-%m-%d %H:%M:%S KST"),
        "market": market_data,
        "analytics": {
            "signal_strength": round(raw_signal, 3),
            "bull_pressure": round(factors["bull_score"], 2),
            "bear_pressure": round(factors["bear_score"], 2)
        }
    }

    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    collect_data()
