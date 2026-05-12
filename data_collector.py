import yfinance as yf
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pytz

# 1. 200% 완벽을 위한 확장된 마스터 유니버스
ASSETS = {
    # [가상의 모델링 적용 구간] 내일 코스피 방향성 예측을 위한 핵심 선행 지표 및 가중치
    "NQ=F": {"name": "나스닥 100 선물", "type": "선행지표", "weight": 0.25, "inv": False},
    "^SOX": {"name": "필라델피아 반도체", "type": "선행지표", "weight": 0.25, "inv": False},
    "^GDAXI": {"name": "독일 DAX 지수", "type": "선행지표", "weight": 0.10, "inv": False},
    "KRW=X": {"name": "원/달러 환율", "type": "선행지표", "weight": 0.20, "inv": True},
    "^VIX": {"name": "VIX 공포지수", "type": "선행지표", "weight": 0.10, "inv": True},
    "^TNX": {"name": "미 국채 10년물", "type": "선행지표", "weight": 0.10, "inv": True},

    # 핵심 포트폴리오 및 벤치마크
    "^KS11": {"name": "KOSPI 종합지수", "type": "국내지수"},
    "000660.KS": {"name": "SK하이닉스", "type": "국내주식"},
    "005930.KS": {"name": "삼성전자", "type": "국내주식"},
    "005380.KS": {"name": "현대차", "type": "국내주식"},
    "086790.KS": {"name": "하나금융지주", "type": "국내주식"},
    
    # 글로벌 빅테크 및 자산
    "NVDA": {"name": "엔비디아", "type": "해외주식"},
    "MSFT": {"name": "마이크로소프트", "type": "해외주식"},
    "META": {"name": "메타 (페이스북)", "type": "해외주식"},
    "TSLA": {"name": "테슬라", "type": "해외주식"},
    "BTC-USD": {"name": "비트코인 (BTC)", "type": "가상자산"}
}

def analyze_trend_and_comment(name, change, rsi, macd, ma20, ma60, is_inv, asset_type):
    """지표 크로스체크를 통한 200% 정밀도의 토스 스타일 코멘트 생성"""
    # 매크로/선행지표 논리
    if is_inv:
        if change > 1.5: return f"{name} 급등은 증시의 발목을 잡는 강한 악재예요. 리스크 관리가 필수입니다. 🚨"
        if change < -1.5: return f"{name} 하락은 시장에 안도감을 줘요. 외국인 수급이 들어올 수 있는 좋은 신호예요. 🌤️"
        return f"{name} 움직임이 안정적이에요. 시장도 큰 충격 없이 흘러갈 가능성이 높아요. ⚖️"
    
    # 기술적 분석 논리 (크로스체크)
    if rsi >= 75 and macd < 0:
        return "과열된 상태에서 추세가 꺾이고 있어요. 단기 고점일 확률이 높으니 추격 매수는 매우 위험해요. ⚠️"
    if rsi <= 30 and macd > 0:
        return "많이 빠진 상태에서 상승 에너지가 들어오고 있어요! 매력적인 줍줍(저점 매수) 타이밍일 수 있어요. 🎯"
    if ma20 > ma60 and change > 1.0:
        return "단기/중기 상승 추세(정배열)가 완벽해요. 모멘텀이 매우 강하게 살아있습니다. 🚀"
    if ma20 < ma60 and change < -1.0:
        return "하락 추세(역배열)가 깊어지고 있어요. 섣부른 물타기보다는 바닥을 확인하는 게 중요해요. 📉"
    
    # 기본 변동성 논리
    if change > 2.0: return "오늘 시장의 주도주 역할을 톡톡히 하고 있어요. 거래량을 동반했다면 긍정적이에요. 🔥"
    if change < -2.0: return "단기적인 투심이 크게 꺾였어요. 악재 뉴스가 있는지 꼭 확인해보세요. 🔍"
    return "뚜렷한 방향성 없이 매수/매도세가 힘겨루기를 하고 있어요. 조용히 관망하기 좋은 때예요. 🧩"

def run_quant_engine():
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    market_data = []
    
    # [가상의 모델링] 오라클 예측 에너지
    bull_energy = 0
    bear_energy = 0
    risk_assets_bull_count = 0
    risk_assets_total = 0

    for sym, info in ASSETS.items():
        try:
            t = yf.Ticker(sym)
            hist = t.history(period="6mo") # MA60 계산을 위해 6개월치 로드
            if len(hist) < 65: continue
            
            close = hist['Close']
            cur = close.iloc[-1]
            prev = close.iloc[-2]
            chg = round(((cur - prev) / prev) * 100, 2)
            
            # 1. RSI (14)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = round(100 - (100 / (1 + rs.iloc[-1])), 1) if not np.isnan(rs.iloc[-1]) else 50
            
            # 2. MACD (12, 26, 9)
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = round((macd - signal).iloc[-1], 2)
            
            # 3. Moving Averages (20, 60)
            ma20 = close.rolling(window=20).mean().iloc[-1]
            ma60 = close.rolling(window=60).mean().iloc[-1]

            # 디테일한 코멘트 생성
            comment = analyze_trend_and_comment(info["name"], chg, rsi, macd_hist, ma20, ma60, info.get("inv", False), info["type"])

            # 시장 온도계 로직 (위험 자산 기준)
            if info["type"] in ["국내주식", "해외주식", "가상자산", "국내지수", "해외지수"]:
                risk_assets_total += 1
                if chg > 0: risk_assets_bull_count += 1

            # [가상의 모델링] 내일의 코스피 예측 엔진
            if info["type"] == "선행지표":
                impact = chg * info["weight"]
                if info.get("inv"): impact = -impact
                
                if impact > 0: bull_energy += impact
                else: bear_energy += abs(impact)

            market_data.append({
                "symbol": sym, "name": info["name"], "type": info["type"],
                "price": round(cur, 2) if cur > 100 else round(cur, 4),
                "change": chg, 
                "tech": {"rsi": rsi, "macd": macd_hist, "ma_trend": "정배열" if ma20 > ma60 else "역배열"},
                "comment": comment
            })
        except Exception as e:
            print(f"Data Fetch Error for {sym}: {e}")

    # 데이터 정렬 (상승률 높은 순)
    market_data.sort(key=lambda x: x["change"], reverse=True)

    # [가상의 모델링] 최종 오라클 스코어 산출
    total_energy = bull_energy + bear_energy if (bull_energy + bear_energy) > 0 else 1
    oracle_score = round((bull_energy / total_energy) * 100)
    
    # 시장 온도계 (0~100도)
    market_temp = round((risk_assets_bull_count / risk_assets_total) * 100) if risk_assets_total > 0 else 50

    output = {
        "kst": now.strftime("%Y년 %m월 %d일 %H:%M:%S"),
        "market_temp": market_temp,
        "prediction": {"score": oracle_score},
        "data": market_data
    }
    
    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    run_quant_engine()
