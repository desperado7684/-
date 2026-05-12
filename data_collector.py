import yfinance as yf
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import logging
import time

# ==============================================================================
# 1. 엔터프라이즈급 로깅 시스템 설정 (GitHub Actions 디버깅용)
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("K-QUANT-ENGINE")

# ==============================================================================
# 2. 글로벌 자산 마스터 유니버스 및 모델링 가중치
# ==============================================================================
# [가상의 모델링 자료] KOSPI 예측을 위한 각 지표별 통계적 가중치(Weight) 설정
ASSETS = {
    # 선행 매크로 & 리스크 (KOSPI 시초가 결정 핵심 요인)
    "NQ=F":     {"name": "나스닥 100 선물", "type": "선행지표", "w": 0.25, "inv": False},
    "^SOX":     {"name": "필라델피아 반도체", "type": "선행지표", "w": 0.25, "inv": False},
    "KRW=X":    {"name": "원/달러 환율", "type": "선행지표", "w": 0.20, "inv": True},
    "^VIX":     {"name": "VIX 공포지수", "type": "선행지표", "w": 0.15, "inv": True},
    "^TNX":     {"name": "미 국채 10년물", "type": "선행지표", "w": 0.10, "inv": True},
    "^GDAXI":   {"name": "독일 DAX (유럽장)", "type": "선행지표", "w": 0.05, "inv": False},

    # 핵심 벤치마크 & 국내 포트폴리오
    "^KS11":    {"name": "KOSPI 종합지수", "type": "국내지수", "w": 0},
    "^KQ11":    {"name": "KOSDAQ 종합지수", "type": "국내지수", "w": 0},
    "005930.KS":{"name": "삼성전자", "type": "국내주식", "w": 0},
    "000660.KS":{"name": "SK하이닉스", "type": "국내주식", "w": 0},
    "005380.KS":{"name": "현대차", "type": "국내주식", "w": 0},
    "086790.KS":{"name": "하나금융지주", "type": "국내주식", "w": 0},
    "035420.KS":{"name": "NAVER", "type": "국내주식", "w": 0},
    "005490.KS":{"name": "POSCO홀딩스", "type": "국내주식", "w": 0},
    
    # 글로벌 빅테크 및 암호화폐
    "NVDA":     {"name": "엔비디아", "type": "해외주식", "w": 0},
    "MSFT":     {"name": "마이크로소프트", "type": "해외주식", "w": 0},
    "AAPL":     {"name": "애플", "type": "해외주식", "w": 0},
    "META":     {"name": "메타 (페이스북)", "type": "해외주식", "w": 0},
    "TSLA":     {"name": "테슬라", "type": "해외주식", "w": 0},
    "BTC-USD":  {"name": "비트코인 (BTC)", "type": "가상자산", "w": 0}
}

# ==============================================================================
# 3. 딥 퀀트 수학적 지표 산출 엔진 (Technical Analysis Matrix)
# ==============================================================================
def calculate_technical_matrix(df):
    """
    Pandas 기반의 고정밀 기술적 지표 산출 모듈
    반환값: RSI, MACD Histogram, Bollinger Bands %B, Stochastic K/D, ATR, 이동평균선(20,60,120)
    """
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    tech = {}
    
    try:
        # 3-1. RSI (Relative Strength Index, 14일)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        tech['rsi'] = round(rsi.iloc[-1], 2) if not np.isnan(rsi.iloc[-1]) else 50.0

        # 3-2. MACD (Moving Average Convergence Divergence, 12-26-9)
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        tech['macd_hist'] = round((macd - signal).iloc[-1], 3)

        # 3-3. Bollinger Bands (20일, 2 표준편차)
        sma20 = close.rolling(window=20, min_periods=1).mean()
        std20 = close.rolling(window=20, min_periods=1).std()
        upper_bb = sma20 + (std20 * 2)
        lower_bb = sma20 - (std20 * 2)
        # %B: 주가가 밴드 내 어느 위치에 있는지 (1 이상 과매수, 0 이하 과매도)
        bb_pct = (close.iloc[-1] - lower_bb.iloc[-1]) / (upper_bb.iloc[-1] - lower_bb.iloc[-1]) * 100
        tech['bb_pct'] = round(bb_pct, 1) if not np.isnan(bb_pct) else 50.0

        # 3-4. Stochastic Oscillator (Fast K 14, Slow D 3)
        lowest_low = low.rolling(window=14, min_periods=1).min()
        highest_high = high.rolling(window=14, min_periods=1).max()
        fast_k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        slow_d = fast_k.rolling(window=3, min_periods=1).mean()
        tech['stoch_k'] = round(fast_k.iloc[-1], 1) if not np.isnan(fast_k.iloc[-1]) else 50.0
        tech['stoch_d'] = round(slow_d.iloc[-1], 1) if not np.isnan(slow_d.iloc[-1]) else 50.0

        # 3-5. ATR (Average True Range, 14일) - 변동성 측정
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14, min_periods=1).mean()
        # ATR을 현재가로 나누어 퍼센티지로 변환 (종목 간 변동성 비교용)
        tech['atr_pct'] = round((atr.iloc[-1] / close.iloc[-1]) * 100, 2)

        # 3-6. Moving Averages (추세 배열 확인용 20, 60, 120일)
        ma20 = sma20.iloc[-1]
        ma60 = close.rolling(window=60, min_periods=1).mean().iloc[-1]
        ma120 = close.rolling(window=120, min_periods=1).mean().iloc[-1]
        
        if ma20 > ma60 > ma120: tech['trend'] = "완벽한 정배열 (강세)"
        elif ma20 < ma60 < ma120: tech['trend'] = "완벽한 역배열 (약세)"
        elif ma20 > ma60: tech['trend'] = "단기 상승 전환"
        else: tech['trend'] = "단기 하락 전환"

        return tech

    except Exception as e:
        logger.error(f"기술적 지표 계산 오류: {e}")
        return {"rsi": 50, "macd_hist": 0, "bb_pct": 50, "stoch_k": 50, "stoch_d": 50, "atr_pct": 0, "trend": "계산 불가"}

# ==============================================================================
# 4. 토스증권 스타일 AI 애널리스트 코멘트 제너레이터
# ==============================================================================
def generate_ai_comment(name, change, tech, is_inv, asset_type):
    """복합 기술적 지표를 활용한 자연어 해설 생성 로직"""
    rsi, macd, bb, trend = tech['rsi'], tech['macd_hist'], tech['bb_pct'], tech['trend']
    
    # [가상의 추론] 1. 거시경제 및 리스크 지표 해설
    if is_inv:
        if change > 1.5: 
            return f"🚨 경계 경보! {name}이(가) 급등하며 시장 전체에 강력한 하방 압력을 가하고 있습니다. 적극적인 리스크 관리가 필요합니다."
        if change < -1.5: 
            return f"🌤️ 맑음! 증시를 짓누르던 {name}이(가) 하락하며 외국인 수급 유입 환경이 조성되고 있습니다."
        return f"⚖️ {name}은(는) 큰 변동 없이 안정적인 흐름을 유지하며 시장에 미치는 영향이 제한적입니다."

    # [가상의 추론] 2. 기술적 과열/침체 해설 (RSI + Bollinger Bands 교차 검증)
    if rsi >= 75 and bb >= 100:
        return "🔥 극단적 과열 상태입니다. 볼린저 밴드 상단을 이탈했으며, 단기 차익 실현 매물이 쏟아질 수 있으니 신규 진입은 위험합니다."
    if rsi <= 25 and bb <= 0:
        return "💡 극단적 과매도 구간입니다. 매도세가 소진되어 가고 있으며, 지지선에서 기술적 반등(V자 랠리)을 노려볼 만한 위치입니다."

    # [가상의 추론] 3. 모멘텀 및 추세 해설 (MACD + Moving Average)
    if macd > 0 and "정배열" in trend:
        if change > 1.0: return "🚀 완벽한 상승 추세입니다. 중장기 이평선이 지지해 주고 매수 모멘텀이 강하게 폭발하고 있습니다."
        else: return "📈 탄탄한 정배열을 유지하며 안정적으로 우상향 중입니다. 눌림목(조정) 발생 시 좋은 매수 기회가 될 수 있습니다."
    
    if macd < 0 and "역배열" in trend:
        if change < -1.0: return "📉 하락 추세가 깊어지고 있습니다. 섣부른 물타기보다는 바닥(지지선)이 명확히 확인될 때까지 기다리세요."
        else: return "⚠️ 이평선이 역배열 상태로 위에 저항 매물이 많습니다. 상승하더라도 제한적인 기술적 반등일 확률이 높습니다."

    # [가상의 추론] 4. 일반적인 변동성 해설
    if change > 2.5: return f"💪 오늘 시장의 주도력을 보여주며 {change}% 급등했습니다. 거래량 동반 여부를 꼭 체크하세요."
    if change < -2.5: return f"💔 심리가 크게 위축되며 {abs(change)}% 급락했습니다. 개별 악재 뉴스가 있는지 확인이 시급합니다."
    
    return "🧩 뚜렷한 방향성 없이 매수와 매도 세력이 치열하게 눈치싸움을 벌이고 있는 횡보 구간입니다."

# ==============================================================================
# 5. 메인 데이터 수집 파이프라인 (안정성 강화)
# ==============================================================================
def fetch_data_with_retry(symbol, retries=3):
    """야후 파이낸스 API 통신 불안정성을 극복하기 위한 재시도 로직"""
    for attempt in range(retries):
        try:
            ticker = yf.Ticker(symbol)
            # 기술적 지표(120일선 등) 계산을 위해 충분한 6개월(약 130거래일) 데이터 수집
            hist = ticker.history(period="6mo") 
            if not hist.empty and len(hist) > 120:
                return hist
            time.sleep(1) # API Rate Limit 방지
        except Exception as e:
            logger.warning(f"[{attempt+1}/{retries}] {symbol} 데이터 수집 실패: {e}")
            time.sleep(2)
    return pd.DataFrame()

def run_quant_master_engine():
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    logger.info("=== K-QUANT MASTER ENGINE 부팅 시작 ===")
    
    market_data = []
    
    # [가상의 모델링] 오라클 예측 에너지 버퍼
    bull_energy = 0.0
    bear_energy = 0.0
    
    # 시장 온도계 버퍼 (위험자산 상승 비율)
    risk_assets_total = 0
    risk_assets_up = 0

    for sym, info in ASSETS.items():
        hist = fetch_data_with_retry(sym)
        
        if hist.empty:
            logger.error(f"❌ {sym} ({info['name']}) 데이터 수집 최종 실패. 스킵합니다.")
            continue
            
        try:
            close_series = hist['Close']
            cur_price = close_series.iloc[-1]
            prev_price = close_series.iloc[-2]
            change_pct = round(((cur_price - prev_price) / prev_price) * 100, 2)
            
            # 기술적 지표 매트릭스 산출
            tech_matrix = calculate_technical_matrix(hist)
            
            # 자연어 코멘트 생성
            comment = generate_ai_comment(info["name"], change_pct, tech_matrix, info.get("inv", False), info["type"])
            
            # --- [가상의 모델링 구간 시작] ---
            # 1. KOSPI 내일 방향성 예측 가중치 합산
            if info["type"] == "선행지표":
                # 기본 가중치 × 당일 변동률
                impact = change_pct * info["w"]
                # 역상관 지표(환율, VIX 등)는 부호를 반대로 계산하여 리스크 반영
                if info.get("inv"): 
                    impact = -impact
                
                # 강한 변동성에 가중치를 더 주는 Momentum 승수 적용
                if abs(change_pct) > 1.5: impact *= 1.2 
                
                if impact > 0: bull_energy += impact
                else: bear_energy += abs(impact)
            # --- [가상의 모델링 구간 종료] ---

            # 2. 시장 온도계 (Market Breadth) 계산
            if info["type"] in ["국내주식", "해외주식", "가상자산", "국내지수", "해외지수"]:
                risk_assets_total += 1
                if change_pct > 0: 
                    risk_assets_up += 1

            # 최종 데이터 구조체 조립
            market_data.append({
                "symbol": sym,
                "name": info["name"],
                "type": info["type"],
                "price": round(cur_price, 2) if cur_price > 100 else round(cur_price, 4),
                "change": change_pct,
                "tech": tech_matrix,
                "comment": comment
            })
            logger.info(f"✅ {info['name']} 처리 완료 (Chg: {change_pct}%)")

        except Exception as e:
            logger.error(f"❌ {sym} 처리 중 치명적 오류 발생: {e}")

    # ==========================================================================
    # 6. 최종 메타 지표 산출 및 저장
    # ==========================================================================
    # 자산 정렬: 상승률이 가장 높은 순으로 정렬하여 UI 편의성 제공
    market_data.sort(key=lambda x: x["change"], reverse=True)

    # [가상의 모델링] 최종 오라클 스코어 정규화 (0 ~ 100)
    total_energy = bull_energy + bear_energy
    if total_energy == 0: total_energy = 1.0 # ZeroDivisionError 방지
    
    # 0~100 스케일로 변환
    oracle_score = round((bull_energy / total_energy) * 100)
    
    # 시장 온도계 산출
    market_temp = round((risk_assets_up / risk_assets_total) * 100) if risk_assets_total > 0 else 50

    # JSON 덤프 생성
    final_output = {
        "kst": now.strftime("%Y년 %m월 %d일 %H:%M:%S"),
        "market_temp": market_temp,
        "prediction": {
            "score": oracle_score,
            "bull_raw": round(bull_energy, 2),
            "bear_raw": round(bear_energy, 2),
            "logic_notice": "본 수치는 통계적 가중치를 부여한 가상의 수학적 추론 모델입니다."
        },
        "data": market_data,
        "meta": {
            "total_assets": len(market_data),
            "engine_version": "V5_ULTIMATE_MASTER"
        }
    }
    
    try:
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        logger.info(f"🎉 data.json 저장 완료! (오라클 스코어: {oracle_score}, 시장 온도: {market_temp})")
    except Exception as e:
        logger.error(f"데이터 파일 저장 실패: {e}")

if __name__ == "__main__":
    run_quant_master_engine()
