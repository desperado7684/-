import yfinance as yf
import json
import pandas as pd
import numpy as np
from datetime import datetime
import pytz
import logging
import time
import math

# ==============================================================================
# [1] 시스템 로깅 및 환경 설정
# ==============================================================================
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [ %(levelname)s ] %(name)s : %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("GOD-QUANT-ENGINE")

# ==============================================================================
# [2] 마스터 자산 유니버스 (대륙별/기능별 완벽 격리)
# ==============================================================================
ASSETS = {
    # 🔴 [가상의 모델링 구간] KOSPI 시초가 예측을 위한 핵심 매크로 선행 지표
    "NQ=F":     {"name": "나스닥 100 선물", "cat": "Macro_Predict", "w": 0.25, "inv": False},
    "^SOX":     {"name": "필라델피아 반도체", "cat": "Macro_Predict", "w": 0.25, "inv": False},
    "KRW=X":    {"name": "원/달러 환율", "cat": "Macro_Predict", "w": 0.20, "inv": True},
    "^VIX":     {"name": "VIX 공포지수", "cat": "Macro_Predict", "w": 0.15, "inv": True},
    "^TNX":     {"name": "미 국채 10년물", "cat": "Macro_Predict", "w": 0.10, "inv": True},
    "^GDAXI":   {"name": "독일 DAX 지수", "cat": "Macro_Predict", "w": 0.05, "inv": False},

    # 🔵 대한민국 (Korea)
    "^KS11":    {"name": "KOSPI 종합지수", "cat": "Korea", "w": 0, "inv": False},
    "^KQ11":    {"name": "KOSDAQ 종합지수", "cat": "Korea", "w": 0, "inv": False},
    "005930.KS":{"name": "삼성전자", "cat": "Korea", "w": 0, "inv": False},
    "000660.KS":{"name": "SK하이닉스", "cat": "Korea", "w": 0, "inv": False},
    "005380.KS":{"name": "현대차", "cat": "Korea", "w": 0, "inv": False},
    "086790.KS":{"name": "하나금융지주", "cat": "Korea", "w": 0, "inv": False},
    
    # 🔵 미국 (USA)
    "NVDA":     {"name": "엔비디아", "cat": "USA", "w": 0, "inv": False},
    "MSFT":     {"name": "마이크로소프트", "cat": "USA", "w": 0, "inv": False},
    "AAPL":     {"name": "애플", "cat": "USA", "w": 0, "inv": False},
    "TSLA":     {"name": "테슬라", "cat": "USA", "w": 0, "inv": False},
    "^GSPC":    {"name": "S&P 500", "cat": "USA", "w": 0, "inv": False},

    # 🔵 아시아 (Asia)
    "^N225":    {"name": "닛케이 225", "cat": "Asia", "w": 0, "inv": False},
    "^HSI":     {"name": "항셍 지수", "cat": "Asia", "w": 0, "inv": False},

    # 🔵 유럽 (Europe)
    "^FTSE":    {"name": "영국 FTSE 100", "cat": "Europe", "w": 0, "inv": False},
    "^FCHI":    {"name": "프랑스 CAC 40", "cat": "Europe", "w": 0, "inv": False}
}

# ==============================================================================
# [3] 데이터 수집 모듈 (Data Fetcher)
# ==============================================================================
class YFinanceFetcher:
    """야후 파이낸스 API의 통신 불안정을 극복하기 위한 지수 백오프(Exponential Backoff) 구현 클래스"""
    @staticmethod
    def fetch(symbol, period="6mo", max_retries=3):
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                if not hist.empty and len(hist) > 60:
                    return hist
                time.sleep(1.5 ** attempt) # 점진적 대기 시간 증가
            except Exception as e:
                logger.warning(f"[API Warning] {symbol} 수집 지연 (시도 {attempt+1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)
        return pd.DataFrame()

# ==============================================================================
# [4] 복합 기술적 분석 모듈 (Technical Analyzer)
# ==============================================================================
class TechnicalIndicator:
    """6중 퀀트 지표를 pandas 벡터 연산으로 고속 산출하는 클래스"""
    @staticmethod
    def calculate(df):
        close = df['Close']
        high = df['High']
        low = df['Low']
        
        try:
            # 1. RSI (14)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rsi = 100 - (100 / (1 + gain / loss)).iloc[-1]
            
            # 2. MACD (12, 26, 9)
            exp1 = close.ewm(span=12, adjust=False).mean()
            exp2 = close.ewm(span=26, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = (macd_line - signal_line).iloc[-1]
            
            # 3. Bollinger Bands (20, 2)
            sma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            upper_bb = sma20 + (std20 * 2)
            lower_bb = sma20 - (std20 * 2)
            bb_pct = (((close - lower_bb) / (upper_bb - lower_bb)) * 100).iloc[-1]

            # 4. Stochastic Oscillator (14, 3)
            lowest_14 = low.rolling(window=14).min()
            highest_14 = high.rolling(window=14).max()
            stoch_k = (100 * ((close - lowest_14) / (highest_14 - lowest_14))).iloc[-1]

            # 5. MA Trend (20일선 vs 60일선)
            ma20 = sma20.iloc[-1]
            ma60 = close.rolling(window=60).mean().iloc[-1]
            trend = "정배열 강세" if ma20 >= ma60 else "역배열 약세"

            return {
                "rsi": round(rsi, 1) if not math.isnan(rsi) else 50.0,
                "macd": round(macd_hist, 2),
                "bb": round(bb_pct, 1) if not math.isnan(bb_pct) else 50.0,
                "stoch_k": round(stoch_k, 1) if not math.isnan(stoch_k) else 50.0,
                "ma_trend": trend
            }
        except Exception as e:
            logger.error(f"지표 연산 실패: {e}")
            return {"rsi": 50, "macd": 0, "bb": 50, "stoch_k": 50, "ma_trend": "알 수 없음"}

# ==============================================================================
# [5] 자연어 해설 생성 모듈 (NLP Commentary Engine)
# ==============================================================================
class NLPCommentaryEngine:
    """기술적 지표를 토스증권 스타일의 친절하고 전문적인 문장으로 번역하는 모듈"""
    @staticmethod
    def generate_asset_comment(name, change_pct, tech, is_inverse):
        rsi = tech['rsi']
        bb = tech['bb']
        macd = tech['macd']
        
        # 매크로/역상관 자산 해설
        if is_inverse:
            if change_pct > 1.2: return f"🚨 {name} 수치가 시장에 부담을 주며 급등 중입니다. 외국인 자본 유출에 대비한 보수적 스탠스가 필요합니다."
            if change_pct < -1.2: return f"🌤️ {name}이(가) 안정세에 접어들었습니다. 억눌렸던 투자 심리가 회복되며 증시에 긍정적인 바람이 불고 있습니다."
            return f"⚖️ {name}은(는) 큰 변동 없이 현재의 추세를 유지하며 시장의 눈치보기를 유도하고 있습니다."

        # 주식 및 지수 해설 (교차 검증)
        base_text = f"현재 {name}은(는) 전일 대비 {change_pct}% 변동을 기록 중입니다. "
        
        if rsi >= 75 and bb >= 100:
            return base_text + "RSI와 볼린저 밴드 모두 '극단적 과열' 신호를 보냅니다. 차익 실현 욕구가 커질 수 있는 위험 구간입니다. 🔥"
        elif rsi <= 25 and bb <= 0:
            return base_text + "시장 투매로 인한 '극단적 과매도' 상태입니다. 지지선에서 기술적 반등이 강하게 일어날 수 있는 줍줍 찬스일 수 있습니다. 💡"
        
        if macd > 0 and "강세" in tech['ma_trend']:
            return base_text + "이동평균선 정배열과 MACD 상승 에너지가 완벽하게 맞물려 강력한 주도주 역할을 하고 있습니다. 🚀"
        elif macd < 0 and "약세" in tech['ma_trend']:
            return base_text + "추세가 완전히 무너진 상태로, 하락 모멘텀이 거셉니다. 무리한 물타기보다는 바닥 확인이 우선입니다. 📉"
            
        return base_text + "상승과 하락의 모멘텀이 팽팽합니다. 시장(Market Breadth)의 전체적인 온도에 순응하는 전략이 유리합니다. 🧩"

# ==============================================================================
# [6] 코스피 오라클 예측 모델링 (Macro Oracle Model)
# ==============================================================================
class MacroOracleModel:
    """[가상의 추론 모델] 선행 매크로 지표를 종합하여 내일의 코스피를 예측하고 리포트를 작성하는 클래스"""
    def __init__(self):
        self.bull_energy = 0.0
        self.bear_energy = 0.0
        self.impact_logs = []
        self.risk_assets_total = 0
        self.risk_assets_up = 0

    def process_macro_factor(self, name, change_pct, weight, is_inverse):
        """매크로 지표의 가중치 및 임팩트 연산"""
        impact = change_pct * weight
        if is_inverse: 
            impact = -impact # 환율/VIX 상승은 증시에 악재(마이너스 임팩트)
        
        # 비선형 모멘텀 가중치 부여 (급등/급락 시 영향력 증폭)
        if abs(change_pct) > 1.5: 
            impact *= 1.3 
            
        self.impact_logs.append({"name": name, "impact": impact})
        
        if impact > 0: self.bull_energy += impact
        else: self.bear_energy += abs(impact)

    def process_market_breadth(self, category, change_pct):
        """시장 온도계 (Market Breadth) 연산을 위한 위험 자산 카운팅"""
        if category in ["Korea", "USA", "Asia", "Europe"]:
            self.risk_assets_total += 1
            if change_pct > 0:
                self.risk_assets_up += 1

    def generate_final_report(self):
        """오라클 스코어, 시장 온도, 그리고 동적 서술형 리포트 최종 산출"""
        # 스코어 정규화 (0~100)
        total_e = self.bull_energy + self.bear_energy
        if total_e == 0: total_e = 1.0
        oracle_score = round((self.bull_energy / total_e) * 100)
        
        # 시장 온도 (0~100)
        market_temp = round((self.risk_assets_up / self.risk_assets_total) * 100) if self.risk_assets_total > 0 else 50

        # 동적 리포트 작성 알고리즘
        best_factor = max(self.impact_logs, key=lambda x: x['impact']) if self.impact_logs else {"name": "알 수 없음", "impact": 0}
        worst_factor = min(self.impact_logs, key=lambda x: x['impact']) if self.impact_logs else {"name": "알 수 없음", "impact": 0}

        report = f"💡 [알고리즘 분석 결과]\n"
        
        # 1. 내일의 예측 설명
        if oracle_score >= 60:
            report += f"내일 코스피 시초가는 긍정적일 확률이 높습니다. 특히 글로벌 시장에서 '{best_factor['name']}'의 우호적인 흐름이 한국 증시에 강력한 상승 에너지를 불어넣고 있습니다. "
        elif oracle_score <= 40:
            report += f"내일 코스피는 하방 압력에 대비해야 합니다. 가장 큰 원인은 '{worst_factor['name']}'의 불안정한 움직임이며, 이로 인해 외국인 투심이 위축될 우려가 큽니다. "
        else:
            report += f"내일 한국 증시는 눈치보기 장세가 열릴 것입니다. 상승 요인({best_factor['name']})과 하락 요인({worst_factor['name']})이 팽팽하게 맞서고 있습니다. "

        # 2. 현재 시장 온도 설명
        report += f"\n\n🌡️ [시장 온도: {market_temp}°C의 의미]\n"
        if market_temp <= 35:
            report += f"현재 온도가 {market_temp}도인 이유는, 전 세계 핵심 주식 및 지수의 65% 이상이 일제히 파란불(하락)을 켜고 있는 '글로벌 리스크 오프(위험 회피)' 상태이기 때문입니다."
        elif market_temp >= 65:
            report += f"현재 온도가 {market_temp}도로 뜨거운 이유는, 대륙을 불문하고 추적 중인 글로벌 자산의 대다수가 빨간불(상승)을 기록하며 강한 매수세가 유입되고 있기 때문입니다."
        else:
            report += f"현재 온도가 {market_temp}도로 미지근한 이유는, 특정 섹터나 국가만 오르는 차별화/순환매 장세가 진행 중이기 때문입니다."

        return oracle_score, market_temp, report

# ==============================================================================
# [7] 메인 오케스트레이터 (K-Quant Terminal Core)
# ==============================================================================
def execute_quant_terminal():
    kst = pytz.timezone('Asia/Seoul')
    now = datetime.now(kst)
    logger.info("==================================================")
    logger.info("🚀 KOSPI ORACLE V6 (OOP EDITION) ENGINE START")
    logger.info("==================================================")

    oracle = MacroOracleModel()
    processed_assets = []

    for sym, info in ASSETS.items():
        hist = YFinanceFetcher.fetch(sym)
        if hist.empty:
            logger.error(f"❌ {info['name']} 데이터 누락으로 분석 제외.")
            continue
            
        # 기본 데이터 가공
        close_series = hist['Close']
        cur_price = close_series.iloc[-1]
        prev_price = close_series.iloc[-2]
        change_pct = round(((cur_price - prev_price) / prev_price) * 100, 2)
        
        # 기술적 분석
        tech_data = TechnicalIndicator.calculate(hist)
        
        # NLP 코멘트 생성
        commentary = NLPCommentaryEngine.generate_asset_comment(info['name'], change_pct, tech_data, info.get('inv', False))

        # 오라클 모델에 데이터 주입
        if info["cat"] == "Macro_Predict":
            oracle.process_macro_factor(info["name"], change_pct, info["w"], info.get("inv", False))
        oracle.process_market_breadth(info["cat"], change_pct)

        # 결과 저장
        processed_assets.append({
            "symbol": sym,
            "name": info["name"],
            "cat": info["cat"],
            "price": round(cur_price, 2) if cur_price > 100 else round(cur_price, 4),
            "change": change_pct,
            "tech": tech_data,
            "comment": commentary
        })

    # 데이터 정렬 (등락률 기준)
    processed_assets.sort(key=lambda x: x["change"], reverse=True)

    # 최종 오라클 리포트 생성
    final_score, final_temp, final_report = oracle.generate_final_report()

    # JSON 덤프
    output_payload = {
        "kst": now.strftime("%Y-%m-%d %H:%M:%S"),
        "market_temp": final_temp,
        "prediction": {
            "score": final_score,
            "report": final_report,
            "disclaimer": "본 분석의 스코어 및 리포트는 과거의 통계적 비중을 임의로 부여한 [가상의 모델링 자료]입니다. 투자 결과에 대한 법적 책임은 지지 않습니다."
        },
        "data": processed_assets
    }
    
    try:
        with open('data.json', 'w', encoding='utf-8') as f:
            json.dump(output_payload, f, ensure_ascii=False, indent=2)
        logger.info(f"✅ 오라클 스코어: {final_score}점 | 시장 온도: {final_temp}도")
        logger.info("🎉 data.json 파일이 성공적으로 빌드되었습니다.")
    except Exception as e:
        logger.critical(f"시스템 치명적 오류 - JSON 저장 실패: {e}")

if __name__ == "__main__":
    execute_quant_terminal()
