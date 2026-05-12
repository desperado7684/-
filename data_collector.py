"""
═══════════════════════════════════════════════════════════════════════════
  KOSPI ORACLE V7 — OMNIPOTENT QUANT ENGINE
  ───────────────────────────────────────────────────────────────────────
  ▸ 글로벌 매크로 자산 21종 실시간 수집 (yfinance)
  ▸ 6중 기술적 지표 (RSI, MACD, BB %B, Stochastic, MA Trend, ATR)
  ▸ 규칙 기반 오라클 모델 (가중치 + 비선형 모멘텀 + 클리핑)
  ▸ Gemini AI 통합 (자연어 리포트 재작성, 페일오버 안전장치)
  ▸ 시계열 누적 (sparkline + 일중 변동 추적)
  ▸ 예측 히스토리 로깅 (적중률 사후 검증용)
  ───────────────────────────────────────────────────────────────────────
  Author : Jinju First Girls' High School — 태경
  Engine : GOD-LEVEL QUANT TERMINAL
═══════════════════════════════════════════════════════════════════════════
"""

import os
import json
import math
import time
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yfinance as yf
import pandas as pd
import numpy as np
import pytz
import requests

# ═══════════════════════════════════════════════════════════════════════════
#  [1] 환경 설정 및 로깅
# ═══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [ %(levelname)-7s ] %(name)-18s │ %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ORACLE-V7")

# 파일 경로 (저장소 루트 기준)
BASE_DIR              = Path(__file__).parent
DATA_FILE             = BASE_DIR / "data.json"
HISTORY_FILE          = BASE_DIR / "history.json"
PREDICTION_LOG_FILE   = BASE_DIR / "prediction_log.json"

# Gemini API 설정
GEMINI_API_KEY        = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL          = "gemini-1.5-flash"
GEMINI_ENDPOINT       = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

# 시간 설정
KST                   = pytz.timezone('Asia/Seoul')
NOW_KST               = datetime.now(KST)
TODAY_KEY             = NOW_KST.strftime("%Y-%m-%d")

# 수집 설정
HIST_PERIOD           = "3mo"   # 3개월 (RSI/MACD/BB/MA60 계산 충분)
SPARKLINE_POINTS      = 30      # 카드별 미니 차트 포인트 수
MAX_RETRIES           = 3
RETRY_BACKOFF_BASE    = 1.5

# ═══════════════════════════════════════════════════════════════════════════
#  [2] 마스터 자산 유니버스
#      ─ cat: 카테고리 (UI 필터링용)
#      ─ w  : 오라클 가중치 (Macro_Predict만 사용, 합계 1.0)
#      ─ inv: 역상관 자산 (환율/VIX/금리는 상승=악재)
# ═══════════════════════════════════════════════════════════════════════════
ASSETS: Dict[str, Dict] = {
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🔴 [Macro_Predict] KOSPI 시초가 예측용 핵심 선행 지표 (총 가중치 1.00)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "NQ=F":      {"name": "나스닥 100 선물",   "cat": "Macro_Predict", "w": 0.25, "inv": False},
    "SOXX":      {"name": "필라델피아 반도체 ETF", "cat": "Macro_Predict", "w": 0.25, "inv": False},  # ^SOX → SOXX 안정화
    "KRW=X":     {"name": "원/달러 환율",       "cat": "Macro_Predict", "w": 0.20, "inv": True},
    "^VIX":      {"name": "VIX 공포지수",       "cat": "Macro_Predict", "w": 0.15, "inv": True},
    "^TNX":      {"name": "미 국채 10년물",     "cat": "Macro_Predict", "w": 0.10, "inv": True},
    "^GDAXI":    {"name": "독일 DAX 지수",      "cat": "Macro_Predict", "w": 0.05, "inv": False},

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🇰🇷 대한민국
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "^KS11":     {"name": "KOSPI 종합지수",     "cat": "Korea", "w": 0, "inv": False},
    "^KQ11":     {"name": "KOSDAQ 종합지수",    "cat": "Korea", "w": 0, "inv": False},
    "005930.KS": {"name": "삼성전자",           "cat": "Korea", "w": 0, "inv": False},
    "000660.KS": {"name": "SK하이닉스",         "cat": "Korea", "w": 0, "inv": False},
    "005380.KS": {"name": "현대차",             "cat": "Korea", "w": 0, "inv": False},
    "086790.KS": {"name": "하나금융지주",       "cat": "Korea", "w": 0, "inv": False},

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🇺🇸 미국
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "^GSPC":     {"name": "S&P 500",            "cat": "USA", "w": 0, "inv": False},
    "NVDA":      {"name": "엔비디아",           "cat": "USA", "w": 0, "inv": False},
    "MSFT":      {"name": "마이크로소프트",     "cat": "USA", "w": 0, "inv": False},
    "AAPL":      {"name": "애플",               "cat": "USA", "w": 0, "inv": False},
    "TSLA":      {"name": "테슬라",             "cat": "USA", "w": 0, "inv": False},

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌏 아시아
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "^N225":     {"name": "닛케이 225",         "cat": "Asia", "w": 0, "inv": False},
    "^HSI":      {"name": "항셍 지수",          "cat": "Asia", "w": 0, "inv": False},

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🇪🇺 유럽
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "^FTSE":     {"name": "영국 FTSE 100",      "cat": "Europe", "w": 0, "inv": False},
    "^FCHI":     {"name": "프랑스 CAC 40",      "cat": "Europe", "w": 0, "inv": False},
}


# ═══════════════════════════════════════════════════════════════════════════
#  [3] 데이터 수집기 (지수 백오프 + 검증)
# ═══════════════════════════════════════════════════════════════════════════
class YFinanceFetcher:
    """야후 파이낸스 호출 안정화 클래스. 통신 실패 시 지수 백오프 재시도."""

    @staticmethod
    def fetch(symbol: str, period: str = HIST_PERIOD) -> pd.DataFrame:
        """단일 자산 시계열 데이터 수집. 실패 시 빈 DataFrame 반환."""
        for attempt in range(MAX_RETRIES):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, auto_adjust=True)

                # 데이터 무결성 검증
                if hist.empty:
                    raise ValueError("응답 데이터 비어있음")
                if len(hist) < 60:
                    raise ValueError(f"데이터 부족 ({len(hist)} < 60)")
                if hist['Close'].isna().all():
                    raise ValueError("종가 데이터 전부 NaN")

                return hist

            except Exception as e:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(f"  ⚠️  {symbol} 시도 {attempt+1}/{MAX_RETRIES} 실패: {e} → {wait:.1f}초 대기")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)

        logger.error(f"  ❌ {symbol} 최종 수집 실패 (모든 재시도 소진)")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
#  [4] 기술적 지표 분석 모듈 (벡터 연산)
# ═══════════════════════════════════════════════════════════════════════════
class TechnicalIndicator:
    """6중 퀀트 지표 산출기. pandas/numpy 벡터화로 고속 처리."""

    @staticmethod
    def calculate(df: pd.DataFrame) -> Dict:
        """RSI, MACD, BB %B, Stochastic, MA Trend, ATR 일괄 계산."""
        close = df['Close']
        high  = df['High']
        low   = df['Low']

        try:
            # ─── 1. RSI (14)  — Relative Strength Index ───
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            # 0 나누기 방지
            rs = gain / loss.replace(0, np.nan)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]

            # ─── 2. MACD (12, 26, 9) ───
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = (macd_line - signal_line).iloc[-1]

            # ─── 3. Bollinger %B (20, 2) ───
            sma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            upper = sma20 + (std20 * 2)
            lower = sma20 - (std20 * 2)
            band_width = upper - lower
            bb_pct = (((close - lower) / band_width.replace(0, np.nan)) * 100).iloc[-1]

            # ─── 4. Stochastic %K (14, 3) ───
            low14  = low.rolling(window=14).min()
            high14 = high.rolling(window=14).max()
            stoch_range = high14 - low14
            stoch_k = (100 * ((close - low14) / stoch_range.replace(0, np.nan))).iloc[-1]

            # ─── 5. MA Trend (20일선 vs 60일선) ───
            ma20 = sma20.iloc[-1]
            ma60 = close.rolling(window=60).mean().iloc[-1]
            if pd.isna(ma60):
                trend = "데이터 부족"
            else:
                gap_pct = ((ma20 - ma60) / ma60) * 100
                if gap_pct > 2:    trend = "강한 정배열"
                elif gap_pct > 0:  trend = "정배열 상승"
                elif gap_pct > -2: trend = "역배열 약세"
                else:              trend = "강한 역배열"

            # ─── 6. ATR (14)  — Average True Range, 변동성 지표 ───
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            atr_pct = (atr / close.iloc[-1]) * 100  # 가격 대비 변동성 비율

            return {
                "rsi":      round(rsi, 1)      if not pd.isna(rsi)      else 50.0,
                "macd":     round(macd_hist, 3) if not pd.isna(macd_hist) else 0.0,
                "bb":       round(bb_pct, 1)   if not pd.isna(bb_pct)   else 50.0,
                "stoch_k":  round(stoch_k, 1)  if not pd.isna(stoch_k)  else 50.0,
                "ma_trend": trend,
                "atr_pct":  round(atr_pct, 2)  if not pd.isna(atr_pct)  else 0.0,
            }

        except Exception as e:
            logger.error(f"  ⚠️  기술적 지표 연산 실패: {e}")
            return {
                "rsi": 50.0, "macd": 0.0, "bb": 50.0,
                "stoch_k": 50.0, "ma_trend": "계산 오류", "atr_pct": 0.0
            }

    @staticmethod
    def sparkline(df: pd.DataFrame, n: int = SPARKLINE_POINTS) -> List[float]:
        """차트용 종가 시계열 (최근 N개)."""
        if df.empty:
            return []
        closes = df['Close'].tail(n).tolist()
        return [round(float(v), 4) for v in closes if not pd.isna(v)]


# ═══════════════════════════════════════════════════════════════════════════
#  [5] 규칙 기반 NLP 코멘트 엔진 (Gemini 폴백용)
# ═══════════════════════════════════════════════════════════════════════════
class RuleBasedCommentary:
    """Gemini 실패 시에도 자산별 의미 있는 코멘트를 보장하는 폴백 엔진."""

    @staticmethod
    def for_asset(name: str, change_pct: float, tech: Dict, is_inverse: bool) -> str:
        rsi  = tech['rsi']
        bb   = tech['bb']
        macd = tech['macd']
        atr  = tech['atr_pct']

        # ─── 매크로 역상관 자산 (환율/VIX/금리) ───
        if is_inverse:
            if change_pct > 1.5:
                return f"🚨 {name}이(가) +{change_pct}% 급등하며 시장에 강한 부담을 주고 있습니다. 외국인 자본 유출 압력에 대비한 보수적 포지션이 필요한 시점입니다."
            if change_pct < -1.5:
                return f"🌤️ {name}이(가) {change_pct}% 안정화되며 억눌렸던 투자 심리가 해소되고 있습니다. 위험자산 선호 분위기가 확산될 가능성이 높습니다."
            return f"⚖️ {name}은(는) {change_pct}% 변동으로 큰 흐름 없이 횡보 중이며, 시장의 방향성 탐색을 유도하고 있습니다."

        # ─── 일반 주식/지수 (다중 조건 교차 검증) ───
        base = f"전일 대비 {'+' if change_pct >= 0 else ''}{change_pct}% 변동. "

        # 극단 과열
        if rsi >= 75 and bb >= 100:
            return base + "RSI와 볼린저밴드 모두 '극단적 과열' 신호. 차익 실현 욕구가 임계치에 도달한 위험 구간입니다. 🔥"
        # 극단 과매도
        if rsi <= 25 and bb <= 0:
            return base + "투매로 인한 '극단적 과매도' 상태. 지지선 근처에서 기술적 반등 가능성이 높은 줍줍 찬스 구간입니다. 💎"
        # 강한 정배열 + MACD 상승
        if macd > 0 and "정배열" in tech['ma_trend']:
            return base + "이동평균선 정배열과 MACD 상승 모멘텀이 결합되어 강력한 주도주 흐름을 보이고 있습니다. 🚀"
        # 강한 역배열 + MACD 하락
        if macd < 0 and "역배열" in tech['ma_trend']:
            return base + "추세가 무너진 상태에서 하락 모멘텀이 가중되고 있습니다. 무리한 매수보다 바닥 확인이 우선입니다. 📉"
        # 변동성 폭발
        if atr > 3.5:
            return base + f"변동성(ATR)이 {atr}%로 매우 높아 단기 트레이딩 리스크가 큰 구간입니다. ⚡"
        # 중립
        return base + "상승과 하락 모멘텀이 팽팽하게 균형을 이루고 있습니다. 전체 시장 온도에 순응하는 전략이 유리합니다. 🧩"


# ═══════════════════════════════════════════════════════════════════════════
#  [6] 오라클 매크로 예측 모델
# ═══════════════════════════════════════════════════════════════════════════
class MacroOracleModel:
    """
    선행 매크로 지표를 가중 합산하여 KOSPI 시초가 방향성을 예측.

    핵심 알고리즘:
      1) 변동률 클리핑 (±3%) → 극단값 왜곡 방지
      2) 비선형 모멘텀 가중치 (|변동| > 1.5% 시 1.3배)
      3) Bull/Bear 에너지 분리 → 비율로 0~100점 정규화
      4) 시장 온도계 (Market Breadth) 별도 산출
    """

    CLIP_RANGE       = 3.0      # ±3% 변동률 클리핑
    NONLINEAR_THRESH = 1.5      # 비선형 가중 발동 임계치
    NONLINEAR_BOOST  = 1.3      # 비선형 가중 배율

    def __init__(self):
        self.bull_energy: float = 0.0
        self.bear_energy: float = 0.0
        self.impact_logs: List[Dict] = []
        self.risk_assets_total: int = 0
        self.risk_assets_up: int = 0

    def process_macro_factor(self, name: str, change_pct: float, weight: float, is_inverse: bool) -> None:
        """매크로 지표의 가중 임팩트 계산 및 누적."""
        # 1) 극단값 클리핑
        clipped = max(min(change_pct, self.CLIP_RANGE), -self.CLIP_RANGE)

        # 2) 기본 임팩트
        impact = clipped * weight

        # 3) 역상관 자산은 부호 반전
        if is_inverse:
            impact = -impact

        # 4) 비선형 모멘텀 가중 (급등락 시 영향력 증폭)
        if abs(change_pct) > self.NONLINEAR_THRESH:
            impact *= self.NONLINEAR_BOOST

        self.impact_logs.append({
            "name":       name,
            "change":     change_pct,
            "weight":     weight,
            "is_inverse": is_inverse,
            "impact":     round(impact, 4),
        })

        if impact > 0:
            self.bull_energy += impact
        else:
            self.bear_energy += abs(impact)

    def process_market_breadth(self, category: str, change_pct: float) -> None:
        """시장 온도계 산출용 위험 자산 카운팅."""
        if category in ("Korea", "USA", "Asia", "Europe"):
            self.risk_assets_total += 1
            if change_pct > 0:
                self.risk_assets_up += 1

    def finalize(self) -> Tuple[int, int, str, Dict]:
        """오라클 스코어 + 시장 온도 + 규칙 기반 리포트 산출."""
        # ─── 오라클 스코어 정규화 (0~100) ───
        total_energy = self.bull_energy + self.bear_energy
        if total_energy < 0.01:
            oracle_score = 50  # 에너지 거의 없음 = 중립
        else:
            oracle_score = round((self.bull_energy / total_energy) * 100)

        # ─── 시장 온도계 (0~100) ───
        if self.risk_assets_total > 0:
            market_temp = round((self.risk_assets_up / self.risk_assets_total) * 100)
        else:
            market_temp = 50

        # ─── 핵심 요인 식별 ───
        best  = max(self.impact_logs, key=lambda x: x['impact']) if self.impact_logs else None
        worst = min(self.impact_logs, key=lambda x: x['impact']) if self.impact_logs else None

        # ─── 규칙 기반 리포트 작성 ───
        report = self._compose_rule_based_report(oracle_score, market_temp, best, worst)

        # ─── 메타데이터 ───
        meta = {
            "bull_energy":       round(self.bull_energy, 3),
            "bear_energy":       round(self.bear_energy, 3),
            "best_factor":       best,
            "worst_factor":      worst,
            "risk_assets_total": self.risk_assets_total,
            "risk_assets_up":    self.risk_assets_up,
            "all_impacts":       sorted(self.impact_logs, key=lambda x: abs(x['impact']), reverse=True),
        }

        return oracle_score, market_temp, report, meta

    def _compose_rule_based_report(self, score: int, temp: int,
                                    best: Optional[Dict], worst: Optional[Dict]) -> str:
        """동적 서술형 리포트 생성 (Gemini 페일오버 시 사용)."""
        lines = ["💡 [알고리즘 분석 결과]"]

        # 1) 예측 설명
        best_name  = best['name']  if best  else "데이터 없음"
        worst_name = worst['name'] if worst else "데이터 없음"

        if score >= 65:
            lines.append(
                f"내일 코스피 시초가는 강한 상승 출발이 예상됩니다. 글로벌 시장에서 "
                f"'{best_name}'의 우호적 흐름이 한국 증시에 강력한 매수 에너지를 공급하고 있습니다."
            )
        elif score >= 55:
            lines.append(
                f"내일 코스피는 완만한 상승 출발이 예상됩니다. '{best_name}'의 긍정적 흐름이 "
                f"외국인 매수세를 자극할 수 있으나, 추세적 상승까지는 추가 확인이 필요합니다."
            )
        elif score <= 35:
            lines.append(
                f"내일 코스피는 하방 압력에 강하게 노출될 전망입니다. "
                f"'{worst_name}'의 불안정한 움직임이 외국인 투심을 위축시킬 우려가 큽니다."
            )
        elif score <= 45:
            lines.append(
                f"내일 코스피는 약세 출발 후 반등을 시도할 가능성이 있습니다. "
                f"'{worst_name}'의 부담이 존재하나, 저가 매수세 유입 여부가 관건입니다."
            )
        else:
            lines.append(
                f"내일 한국 증시는 눈치보기 장세가 전개될 것으로 보입니다. "
                f"상승 요인('{best_name}')과 하락 요인('{worst_name}')이 팽팽하게 맞서고 있어 "
                f"방향성 베팅보다 종목별 차별화 대응이 유리합니다."
            )

        # 2) 시장 온도 해설
        lines.append("")
        lines.append(f"🌡️ [시장 온도: {temp}°C의 의미]")
        if temp <= 30:
            lines.append(
                f"현재 온도가 {temp}도로 매우 차가운 이유는, 전 세계 핵심 주식과 지수의 70% 이상이 "
                f"일제히 파란불(하락)을 켜고 있는 '글로벌 리스크 오프' 상태이기 때문입니다."
            )
        elif temp <= 45:
            lines.append(
                f"현재 온도가 {temp}도로 미지근한 이유는, 위험자산의 절반 이상이 약세를 보이며 "
                f"시장 전반의 매수 의지가 약화된 '관망 우세' 국면이기 때문입니다."
            )
        elif temp >= 70:
            lines.append(
                f"현재 온도가 {temp}도로 뜨거운 이유는, 대륙을 불문하고 추적 중인 글로벌 자산의 "
                f"대다수가 빨간불(상승)을 기록하며 '리스크 온' 매수세가 유입되고 있기 때문입니다."
            )
        elif temp >= 55:
            lines.append(
                f"현재 온도가 {temp}도로 따뜻한 이유는, 위험자산 대다수가 강세 흐름을 보이며 "
                f"투자 심리가 점진적으로 회복되고 있기 때문입니다."
            )
        else:
            lines.append(
                f"현재 온도가 {temp}도로 중립적인 이유는, 특정 섹터나 국가만 차별적으로 움직이는 "
                f"순환매 장세가 진행 중이기 때문입니다."
            )

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  [7] Gemini AI 통합 모듈
# ═══════════════════════════════════════════════════════════════════════════
class GeminiOracle:
    """
    Gemini AI로 규칙 기반 결과를 자연어로 재해석.
    실패 시 None 반환 → 규칙 기반 결과가 그대로 살아남음.
    """

    @staticmethod
    def is_available() -> bool:
        return bool(GEMINI_API_KEY)

    @staticmethod
    def enhance_report(score: int, temp: int, meta: Dict, assets_summary: List[Dict]) -> Optional[Dict]:
        """규칙 기반 결과를 Gemini가 재작성하여 더 풍부한 리포트 생성."""
        if not GeminiOracle.is_available():
            logger.warning("  ⚠️  GEMINI_API_KEY 미설정 → AI 강화 스킵")
            return None

        # ─── 프롬프트 입력용 데이터 직렬화 ───
        impact_table = "\n".join([
            f"  · {x['name']}: 변동 {x['change']:+.2f}% × 가중치 {x['weight']} = 임팩트 {x['impact']:+.4f}"
            for x in meta['all_impacts'][:8]
        ])

        top_movers = sorted(assets_summary, key=lambda x: abs(x.get('change', 0)), reverse=True)[:6]
        movers_table = "\n".join([
            f"  · {x['name']}: {x['change']:+.2f}% (RSI {x['tech']['rsi']}, {x['tech']['ma_trend']})"
            for x in top_movers
        ])

        kst_str = NOW_KST.strftime("%Y-%m-%d %H:%M KST")

        # ─── Gemini 프롬프트 (전문가 페르소나) ───
        prompt = f"""당신은 한국 증시 전문 매크로 애널리스트입니다. 다음 정량 데이터를 바탕으로 토스증권 스타일의 친절하고 전문적인 시장 분석 리포트를 작성하세요.

═══════════════════════════════════════════════
[분석 시각] {kst_str}

[1] 알고리즘 오라클 스코어: {score}점 (50점 중립, 100점에 가까울수록 상승)
[2] 시장 온도: {temp}°C (50도 중립, 100도에 가까울수록 위험자산 강세)

[3] 매크로 지표 임팩트 (가중치 적용)
{impact_table}

[4] 변동성 상위 자산 (TOP 6)
{movers_table}

[5] 위험자산 추세: {meta['risk_assets_up']}개 상승 / {meta['risk_assets_total']}개 중
═══════════════════════════════════════════════

다음 형식으로 응답하세요. 반드시 JSON 블록을 먼저 작성하고, 다른 설명은 일절 추가하지 마세요.

<REPORT>
{{
  "headline": "내일 KOSPI 방향성을 한 문장으로 (예: '내일 코스피는 강한 상승세가 예상됩니다 🚀')",
  "main_report": "💡 [알고리즘 분석 결과]\\n(2~3문장으로 예측 근거 설명)\\n\\n🌡️ [시장 온도: {temp}°C의 의미]\\n(2문장으로 현재 시장 분위기 설명)\\n\\n🎯 [투자자 행동 가이드]\\n(2문장으로 구체적 전략 제안)",
  "key_drivers": ["핵심 상승 요인 1", "핵심 상승 요인 2"],
  "key_risks": ["주요 리스크 1", "주요 리스크 2"],
  "confidence": "HIGH" 또는 "MEDIUM" 또는 "LOW"
}}
</REPORT>

작성 원칙:
1. 한국어로 작성
2. 토스증권처럼 친절하고 명확한 어조
3. 데이터에 근거한 구체적 분석 (추상적 표현 금지)
4. 매크로 지표의 임팩트를 반드시 언급
5. main_report는 \\n 으로 줄바꿈, 이모지 적극 활용"""

        # ─── API 호출 ───
        try:
            url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature":     0.4,   # 낮을수록 일관된 분석
                    "maxOutputTokens": 1500,
                    "topP":            0.9,
                },
            }

            logger.info("  🤖 Gemini AI 리포트 재작성 요청...")
            res = requests.post(url, json=payload, timeout=30)

            if res.status_code != 200:
                logger.error(f"  ❌ Gemini API HTTP {res.status_code}: {res.text[:200]}")
                return None

            data = res.json()
            text = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')

            if not text:
                logger.error("  ❌ Gemini 빈 응답")
                return None

            # ─── JSON 블록 추출 ───
            match = re.search(r'<REPORT>(.*?)</REPORT>', text, re.DOTALL)
            if not match:
                logger.warning("  ⚠️  Gemini 응답에서 <REPORT> 블록 없음 → 전체 텍스트 사용")
                return {"raw_text": text.strip()}

            json_str = match.group(1).strip()
            try:
                parsed = json.loads(json_str)
                logger.info(f"  ✅ Gemini 리포트 생성 완료 (신뢰도: {parsed.get('confidence', '?')})")
                return parsed
            except json.JSONDecodeError as e:
                logger.error(f"  ❌ Gemini JSON 파싱 실패: {e}")
                logger.debug(f"     원본: {json_str[:300]}")
                return {"raw_text": json_str}

        except requests.Timeout:
            logger.error("  ❌ Gemini 타임아웃 (30초 초과)")
            return None
        except Exception as e:
            logger.error(f"  ❌ Gemini 호출 오류: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
#  [8] 히스토리 관리 (당일 누적 + 자정 초기화)
# ═══════════════════════════════════════════════════════════════════════════
class HistoryManager:
    """일중 시계열 누적 (sparkline 풍부화 + 변동 추적)."""

    @staticmethod
    def load() -> Dict:
        """기존 히스토리 로드. 날짜 바뀌면 자동 초기화."""
        if not HISTORY_FILE.exists():
            return {"date": TODAY_KEY, "snapshots": []}

        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                hist = json.load(f)

            # 날짜 바뀌었으면 초기화
            if hist.get('date') != TODAY_KEY:
                logger.info(f"  🔄 날짜 변경 감지 ({hist.get('date')} → {TODAY_KEY}). 히스토리 초기화.")
                return {"date": TODAY_KEY, "snapshots": []}

            return hist
        except Exception as e:
            logger.warning(f"  ⚠️  히스토리 로드 실패: {e} → 새로 시작")
            return {"date": TODAY_KEY, "snapshots": []}

    @staticmethod
    def append(history: Dict, assets: List[Dict], oracle_score: int, market_temp: int) -> Dict:
        """현재 스냅샷을 히스토리에 추가 (최대 144개 = 24시간 × 6회/시간)."""
        snapshot = {
            "t":      NOW_KST.strftime("%H:%M"),
            "ts":     int(NOW_KST.timestamp()),
            "score":  oracle_score,
            "temp":   market_temp,
            "prices": {a['symbol']: a['price'] for a in assets},
        }
        history['snapshots'].append(snapshot)

        # 최대 144개 유지 (10분 × 144 = 24시간)
        if len(history['snapshots']) > 144:
            history['snapshots'] = history['snapshots'][-144:]

        return history

    @staticmethod
    def save(history: Dict) -> None:
        """히스토리 디스크 저장."""
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            logger.info(f"  💾 history.json 저장 (스냅샷 {len(history['snapshots'])}개)")
        except Exception as e:
            logger.error(f"  ❌ history.json 저장 실패: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  [9] 예측 로그 관리 (사후 적중률 검증용)
# ═══════════════════════════════════════════════════════════════════════════
class PredictionLogger:
    """매일 첫 예측을 별도 로그로 보관 → 나중에 실제 KOSPI 결과와 비교."""

    @staticmethod
    def log_if_first_today(oracle_score: int, market_temp: int, gemini_result: Optional[Dict]) -> None:
        """오늘 첫 예측이면 prediction_log.json에 기록."""
        try:
            if PREDICTION_LOG_FILE.exists():
                with open(PREDICTION_LOG_FILE, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []

            # 오늘 이미 로그가 있으면 스킵
            if logs and logs[-1].get('date') == TODAY_KEY:
                return

            entry = {
                "date":         TODAY_KEY,
                "first_logged": NOW_KST.strftime("%H:%M"),
                "score":        oracle_score,
                "temp":         market_temp,
                "headline":     gemini_result.get('headline') if gemini_result else None,
                "confidence":   gemini_result.get('confidence') if gemini_result else None,
                "actual_kospi_close_pct": None,  # 익일 수동/자동 채워질 필드
            }
            logs.append(entry)

            # 최근 90일치만 유지
            logs = logs[-90:]

            with open(PREDICTION_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)

            logger.info(f"  📝 오늘 첫 예측 로그 기록 (스코어: {oracle_score})")
        except Exception as e:
            logger.warning(f"  ⚠️  예측 로그 저장 실패: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  [10] 메인 오케스트레이터
# ═══════════════════════════════════════════════════════════════════════════
def execute_quant_terminal() -> None:
    """전체 파이프라인 오케스트레이션."""
    logger.info("═" * 60)
    logger.info("🚀 KOSPI ORACLE V7 — OMNIPOTENT ENGINE START")
    logger.info(f"   시작 시각: {NOW_KST.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"   Gemini AI: {'활성화 ✅' if GeminiOracle.is_available() else '비활성화 (폴백 모드)'}")
    logger.info("═" * 60)

    oracle = MacroOracleModel()
    processed_assets: List[Dict] = []
    failed_symbols: List[str] = []

    # ─── 자산별 순차 수집 + 분석 ───
    total = len(ASSETS)
    for idx, (sym, info) in enumerate(ASSETS.items(), 1):
        logger.info(f"[{idx:2d}/{total}] 📡 {info['name']} ({sym}) 수집...")

        hist = YFinanceFetcher.fetch(sym)
        if hist.empty:
            failed_symbols.append(sym)
            continue

        try:
            # 기본 가격 데이터
            close_series = hist['Close']
            cur_price  = float(close_series.iloc[-1])
            prev_price = float(close_series.iloc[-2])
            change_pct = round(((cur_price - prev_price) / prev_price) * 100, 2)

            # 기술적 지표
            tech_data = TechnicalIndicator.calculate(hist)
            sparkline = TechnicalIndicator.sparkline(hist)

            # 규칙 기반 코멘트 (Gemini와 무관하게 항상 보존)
            commentary = RuleBasedCommentary.for_asset(
                info['name'], change_pct, tech_data, info.get('inv', False)
            )

            # 오라클 모델에 데이터 주입
            if info['cat'] == 'Macro_Predict':
                oracle.process_macro_factor(
                    info['name'], change_pct, info['w'], info.get('inv', False)
                )
            oracle.process_market_breadth(info['cat'], change_pct)

            # 결과 저장
            processed_assets.append({
                "symbol":    sym,
                "name":      info['name'],
                "cat":       info['cat'],
                "price":     round(cur_price, 2) if cur_price > 100 else round(cur_price, 4),
                "change":    change_pct,
                "tech":      tech_data,
                "comment":   commentary,
                "sparkline": sparkline,
            })

            change_emoji = "🔺" if change_pct > 0 else "🔻" if change_pct < 0 else "➖"
            logger.info(f"        {change_emoji} {change_pct:+.2f}% │ RSI {tech_data['rsi']} │ {tech_data['ma_trend']}")

        except Exception as e:
            logger.error(f"  ❌ {sym} 가공 중 오류: {e}")
            failed_symbols.append(sym)

    # ─── 정렬 (변동률 절대값 내림차순 → 변동 큰 자산이 위로) ───
    processed_assets.sort(key=lambda x: abs(x['change']), reverse=True)

    # ─── 오라클 최종 산출 ───
    logger.info("─" * 60)
    logger.info("🔮 오라클 모델 최종 산출...")
    final_score, final_temp, rule_report, oracle_meta = oracle.finalize()
    logger.info(f"   📊 오라클 스코어: {final_score}점")
    logger.info(f"   🌡️  시장 온도:    {final_temp}°C")
    logger.info(f"   ⚡ Bull/Bear:    {oracle_meta['bull_energy']} / {oracle_meta['bear_energy']}")

    # ─── Gemini AI 강화 (실패해도 안전) ───
    logger.info("─" * 60)
    gemini_result = GeminiOracle.enhance_report(final_score, final_temp, oracle_meta, processed_assets)

    # ─── 히스토리 누적 ───
    logger.info("─" * 60)
    history = HistoryManager.load()
    history = HistoryManager.append(history, processed_assets, final_score, final_temp)
    HistoryManager.save(history)

    # ─── 예측 로그 (오늘 첫 예측만) ───
    PredictionLogger.log_if_first_today(final_score, final_temp, gemini_result)

    # ─── 최종 payload 구성 ───
    payload = {
        "version":     "V7.OMNIPOTENT",
        "kst":         NOW_KST.strftime("%Y-%m-%d %H:%M:%S"),
        "ts":          int(NOW_KST.timestamp()),
        "market_temp": final_temp,

        "prediction": {
            "score":        final_score,
            "report":       (gemini_result.get('main_report') if gemini_result else rule_report),
            "headline":     gemini_result.get('headline')    if gemini_result else None,
            "key_drivers":  gemini_result.get('key_drivers') if gemini_result else [],
            "key_risks":    gemini_result.get('key_risks')   if gemini_result else [],
            "confidence":   gemini_result.get('confidence')  if gemini_result else "MEDIUM",
            "ai_enhanced":  gemini_result is not None,
            "rule_report":  rule_report,                       # 폴백용 원본 보존
            "disclaimer":   "본 분석은 통계 기반 [가상의 모델링 자료]이며, 투자 결과에 대한 법적 책임을 지지 않습니다.",
        },

        "oracle_meta": {
            "bull_energy":  oracle_meta['bull_energy'],
            "bear_energy":  oracle_meta['bear_energy'],
            "best_factor":  oracle_meta['best_factor'],
            "worst_factor": oracle_meta['worst_factor'],
            "all_impacts":  oracle_meta['all_impacts'],
        },

        "stats": {
            "total_assets":  total,
            "success":       len(processed_assets),
            "failed":        len(failed_symbols),
            "failed_symbols": failed_symbols,
            "history_count": len(history['snapshots']),
        },

        "data": processed_assets,
    }

    # ─── data.json 저장 ───
    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("═" * 60)
        logger.info(f"🎉 data.json 빌드 완료 ({len(processed_assets)}/{total} 자산)")
        logger.info(f"   AI 강화: {'✅ Gemini' if gemini_result else '❌ 규칙 기반 폴백'}")
        logger.info("═" * 60)
    except Exception as e:
        logger.critical(f"💥 치명적 오류 - data.json 저장 실패: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════
#  [11] 엔트리 포인트
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        execute_quant_terminal()
    except KeyboardInterrupt:
        logger.warning("⚠️ 사용자 중단")
    except Exception as e:
        logger.critical(f"💥 파이프라인 치명적 오류: {e}", exc_info=True)
        raise
