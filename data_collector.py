"""
═══════════════════════════════════════════════════════════════════════════
  KOSPI ORACLE V8 — INDIVIDUAL STOCK PREDICTION ENGINE
  ───────────────────────────────────────────────────────────────────────
  ▸ 글로벌 매크로 자산 41종 수집 (KOSPI 20 + KOSDAQ 5 + 글로벌 16)
  ▸ 6중 기술적 지표 (RSI, MACD, BB %B, Stochastic, MA, ATR)
  ▸ KOSPI 시초가 오라클 예측 모델
  ▸ Gemini AI 자연어 리포트 재작성
  ▸ ★ 종목별 Beta 모델 (90일 회귀분석 → KOSPI 민감도 계산)
  ▸ ★ 섹터별 가중치 보정 (반도체/자동차/바이오/방산/배터리/금융 등)
  ▸ ★ 기술적 지표 보정 (RSI 과열/과매도 자동 감쇄/부스트)
  ▸ 시계열 누적 + 예측 히스토리 로깅
  ───────────────────────────────────────────────────────────────────────
  Author : Jinju First Girls' High School — 태경
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
#  [1] 환경 설정
# ═══════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [ %(levelname)-7s ] %(name)-18s │ %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("ORACLE-V8")

BASE_DIR              = Path(__file__).parent
DATA_FILE             = BASE_DIR / "data.json"
HISTORY_FILE          = BASE_DIR / "history.json"
PREDICTION_LOG_FILE   = BASE_DIR / "prediction_log.json"

GEMINI_API_KEY        = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL          = "gemini-1.5-flash"
GEMINI_ENDPOINT       = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

KST                   = pytz.timezone('Asia/Seoul')
NOW_KST               = datetime.now(KST)
TODAY_KEY             = NOW_KST.strftime("%Y-%m-%d")

HIST_PERIOD           = "6mo"
SPARKLINE_POINTS      = 30
MAX_RETRIES           = 3
RETRY_BACKOFF_BASE    = 1.5
BETA_WINDOW_DAYS      = 90

# ═══════════════════════════════════════════════════════════════════════════
#  [2] 자산 유니버스 (V8 — 41종)
# ═══════════════════════════════════════════════════════════════════════════
# sector: 섹터별 가중치 보정용
#   - 'semiconductor'   : 반도체 (NQ=F, SOXX 추가 가중)
#   - 'auto'            : 자동차 (KRW=X 역가중)
#   - 'battery'         : 2차전지 (CL=F 가중)
#   - 'bio'             : 바이오/제약 (^IXIC 가중)
#   - 'defense_nuclear' : 방산/원전 (^VIX 가중)
#   - 'finance'         : 금융 (^TNX 가중)
#   - 'platform'        : IT 플랫폼 (^IXIC 가중)
#   - 'shipbuilding'    : 조선 (KRW=X 역가중)
#   - 'general'         : 일반
# ═══════════════════════════════════════════════════════════════════════════

ASSETS: Dict[str, Dict] = {
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🔴 [Macro_Predict] KOSPI 시초가 예측용 핵심 선행 지표
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "NQ=F":      {"name": "나스닥 100 선물",      "cat": "Macro_Predict", "w": 0.22, "inv": False, "sector": "general"},
    "SOXX":      {"name": "필라델피아 반도체 ETF",  "cat": "Macro_Predict", "w": 0.22, "inv": False, "sector": "general"},
    "KRW=X":     {"name": "원/달러 환율",          "cat": "Macro_Predict", "w": 0.18, "inv": True,  "sector": "general"},
    "^VIX":      {"name": "VIX 공포지수",          "cat": "Macro_Predict", "w": 0.13, "inv": True,  "sector": "general"},
    "^TNX":      {"name": "미 국채 10년물",        "cat": "Macro_Predict", "w": 0.10, "inv": True,  "sector": "general"},
    "^GDAXI":    {"name": "독일 DAX 지수",         "cat": "Macro_Predict", "w": 0.05, "inv": False, "sector": "general"},
    "CL=F":      {"name": "WTI 유가",              "cat": "Macro_Predict", "w": 0.05, "inv": False, "sector": "general"},
    "GC=F":      {"name": "금 선물",                "cat": "Macro_Predict", "w": 0.05, "inv": False, "sector": "general"},

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🇰🇷 대한민국 — KOSPI 지수 + 시총 상위 20종 + KOSDAQ 상위 5종
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "^KS11":     {"name": "KOSPI 종합지수",      "cat": "Korea", "w": 0, "inv": False, "sector": "index"},
    "^KQ11":     {"name": "KOSDAQ 종합지수",     "cat": "Korea", "w": 0, "inv": False, "sector": "index"},

    # ─── KOSPI 시총 상위 20종 (2026.05 기준) ───
    "005930.KS": {"name": "삼성전자",             "cat": "Korea", "w": 0, "inv": False, "sector": "semiconductor"},
    "000660.KS": {"name": "SK하이닉스",           "cat": "Korea", "w": 0, "inv": False, "sector": "semiconductor"},
    "402340.KS": {"name": "SK스퀘어",             "cat": "Korea", "w": 0, "inv": False, "sector": "semiconductor"},
    "373220.KS": {"name": "LG에너지솔루션",       "cat": "Korea", "w": 0, "inv": False, "sector": "battery"},
    "207940.KS": {"name": "삼성바이오로직스",     "cat": "Korea", "w": 0, "inv": False, "sector": "bio"},
    "005380.KS": {"name": "현대차",               "cat": "Korea", "w": 0, "inv": False, "sector": "auto"},
    "034020.KS": {"name": "두산에너빌리티",       "cat": "Korea", "w": 0, "inv": False, "sector": "defense_nuclear"},
    "012450.KS": {"name": "한화에어로스페이스",   "cat": "Korea", "w": 0, "inv": False, "sector": "defense_nuclear"},
    "006400.KS": {"name": "삼성SDI",              "cat": "Korea", "w": 0, "inv": False, "sector": "battery"},
    "000270.KS": {"name": "기아",                 "cat": "Korea", "w": 0, "inv": False, "sector": "auto"},
    "068270.KS": {"name": "셀트리온",             "cat": "Korea", "w": 0, "inv": False, "sector": "bio"},
    "035420.KS": {"name": "NAVER",                "cat": "Korea", "w": 0, "inv": False, "sector": "platform"},
    "035720.KS": {"name": "카카오",               "cat": "Korea", "w": 0, "inv": False, "sector": "platform"},
    "105560.KS": {"name": "KB금융",               "cat": "Korea", "w": 0, "inv": False, "sector": "finance"},
    "055550.KS": {"name": "신한지주",             "cat": "Korea", "w": 0, "inv": False, "sector": "finance"},
    "086790.KS": {"name": "하나금융지주",         "cat": "Korea", "w": 0, "inv": False, "sector": "finance"},
    "028260.KS": {"name": "삼성물산",             "cat": "Korea", "w": 0, "inv": False, "sector": "general"},
    "009540.KS": {"name": "HD한국조선해양",       "cat": "Korea", "w": 0, "inv": False, "sector": "shipbuilding"},
    "329180.KS": {"name": "HD현대중공업",         "cat": "Korea", "w": 0, "inv": False, "sector": "shipbuilding"},
    "009150.KS": {"name": "삼성전기",             "cat": "Korea", "w": 0, "inv": False, "sector": "semiconductor"},

    # ─── KOSDAQ 시총 상위 5종 ───
    "247540.KQ": {"name": "에코프로비엠",         "cat": "Korea", "w": 0, "inv": False, "sector": "battery"},
    "086520.KQ": {"name": "에코프로",             "cat": "Korea", "w": 0, "inv": False, "sector": "battery"},
    "196170.KQ": {"name": "알테오젠",             "cat": "Korea", "w": 0, "inv": False, "sector": "bio"},
    "028300.KQ": {"name": "HLB",                  "cat": "Korea", "w": 0, "inv": False, "sector": "bio"},
    "058470.KQ": {"name": "리노공업",             "cat": "Korea", "w": 0, "inv": False, "sector": "semiconductor"},

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🇺🇸 미국 — 4대 지수 + 빅테크 4종
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "^GSPC":     {"name": "S&P 500",              "cat": "USA", "w": 0, "inv": False, "sector": "index"},
    "^IXIC":     {"name": "나스닥 종합",          "cat": "USA", "w": 0, "inv": False, "sector": "index"},
    "^DJI":      {"name": "다우존스",             "cat": "USA", "w": 0, "inv": False, "sector": "index"},
    "^RUT":      {"name": "러셀 2000",            "cat": "USA", "w": 0, "inv": False, "sector": "index"},
    "NVDA":      {"name": "엔비디아",             "cat": "USA", "w": 0, "inv": False, "sector": "general"},
    "MSFT":      {"name": "마이크로소프트",       "cat": "USA", "w": 0, "inv": False, "sector": "general"},
    "AAPL":      {"name": "애플",                 "cat": "USA", "w": 0, "inv": False, "sector": "general"},
    "TSLA":      {"name": "테슬라",               "cat": "USA", "w": 0, "inv": False, "sector": "general"},

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🌏 아시아 — 5종 (일본 2 + 중국/홍콩 2 + 대만/인도)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "^N225":     {"name": "닛케이 225",           "cat": "Asia", "w": 0, "inv": False, "sector": "index"},
    "^TOPX":     {"name": "토픽스 (TOPIX)",       "cat": "Asia", "w": 0, "inv": False, "sector": "index"},
    "^HSI":      {"name": "항셍 지수",            "cat": "Asia", "w": 0, "inv": False, "sector": "index"},
    "000001.SS": {"name": "상하이 종합",          "cat": "Asia", "w": 0, "inv": False, "sector": "index"},
    "^TWII":     {"name": "대만 가권지수",         "cat": "Asia", "w": 0, "inv": False, "sector": "index"},
    "^BSESN":    {"name": "인도 SENSEX",          "cat": "Asia", "w": 0, "inv": False, "sector": "index"},

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 🇪🇺 유럽 — 4종
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    "^FTSE":     {"name": "영국 FTSE 100",        "cat": "Europe", "w": 0, "inv": False, "sector": "index"},
    "^FCHI":     {"name": "프랑스 CAC 40",        "cat": "Europe", "w": 0, "inv": False, "sector": "index"},
    "^STOXX50E": {"name": "유로스톡스 50",        "cat": "Europe", "w": 0, "inv": False, "sector": "index"},
    "^SSMI":     {"name": "스위스 SMI",           "cat": "Europe", "w": 0, "inv": False, "sector": "index"},
}


# ═══════════════════════════════════════════════════════════════════════════
#  [3] 섹터별 추가 가중치 매핑
#      ▸ KOSPI 예측치만으로 부족한 섹터별 고유 영향을 보정
# ═══════════════════════════════════════════════════════════════════════════
SECTOR_DRIVERS: Dict[str, List[Dict]] = {
    'semiconductor': [
        {'symbol': 'SOXX',   'weight': 0.55, 'invert': False},
        {'symbol': 'NQ=F',   'weight': 0.30, 'invert': False},
        {'symbol': 'KRW=X',  'weight': 0.15, 'invert': True},   # 환율 상승 → 반도체 수출주 유리
    ],
    'auto': [
        {'symbol': 'KRW=X',  'weight': 0.50, 'invert': True},   # 원화 약세 → 수출 유리
        {'symbol': '^GSPC',  'weight': 0.30, 'invert': False},
        {'symbol': 'CL=F',   'weight': 0.20, 'invert': True},   # 유가 상승 → 자동차 수요 감소
    ],
    'battery': [
        {'symbol': '^IXIC',  'weight': 0.40, 'invert': False},
        {'symbol': 'CL=F',   'weight': 0.30, 'invert': True},   # 유가 하락 → 전기차 매력↓
        {'symbol': 'TSLA',   'weight': 0.30, 'invert': False},  # 테슬라 직접 연동
    ],
    'bio': [
        {'symbol': '^IXIC',  'weight': 0.50, 'invert': False},
        {'symbol': '^VIX',   'weight': 0.30, 'invert': True},
        {'symbol': '^TNX',   'weight': 0.20, 'invert': True},   # 금리 하락 → 성장주 유리
    ],
    'defense_nuclear': [
        {'symbol': '^VIX',   'weight': 0.45, 'invert': False},  # 공포지수 ↑ → 방산 수혜
        {'symbol': 'CL=F',   'weight': 0.25, 'invert': False},  # 유가 ↑ → 지정학 긴장
        {'symbol': 'GC=F',   'weight': 0.30, 'invert': False},  # 금 ↑ → 안전자산 선호
    ],
    'finance': [
        {'symbol': '^TNX',   'weight': 0.60, 'invert': False},  # 금리 상승 → 은행주 유리
        {'symbol': '^GSPC',  'weight': 0.40, 'invert': False},
    ],
    'platform': [
        {'symbol': '^IXIC',  'weight': 0.55, 'invert': False},
        {'symbol': '^TNX',   'weight': 0.25, 'invert': True},
        {'symbol': 'NQ=F',   'weight': 0.20, 'invert': False},
    ],
    'shipbuilding': [
        {'symbol': 'KRW=X',  'weight': 0.55, 'invert': True},   # 원화 약세 → 수주가 ↑
        {'symbol': 'CL=F',   'weight': 0.30, 'invert': False},
        {'symbol': '^GSPC',  'weight': 0.15, 'invert': False},
    ],
    'general': [
        {'symbol': '^GSPC',  'weight': 1.0,  'invert': False},
    ],
    'index': [
        {'symbol': '^GSPC',  'weight': 1.0,  'invert': False},
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
#  [4] 데이터 수집기
# ═══════════════════════════════════════════════════════════════════════════
class YFinanceFetcher:
    """야후 파이낸스 호출 안정화 클래스."""

    @staticmethod
    def fetch(symbol: str, period: str = HIST_PERIOD) -> pd.DataFrame:
        for attempt in range(MAX_RETRIES):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period, auto_adjust=True)
                if hist.empty:
                    raise ValueError("응답 비어있음")
                if len(hist) < 60:
                    raise ValueError(f"데이터 부족 ({len(hist)} < 60)")
                if hist['Close'].isna().all():
                    raise ValueError("종가 전부 NaN")
                return hist
            except Exception as e:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(f"  ⚠️  {symbol} 시도 {attempt+1}/{MAX_RETRIES}: {e} → {wait:.1f}초 대기")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(wait)
        logger.error(f"  ❌ {symbol} 최종 수집 실패")
        return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
#  [5] 기술적 지표
# ═══════════════════════════════════════════════════════════════════════════
class TechnicalIndicator:
    """6중 퀀트 지표."""

    @staticmethod
    def calculate(df: pd.DataFrame) -> Dict:
        close = df['Close']
        high  = df['High']
        low   = df['Low']

        try:
            # RSI (14)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = (100 - (100 / (1 + rs))).iloc[-1]

            # MACD (12, 26, 9)
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema12 - ema26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            macd_hist = (macd_line - signal_line).iloc[-1]

            # Bollinger %B (20, 2)
            sma20 = close.rolling(window=20).mean()
            std20 = close.rolling(window=20).std()
            upper = sma20 + (std20 * 2)
            lower = sma20 - (std20 * 2)
            band_width = upper - lower
            bb_pct = (((close - lower) / band_width.replace(0, np.nan)) * 100).iloc[-1]

            # Stochastic %K
            low14  = low.rolling(window=14).min()
            high14 = high.rolling(window=14).max()
            stoch_range = high14 - low14
            stoch_k = (100 * ((close - low14) / stoch_range.replace(0, np.nan))).iloc[-1]

            # MA Trend
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

            # ATR
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]
            atr_pct = (atr / close.iloc[-1]) * 100

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
            return {"rsi": 50.0, "macd": 0.0, "bb": 50.0, "stoch_k": 50.0, "ma_trend": "계산 오류", "atr_pct": 0.0}

    @staticmethod
    def sparkline(df: pd.DataFrame, n: int = SPARKLINE_POINTS) -> List[float]:
        if df.empty:
            return []
        closes = df['Close'].tail(n).tolist()
        return [round(float(v), 4) for v in closes if not pd.isna(v)]


# ═══════════════════════════════════════════════════════════════════════════
#  [6] ★ Beta 계산 엔진 (V8 신규)
# ═══════════════════════════════════════════════════════════════════════════
class BetaCalculator:
    """
    개별 종목의 KOSPI 대비 베타(민감도) 계산.

    Beta = Cov(종목수익률, KOSPI수익률) / Var(KOSPI수익률)

    해석:
      ─ Beta = 1.0  : KOSPI와 동일하게 움직임
      ─ Beta = 1.5  : KOSPI의 1.5배로 민감하게 반응
      ─ Beta = 0.5  : KOSPI 변동의 절반만 반응 (방어주)
      ─ Beta < 0    : KOSPI와 역방향 (보기 드문 케이스)
    """

    @staticmethod
    def calculate(stock_df: pd.DataFrame, kospi_df: pd.DataFrame, window: int = BETA_WINDOW_DAYS) -> Optional[float]:
        try:
            if stock_df.empty or kospi_df.empty:
                return None

            # 종가 시계열에서 일일 수익률 계산
            stock_ret = stock_df['Close'].pct_change().dropna().tail(window)
            kospi_ret = kospi_df['Close'].pct_change().dropna().tail(window)

            # 동일 인덱스(거래일)만 매칭
            aligned = pd.concat([stock_ret, kospi_ret], axis=1, join='inner').dropna()
            if len(aligned) < 30:
                return None

            stock_aligned = aligned.iloc[:, 0]
            kospi_aligned = aligned.iloc[:, 1]

            # Beta = Cov / Var
            covariance = stock_aligned.cov(kospi_aligned)
            variance   = kospi_aligned.var()

            if variance == 0 or pd.isna(variance):
                return None

            beta = covariance / variance

            # 비정상치 클리핑 (-3 ~ +3)
            beta = max(min(beta, 3.0), -3.0)

            return round(beta, 3)
        except Exception as e:
            logger.warning(f"  ⚠️  Beta 계산 실패: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
#  [7] ★ 종목별 예측 모델 (V8 신규)
# ═══════════════════════════════════════════════════════════════════════════
class IndividualStockPredictor:
    """
    KOSPI 예측치 + 종목 베타 + 섹터 가중치 + 기술적 보정으로
    종목별 내일 예상 변동률을 산출.

    ┌─────────────────────────────────────────────────────────┐
    │  Step 1: KOSPI 기대 변동률 추정                            │
    │    score >= 70 → +1.5%                                  │
    │    score 60-70 → +0.8%                                  │
    │    score 55-60 → +0.4%                                  │
    │    score 45-55 → ±0.1%                                  │
    │    score 40-45 → -0.4%                                  │
    │    score 30-40 → -0.8%                                  │
    │    score < 30  → -1.5%                                  │
    │                                                          │
    │  Step 2: Beta 곱셈                                       │
    │    base_pred = kospi_expected * stock_beta              │
    │                                                          │
    │  Step 3: 섹터 가중치 보정                                  │
    │    sector_adj = Σ (driver_change × weight)              │
    │    sector_adj *= 0.3  (전체 영향력 30%로 캡)              │
    │                                                          │
    │  Step 4: 기술적 보정                                       │
    │    RSI ≥ 75 → -25% 감쇄 (과열)                          │
    │    RSI ≤ 25 → +25% 부스트 (과매도 반등)                  │
    │    BB ≥ 100 → -15% 감쇄 (상단 돌파)                     │
    │    BB ≤ 0   → +15% 부스트 (하단 이탈)                    │
    │                                                          │
    │  Step 5: 신뢰도 계산                                       │
    │    Beta가 안정적이고 섹터 시그널이 일치할수록 ↑            │
    └─────────────────────────────────────────────────────────┘
    """

    KOSPI_EXPECTED_MAP = [
        (70, 1.5),
        (60, 0.8),
        (55, 0.4),
        (45, 0.1),
        (40, -0.4),
        (30, -0.8),
        (0,  -1.5),
    ]

    @classmethod
    def kospi_expected(cls, score: int) -> float:
        """오라클 스코어 → KOSPI 예상 변동률(%) 변환."""
        # 점수에 따라 비례 보간 추가 (단순 lookup보다 부드러움)
        if score >= 70:
            return 1.5 + (score - 70) * 0.04   # 70=1.5, 100=2.7
        elif score >= 60:
            return 0.8 + (score - 60) * 0.07
        elif score >= 55:
            return 0.4 + (score - 55) * 0.08
        elif score >= 45:
            return -0.1 + (score - 45) * 0.05
        elif score >= 40:
            return -0.4 + (score - 40) * 0.06
        elif score >= 30:
            return -0.8 + (score - 30) * 0.04
        else:
            return -1.5 + score * 0.02

    @staticmethod
    def predict(stock_data: Dict, beta: Optional[float], kospi_score: int,
                market_data: Dict, sector: str) -> Dict:
        """
        종목별 내일 예상 변동률 및 예상 가격 산출.

        Returns:
            {
                "expected_change_pct": 1.34,
                "expected_price":      75500,
                "confidence":          "HIGH" | "MEDIUM" | "LOW",
                "method":              "beta+sector+tech",
                "components": {
                    "base_kospi_effect":  0.92,
                    "sector_adjustment":  0.38,
                    "tech_modifier":      0.04,
                },
                "beta_used": 1.15,
                "warning": null | "베타 데이터 부족"
            }
        """
        warning = None

        # ─── Step 1: KOSPI 기대 변동률 ───
        kospi_exp = IndividualStockPredictor.kospi_expected(kospi_score)

        # ─── Step 2: Beta 곱셈 ───
        if beta is None or pd.isna(beta):
            # Beta 없으면 섹터 평균 베타 사용 (보수적)
            beta = 1.0
            warning = "베타 데이터 부족 → 평균값 사용"

        base_pred = kospi_exp * beta

        # ─── Step 3: 섹터 가중치 보정 ───
        sector_adj = 0.0
        sector_drivers = SECTOR_DRIVERS.get(sector, SECTOR_DRIVERS['general'])

        for driver in sector_drivers:
            driver_data = market_data.get(driver['symbol'])
            if driver_data and driver_data.get('ok'):
                d_change = driver_data['change']
                if driver['invert']:
                    d_change = -d_change
                sector_adj += d_change * driver['weight']

        # 섹터 보정의 전체 영향력은 30%로 캡
        sector_adj *= 0.3

        # ─── Step 4: 기술적 보정 ───
        tech = stock_data.get('tech', {})
        rsi = tech.get('rsi', 50)
        bb  = tech.get('bb', 50)

        tech_modifier = 1.0
        if rsi >= 75:
            tech_modifier *= 0.75
        elif rsi <= 25:
            tech_modifier *= 1.25
        elif rsi >= 70:
            tech_modifier *= 0.88
        elif rsi <= 30:
            tech_modifier *= 1.12

        if bb >= 100:
            tech_modifier *= 0.85
        elif bb <= 0:
            tech_modifier *= 1.15

        # ─── 최종 예측 ───
        adjusted_base = base_pred * tech_modifier
        final_pred = adjusted_base + sector_adj

        # 비정상치 클리핑 (-8% ~ +8%)
        final_pred = max(min(final_pred, 8.0), -8.0)

        # 예상 가격 계산
        cur_price = stock_data.get('price', 0)
        expected_price = cur_price * (1 + final_pred / 100)

        # ─── Step 5: 신뢰도 계산 ───
        confidence = IndividualStockPredictor._compute_confidence(
            beta, sector_drivers, market_data, kospi_score, rsi
        )

        return {
            "expected_change_pct": round(final_pred, 2),
            "expected_price":      round(expected_price, 2) if cur_price > 100 else round(expected_price, 4),
            "confidence":          confidence,
            "method":              "beta+sector+tech",
            "components": {
                "base_kospi_effect":  round(base_pred, 3),
                "sector_adjustment":  round(sector_adj, 3),
                "tech_modifier":      round((tech_modifier - 1) * 100, 1),  # %로 표시
            },
            "beta_used": round(beta, 3),
            "warning":   warning,
        }

    @staticmethod
    def _compute_confidence(beta: float, sector_drivers: List[Dict],
                            market_data: Dict, kospi_score: int, rsi: float) -> str:
        """예측 신뢰도 등급 산출."""
        score = 50  # 시작점

        # 1) Beta 안정성 (1.0 근처 = 안정적)
        if 0.7 <= abs(beta) <= 1.5:
            score += 15
        elif 0.5 <= abs(beta) <= 2.0:
            score += 8
        else:
            score -= 5

        # 2) 섹터 시그널 일치도
        signs = []
        for driver in sector_drivers:
            d = market_data.get(driver['symbol'])
            if d and d.get('ok'):
                eff_change = -d['change'] if driver['invert'] else d['change']
                signs.append(1 if eff_change > 0 else -1 if eff_change < 0 else 0)
        if len(signs) >= 2:
            unanimity = abs(sum(signs)) / len(signs)
            if unanimity >= 0.8:
                score += 20
            elif unanimity >= 0.5:
                score += 10

        # 3) KOSPI 스코어 극단성 (강한 신호일수록 신뢰도↑)
        if kospi_score >= 70 or kospi_score <= 30:
            score += 15
        elif kospi_score >= 60 or kospi_score <= 40:
            score += 8

        # 4) RSI 극단치는 반대 신호이므로 신뢰도↓
        if rsi >= 80 or rsi <= 20:
            score -= 8

        if score >= 75:
            return "HIGH"
        elif score >= 55:
            return "MEDIUM"
        else:
            return "LOW"


# ═══════════════════════════════════════════════════════════════════════════
#  [8] 규칙 기반 NLP 코멘트
# ═══════════════════════════════════════════════════════════════════════════
class RuleBasedCommentary:
    @staticmethod
    def for_asset(name: str, change_pct: float, tech: Dict, is_inverse: bool,
                  prediction: Optional[Dict] = None) -> str:
        rsi  = tech.get('rsi', 50)
        bb   = tech.get('bb', 50)
        macd = tech.get('macd', 0)
        atr  = tech.get('atr_pct', 0)

        # ─── 예측 기반 코멘트 (한국 종목에만 적용) ───
        pred_text = ""
        if prediction:
            exp = prediction['expected_change_pct']
            exp_price = prediction['expected_price']
            conf = prediction['confidence']
            conf_kr = {"HIGH": "높음", "MEDIUM": "보통", "LOW": "낮음"}.get(conf, "보통")

            if exp > 1.5:
                emoji = "🚀"
            elif exp > 0.5:
                emoji = "📈"
            elif exp < -1.5:
                emoji = "📉"
            elif exp < -0.5:
                emoji = "🔻"
            else:
                emoji = "➖"

            pred_text = f"📌 [내일 예측] {emoji} 예상 변동 {exp:+.2f}% (목표가 {exp_price:,.0f}, 신뢰도 {conf_kr})\n\n"

        # ─── 매크로 역상관 자산 ───
        if is_inverse:
            if change_pct > 1.5:
                return f"{pred_text}🚨 {name}이(가) +{change_pct:.2f}% 급등하며 시장에 부담을 주고 있습니다. 외국인 자본 유출 압력에 주의가 필요합니다."
            if change_pct < -1.5:
                return f"{pred_text}🌤️ {name}이(가) {change_pct:.2f}% 안정화되며 투자 심리 회복이 기대됩니다."
            return f"{pred_text}⚖️ {name}은(는) {change_pct:.2f}% 변동으로 횡보 중입니다."

        # ─── 일반 주식/지수 ───
        base = f"{pred_text}전일 대비 {'+' if change_pct >= 0 else ''}{change_pct:.2f}% 변동. "

        if rsi >= 75 and bb >= 100:
            return base + "RSI와 볼린저밴드 모두 '극단적 과열' 신호. 차익 실현 압력에 주의. 🔥"
        if rsi <= 25 and bb <= 0:
            return base + "투매로 인한 '극단적 과매도' 상태. 기술적 반등 가능성. 💎"
        if macd > 0 and "정배열" in tech.get('ma_trend', ''):
            return base + "이동평균선 정배열과 MACD 상승 결합 → 강력한 주도주 흐름. 🚀"
        if macd < 0 and "역배열" in tech.get('ma_trend', ''):
            return base + "추세 붕괴 + 하락 모멘텀 가중. 바닥 확인 우선. 📉"
        if atr > 3.5:
            return base + f"변동성(ATR) {atr:.1f}%로 매우 높음. 단기 트레이딩 리스크 큼. ⚡"
        return base + "상승/하락 모멘텀 균형. 시장 온도 추종 전략 유리. 🧩"


# ═══════════════════════════════════════════════════════════════════════════
#  [9] 오라클 매크로 예측 모델 (KOSPI 시초가)
# ═══════════════════════════════════════════════════════════════════════════
class MacroOracleModel:
    CLIP_RANGE       = 3.0
    NONLINEAR_THRESH = 1.5
    NONLINEAR_BOOST  = 1.3

    def __init__(self):
        self.bull_energy: float = 0.0
        self.bear_energy: float = 0.0
        self.impact_logs: List[Dict] = []
        self.risk_assets_total: int = 0
        self.risk_assets_up: int = 0

    def process_macro_factor(self, name: str, change_pct: float, weight: float, is_inverse: bool) -> None:
        clipped = max(min(change_pct, self.CLIP_RANGE), -self.CLIP_RANGE)
        impact = clipped * weight
        if is_inverse:
            impact = -impact
        if abs(change_pct) > self.NONLINEAR_THRESH:
            impact *= self.NONLINEAR_BOOST

        self.impact_logs.append({
            "name": name, "change": change_pct, "weight": weight,
            "is_inverse": is_inverse, "impact": round(impact, 4),
        })

        if impact > 0:
            self.bull_energy += impact
        else:
            self.bear_energy += abs(impact)

    def process_market_breadth(self, category: str, change_pct: float) -> None:
        if category in ("Korea", "USA", "Asia", "Europe"):
            self.risk_assets_total += 1
            if change_pct > 0:
                self.risk_assets_up += 1

    def finalize(self) -> Tuple[int, int, str, Dict]:
        total_energy = self.bull_energy + self.bear_energy
        if total_energy < 0.01:
            oracle_score = 50
        else:
            oracle_score = round((self.bull_energy / total_energy) * 100)

        if self.risk_assets_total > 0:
            market_temp = round((self.risk_assets_up / self.risk_assets_total) * 100)
        else:
            market_temp = 50

        best  = max(self.impact_logs, key=lambda x: x['impact']) if self.impact_logs else None
        worst = min(self.impact_logs, key=lambda x: x['impact']) if self.impact_logs else None

        report = self._compose_report(oracle_score, market_temp, best, worst)

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

    def _compose_report(self, score: int, temp: int,
                        best: Optional[Dict], worst: Optional[Dict]) -> str:
        lines = ["💡 [알고리즘 분석 결과]"]
        best_name  = best['name']  if best  else "데이터 없음"
        worst_name = worst['name'] if worst else "데이터 없음"

        if score >= 65:
            lines.append(f"내일 코스피 시초가는 강한 상승 출발이 예상됩니다. '{best_name}'의 우호적 흐름이 한국 증시에 강력한 매수 에너지를 공급하고 있습니다.")
        elif score >= 55:
            lines.append(f"내일 코스피는 완만한 상승 출발이 예상됩니다. '{best_name}'의 긍정적 흐름이 외국인 매수세를 자극할 수 있습니다.")
        elif score <= 35:
            lines.append(f"내일 코스피는 하방 압력에 강하게 노출될 전망입니다. '{worst_name}'의 불안정한 움직임이 외국인 투심을 위축시킬 우려가 큽니다.")
        elif score <= 45:
            lines.append(f"내일 코스피는 약세 출발 후 반등을 시도할 가능성이 있습니다. '{worst_name}'의 부담이 있으나 저가 매수 유입 여부가 관건입니다.")
        else:
            lines.append(f"내일 한국 증시는 눈치보기 장세입니다. 상승 요인('{best_name}')과 하락 요인('{worst_name}')이 팽팽하게 맞서고 있어 종목별 차별화 대응이 유리합니다.")

        lines.append("")
        lines.append(f"🌡️ [시장 온도: {temp}°C의 의미]")
        if temp <= 30:
            lines.append(f"현재 온도가 {temp}도로 매우 차가운 이유는 전 세계 핵심 자산의 70% 이상이 일제히 파란불(하락)을 켜고 있는 '글로벌 리스크 오프' 상태이기 때문입니다.")
        elif temp <= 45:
            lines.append(f"현재 온도가 {temp}도로 미지근한 이유는 위험자산의 절반 이상이 약세를 보이며 매수 의지가 약화된 관망 우세 국면이기 때문입니다.")
        elif temp >= 70:
            lines.append(f"현재 온도가 {temp}도로 뜨거운 이유는 글로벌 자산의 대다수가 빨간불(상승)을 기록하며 '리스크 온' 매수세가 유입되고 있기 때문입니다.")
        elif temp >= 55:
            lines.append(f"현재 온도가 {temp}도로 따뜻한 이유는 위험자산 대다수가 강세 흐름을 보이며 투자 심리가 회복되고 있기 때문입니다.")
        else:
            lines.append(f"현재 온도가 {temp}도로 중립적인 이유는 특정 섹터/국가만 차별적으로 움직이는 순환매 장세가 진행 중이기 때문입니다.")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  [10] Gemini AI
# ═══════════════════════════════════════════════════════════════════════════
class GeminiOracle:
    @staticmethod
    def is_available() -> bool:
        return bool(GEMINI_API_KEY)

    @staticmethod
    def enhance_report(score: int, temp: int, meta: Dict,
                       assets_summary: List[Dict], top_predictions: List[Dict]) -> Optional[Dict]:
        if not GeminiOracle.is_available():
            logger.warning("  ⚠️  GEMINI_API_KEY 미설정 → AI 강화 스킵")
            return None

        impact_table = "\n".join([
            f"  · {x['name']}: 변동 {x['change']:+.2f}% × 가중치 {x['weight']} = 임팩트 {x['impact']:+.4f}"
            for x in meta['all_impacts'][:8]
        ])

        top_movers = sorted(assets_summary, key=lambda x: abs(x.get('change', 0)), reverse=True)[:6]
        movers_table = "\n".join([
            f"  · {x['name']}: {x['change']:+.2f}% (RSI {x['tech']['rsi']}, {x['tech']['ma_trend']})"
            for x in top_movers
        ])

        pred_table = "\n".join([
            f"  · {p['name']}: 예상 {p['prediction']['expected_change_pct']:+.2f}% → {p['prediction']['expected_price']:,.0f} (신뢰도 {p['prediction']['confidence']})"
            for p in top_predictions[:5]
        ])

        kst_str = NOW_KST.strftime("%Y-%m-%d %H:%M KST")

        prompt = f"""당신은 한국 증시 전문 매크로 애널리스트입니다. 다음 정량 데이터를 바탕으로 토스증권 스타일의 친절하고 전문적인 시장 분석 리포트를 작성하세요.

═══════════════════════════════════════════════
[분석 시각] {kst_str}

[1] 알고리즘 오라클 스코어: {score}점 (50점 중립, 100점에 가까울수록 상승)
[2] 시장 온도: {temp}°C

[3] 매크로 지표 임팩트
{impact_table}

[4] 변동성 상위 자산 TOP 6
{movers_table}

[5] 종목별 예측 변동률 TOP 5
{pred_table}

[6] 위험자산 추세: {meta['risk_assets_up']}/{meta['risk_assets_total']}개 상승
═══════════════════════════════════════════════

다음 형식으로 응답하세요. 반드시 JSON 블록을 먼저 작성하고 다른 설명은 일절 추가하지 마세요.

<REPORT>
{{
  "headline": "내일 KOSPI 방향성을 한 문장으로",
  "main_report": "💡 [알고리즘 분석 결과]\\n(2~3문장)\\n\\n🌡️ [시장 온도: {temp}°C의 의미]\\n(2문장)\\n\\n🎯 [투자자 행동 가이드]\\n(2문장)",
  "key_drivers": ["핵심 상승 요인 1", "핵심 상승 요인 2"],
  "key_risks": ["주요 리스크 1", "주요 리스크 2"],
  "confidence": "HIGH" 또는 "MEDIUM" 또는 "LOW"
}}
</REPORT>

작성 원칙:
1. 한국어, 토스증권 스타일
2. 데이터 근거 구체적 인용
3. 종목별 예측 결과를 자연스럽게 언급
4. \\n 줄바꿈, 이모지 적극 활용"""

        try:
            url = f"{GEMINI_ENDPOINT}?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1500, "topP": 0.9},
            }
            logger.info("  🤖 Gemini AI 리포트 재작성 요청...")
            res = requests.post(url, json=payload, timeout=30)
            if res.status_code != 200:
                logger.error(f"  ❌ Gemini API HTTP {res.status_code}")
                return None

            data = res.json()
            text = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
            if not text:
                return None

            match = re.search(r'<REPORT>(.*?)</REPORT>', text, re.DOTALL)
            if not match:
                return {"raw_text": text.strip()}

            json_str = match.group(1).strip()
            try:
                parsed = json.loads(json_str)
                logger.info(f"  ✅ Gemini 리포트 완료 (신뢰도: {parsed.get('confidence', '?')})")
                return parsed
            except json.JSONDecodeError as e:
                logger.error(f"  ❌ Gemini JSON 파싱 실패: {e}")
                return {"raw_text": json_str}

        except Exception as e:
            logger.error(f"  ❌ Gemini 호출 오류: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
#  [11] 히스토리 + 예측 로그 관리
# ═══════════════════════════════════════════════════════════════════════════
class HistoryManager:
    @staticmethod
    def load() -> Dict:
        if not HISTORY_FILE.exists():
            return {"date": TODAY_KEY, "snapshots": []}
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                hist = json.load(f)
            if hist.get('date') != TODAY_KEY:
                logger.info(f"  🔄 날짜 변경 ({hist.get('date')} → {TODAY_KEY}). 초기화.")
                return {"date": TODAY_KEY, "snapshots": []}
            return hist
        except Exception as e:
            logger.warning(f"  ⚠️  히스토리 로드 실패: {e}")
            return {"date": TODAY_KEY, "snapshots": []}

    @staticmethod
    def append(history: Dict, assets: List[Dict], oracle_score: int, market_temp: int) -> Dict:
        snapshot = {
            "t":      NOW_KST.strftime("%H:%M"),
            "ts":     int(NOW_KST.timestamp()),
            "score":  oracle_score,
            "temp":   market_temp,
            "prices": {a['symbol']: a['price'] for a in assets},
        }
        history['snapshots'].append(snapshot)
        if len(history['snapshots']) > 144:
            history['snapshots'] = history['snapshots'][-144:]
        return history

    @staticmethod
    def save(history: Dict) -> None:
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(history, f, ensure_ascii=False, indent=2)
            logger.info(f"  💾 history.json ({len(history['snapshots'])}개 스냅샷)")
        except Exception as e:
            logger.error(f"  ❌ history.json 저장 실패: {e}")


class PredictionLogger:
    @staticmethod
    def log_if_first_today(oracle_score: int, market_temp: int, gemini_result: Optional[Dict]) -> None:
        try:
            if PREDICTION_LOG_FILE.exists():
                with open(PREDICTION_LOG_FILE, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            else:
                logs = []

            if logs and logs[-1].get('date') == TODAY_KEY:
                return

            entry = {
                "date":         TODAY_KEY,
                "first_logged": NOW_KST.strftime("%H:%M"),
                "score":        oracle_score,
                "temp":         market_temp,
                "headline":     gemini_result.get('headline') if gemini_result else None,
                "confidence":   gemini_result.get('confidence') if gemini_result else None,
                "actual_kospi_close_pct": None,
            }
            logs.append(entry)
            logs = logs[-90:]

            with open(PREDICTION_LOG_FILE, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            logger.info(f"  📝 오늘 첫 예측 기록 (스코어: {oracle_score})")
        except Exception as e:
            logger.warning(f"  ⚠️  예측 로그 저장 실패: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  [12] 메인 오케스트레이터
# ═══════════════════════════════════════════════════════════════════════════
def execute_quant_terminal() -> None:
    logger.info("═" * 60)
    logger.info("🚀 KOSPI ORACLE V8 — STOCK PREDICTION ENGINE")
    logger.info(f"   시각: {NOW_KST.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    logger.info(f"   AI:   {'Gemini ✅' if GeminiOracle.is_available() else '폴백 모드'}")
    logger.info(f"   자산: {len(ASSETS)}종")
    logger.info("═" * 60)

    oracle = MacroOracleModel()
    processed_assets: List[Dict] = []
    failed_symbols: List[str] = []
    raw_history: Dict[str, pd.DataFrame] = {}  # Beta 계산용 시계열 보관

    # ─── Phase 1: 전체 자산 수집 + 기본 분석 ───
    total = len(ASSETS)
    for idx, (sym, info) in enumerate(ASSETS.items(), 1):
        logger.info(f"[{idx:2d}/{total}] 📡 {info['name']} ({sym})")
        hist = YFinanceFetcher.fetch(sym)
        if hist.empty:
            failed_symbols.append(sym)
            continue

        try:
            close_series = hist['Close']
            cur_price  = float(close_series.iloc[-1])
            prev_price = float(close_series.iloc[-2])
            change_pct = round(((cur_price - prev_price) / prev_price) * 100, 2)
            tech_data  = TechnicalIndicator.calculate(hist)
            sparkline  = TechnicalIndicator.sparkline(hist)

            if info['cat'] == 'Macro_Predict':
                oracle.process_macro_factor(info['name'], change_pct, info['w'], info.get('inv', False))
            oracle.process_market_breadth(info['cat'], change_pct)

            asset_data = {
                "symbol":    sym,
                "name":      info['name'],
                "cat":       info['cat'],
                "sector":    info.get('sector', 'general'),
                "price":     round(cur_price, 2) if cur_price > 100 else round(cur_price, 4),
                "change":    change_pct,
                "tech":      tech_data,
                "sparkline": sparkline,
                "ok":        True,
            }
            processed_assets.append(asset_data)
            raw_history[sym] = hist

            ch_emoji = "🔺" if change_pct > 0 else "🔻" if change_pct < 0 else "➖"
            logger.info(f"        {ch_emoji} {change_pct:+.2f}% │ RSI {tech_data['rsi']}")
        except Exception as e:
            logger.error(f"  ❌ {sym} 가공 오류: {e}")
            failed_symbols.append(sym)

    # ─── Phase 2: 오라클 KOSPI 예측 산출 ───
    logger.info("─" * 60)
    logger.info("🔮 KOSPI 시초가 오라클 산출...")
    final_score, final_temp, rule_report, oracle_meta = oracle.finalize()
    logger.info(f"   📊 오라클 스코어: {final_score}점")
    logger.info(f"   🌡️  시장 온도:    {final_temp}°C")

    # ─── Phase 3: ★ 종목별 Beta 계산 + 예측 ★ ───
    logger.info("─" * 60)
    logger.info("⚡ Phase 3: 종목별 Beta 분석 + 예측...")
    kospi_df = raw_history.get('^KS11', pd.DataFrame())

    # 빠른 lookup을 위해 dict 변환
    market_data_dict = {a['symbol']: a for a in processed_assets}

    for asset in processed_assets:
        # 한국 종목만 예측 (지수 제외)
        if asset['cat'] != 'Korea' or asset['symbol'] in ['^KS11', '^KQ11']:
            asset['prediction'] = None
            continue

        # Beta 계산
        stock_df = raw_history.get(asset['symbol'])
        beta = BetaCalculator.calculate(stock_df, kospi_df) if stock_df is not None else None
        asset['beta'] = beta

        # 예측 산출
        pred = IndividualStockPredictor.predict(
            stock_data=asset,
            beta=beta,
            kospi_score=final_score,
            market_data=market_data_dict,
            sector=asset['sector'],
        )
        asset['prediction'] = pred

        exp_pct = pred['expected_change_pct']
        emoji = "🚀" if exp_pct > 1.5 else "📈" if exp_pct > 0.5 else "📉" if exp_pct < -1.5 else "🔻" if exp_pct < -0.5 else "➖"
        logger.info(f"   {emoji} {asset['name']:15s} β={beta if beta else 'N/A':>5} → {exp_pct:+.2f}% ({pred['confidence']})")

    # ─── Phase 4: 코멘트 생성 (예측 결과 포함) ───
    for asset in processed_assets:
        info = ASSETS.get(asset['symbol'], {})
        asset['comment'] = RuleBasedCommentary.for_asset(
            asset['name'], asset['change'], asset['tech'],
            info.get('inv', False), asset.get('prediction')
        )

    # ─── Phase 5: 정렬 ───
    # 한국 종목: 예측 변동률 큰 순
    # 그 외:   당일 변동률 큰 순
    def sort_key(a):
        if a.get('prediction'):
            return abs(a['prediction']['expected_change_pct'])
        return abs(a['change'])
    processed_assets.sort(key=sort_key, reverse=True)

    # ─── Phase 6: Gemini AI 강화 ───
    logger.info("─" * 60)
    top_predictions = [a for a in processed_assets if a.get('prediction')][:10]
    gemini_result = GeminiOracle.enhance_report(
        final_score, final_temp, oracle_meta, processed_assets, top_predictions
    )

    # ─── Phase 7: 히스토리 + 로그 ───
    history = HistoryManager.load()
    history = HistoryManager.append(history, processed_assets, final_score, final_temp)
    HistoryManager.save(history)
    PredictionLogger.log_if_first_today(final_score, final_temp, gemini_result)

    # ─── Phase 8: 최종 payload ───
    # 종목별 예측 요약 (UI에서 바로 사용)
    stock_predictions_summary = [
        {
            "symbol":               a['symbol'],
            "name":                 a['name'],
            "current_price":        a['price'],
            "today_change":         a['change'],
            "expected_change_pct":  a['prediction']['expected_change_pct'],
            "expected_price":       a['prediction']['expected_price'],
            "confidence":           a['prediction']['confidence'],
            "beta":                 a.get('beta'),
            "sector":               a['sector'],
        }
        for a in processed_assets if a.get('prediction')
    ]
    # 예측 강도 순으로 정렬
    stock_predictions_summary.sort(key=lambda x: x['expected_change_pct'], reverse=True)

    payload = {
        "version":     "V8.STOCK_PREDICT",
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
            "rule_report":  rule_report,
            "disclaimer":   "본 분석은 통계 기반 [가상의 모델링 자료]이며 투자 결과에 법적 책임을 지지 않습니다.",
        },

        # ★ V8 신규 필드 ★
        "stock_predictions": {
            "top_bullish":    stock_predictions_summary[:5],
            "top_bearish":    stock_predictions_summary[-5:][::-1] if len(stock_predictions_summary) >= 5 else [],
            "all":            stock_predictions_summary,
        },

        "oracle_meta": {
            "bull_energy":  oracle_meta['bull_energy'],
            "bear_energy":  oracle_meta['bear_energy'],
            "best_factor":  oracle_meta['best_factor'],
            "worst_factor": oracle_meta['worst_factor'],
            "all_impacts":  oracle_meta['all_impacts'],
        },

        "stats": {
            "total_assets":   total,
            "success":        len(processed_assets),
            "failed":         len(failed_symbols),
            "failed_symbols": failed_symbols,
            "predicted_stocks": len(stock_predictions_summary),
            "history_count":  len(history['snapshots']),
        },

        "data": processed_assets,
    }

    try:
        with open(DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("═" * 60)
        logger.info(f"🎉 data.json 완료 ({len(processed_assets)}/{total} 자산, {len(stock_predictions_summary)}개 예측)")
        logger.info(f"   AI: {'✅ Gemini' if gemini_result else '❌ 폴백'}")
        logger.info("═" * 60)
    except Exception as e:
        logger.critical(f"💥 data.json 저장 실패: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════
#  [13] 엔트리 포인트
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        execute_quant_terminal()
    except KeyboardInterrupt:
        logger.warning("⚠️ 사용자 중단")
    except Exception as e:
        logger.critical(f"💥 파이프라인 오류: {e}", exc_info=True)
        raise
