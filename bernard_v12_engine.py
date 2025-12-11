import json
import math
import pathlib
import time
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from zoneinfo import ZoneInfo

import ccxt
import joblib
import numpy as np
import pandas as pd

# Optional Supabase + OpenAI imports
try:
    from supabase import create_client, Client  # type: ignore
except Exception:
    create_client = None
    Client = Any

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ZONES_DIR = DATA_DIR / "zones"
MODELS_DIR = ROOT / "models"
LIVE_DIR = ROOT / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)

LIVE_STATE_PATH = LIVE_DIR / "v12_state.json"
LIVE_HISTORY_PATH = LIVE_DIR / "v12_signals.parquet"
DASHBOARD_JSON_PATH = LIVE_DIR / "v12_dashboard.json"

# Use V12 meta model (trained on Bernard v12 base)
META_MODEL_PATH = MODELS_DIR / "bernard_v12_meta_rf.joblib"

# 46-symbol watchlist (matches the symbols seen in your logs)
WATCHLIST: List[str] = [
    "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "LINKUSDT",
    "AVAXUSDT", "BNBUSDT", "UNIUSDT", "DOTUSDT", "LTCUSDT",
    "BCHUSDT", "FILUSDT", "APTUSDT", "NEARUSDT", "ARBUSDT",
    "OPUSDT", "SUIUSDT", "INJUSDT", "TIAUSDT", "RUNEUSDT",
    "SEIUSDT", "ORDIUSDT", "STXUSDT", "IMXUSDT", "FETUSDT",
    "GRTUSDT", "LDOUSDT", "GALAUSDT", "CHZUSDT", "ETCUSDT",
    "FTMUSDT", "VETUSDT", "THETAUSDT", "CRVUSDT", "APEUSDT",
    "MINAUSDT", "CAKEUSDT", "ILVUSDT", "JTOUSDT", "PYTHUSDT",
    "WLDUSDT", "JUPUSDT", "ENAUSDT", "ETHFIUSDT", "OMUSDT",
]

TIMEFRAME = "1h"
KL_TZ = ZoneInfo("Asia/Kuala_Lumpur")

# V12 behaviour
RVOL_THRESHOLD = 1.0          # last 1H vol must be >= average hour-of-day volume
META_TRIGGER_BASE = 0.60      # base p_good threshold for TRIGGER
META_TRIGGER_SESSION = 0.55   # slightly relaxed threshold in London/NY
MAX_ZONE_TOUCHES = 3          # dynamic "zone exhaustion"

# Cloud / online integration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("SUPABASE_ANON_KEY", "")
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GPT_MODEL = os.getenv("BERNARD_GPT_MODEL", "gpt-4.1-mini")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BernardV12")


# ---------------------------------------------------------------------
# DATA STRUCTURES & ENCODERS
# ---------------------------------------------------------------------

@dataclass
class MarketContext:
    btc_trend4: str
    btc_regime: str
    dom_regime: str
    is_high_vol_session: bool


def encode_features(trend4: str, btc_regime: str, dom_regime: str, side: str, ztype: str):
    """
    Encoders must match the training used for the V12 meta model.
    """
    t_map = {"up": 1, "down": -1, "chop": 0}
    b_map = {"risk_on": 1, "risk_off": -1, "neutral": 0}
    d_map = {"alt_favored": 1, "btc_favored": -1, "mixed": 0}
    side_code = 1 if side.upper() == "LONG" else -1
    z_code = 1 if ztype == "demand" else -1
    return (
        t_map.get(trend4, 0),
        b_map.get(btc_regime, 0),
        d_map.get(dom_regime, 0),
        side_code,
        z_code,
    )


# ---------------------------------------------------------------------
# FETCH HELPERS
# ---------------------------------------------------------------------

def sleep_until_next_quarter():
    """
    Sleep until the next 00/15/30/45 minute mark in UTC.
    """
    now = datetime.now(timezone.utc)
    current_minute = now.minute
    next_q = ((current_minute // 15) + 1) * 15
    if next_q >= 60:
        target = (now.replace(minute=0, second=0, microsecond=0)
                  + timedelta(hours=1))
    else:
        target = now.replace(minute=next_q, second=0, microsecond=0)

    delta = max((target - now).total_seconds(), 5.0)
    logger.info(f"[SLEEP] {int(delta)}s until next quarter: {target.isoformat()}")
    time.sleep(delta)


def fetch_ohlcv_with_retry(
    ex: ccxt.Exchange,
    symbol: str,
    timeframe: str = TIMEFRAME,
    limit: int = 300,
    max_tries: int = 3,
) -> Optional[pd.DataFrame]:
    """
    Synchronous CCXT OHLCV fetch with retry, returning a DataFrame indexed by UTC timestamp.
    """
    for attempt in range(1, max_tries + 1):
        try:
            ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            if not ohlcv:
                return None
            df = pd.DataFrame(
                ohlcv,
                columns=["ts", "open", "high", "low", "close", "volume"],
            )
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df = df.set_index("ts").sort_index()
            return df
        except Exception as e:
            logger.warning(
                f"[FETCH] {symbol} attempt {attempt}/{max_tries} failed: {e}"
            )
            time.sleep(1.0)

    return None


# ---------------------------------------------------------------------
# TECHNICALS & CONTEXT
# ---------------------------------------------------------------------

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic indicator set (EMAs, MACD, RSI, ATR).
    """
    df = df.copy()
    close = df["close"]

    df["ema10"] = close.ewm(span=10, adjust=False).mean()
    df["ema21"] = close.ewm(span=21, adjust=False).mean()
    df["ema50"] = close.ewm(span=50, adjust=False).mean()
    df["sma200"] = close.rolling(200).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26

    # RSI
    d = close.diff()
    up = d.where(d > 0, 0.0)
    dn = -d.where(d < 0, 0.0)
    roll_up = up.rolling(14).mean()
    roll_dn = dn.rolling(14).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))
    df["rsi"] = df["rsi"].bfill().ffill().fillna(50.0)

    # ATR(14)
    prev_close = close.shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14).mean()

    return df


def calc_rvol(df: pd.DataFrame) -> float:
    """
    Relative volume: last closed bar volume vs average of the same hour-of-day
    over the last ~20 occurrences.
    """
    if len(df) < 24 * 20:
        return 1.0  # not enough data, treat as normal

    # last closed bar
    last_bar = df.iloc[-2]
    hour = last_bar.name.hour

    same_hour = df[df.index.hour == hour]
    if len(same_hour) < 21:
        return 1.0

    avg_vol = same_hour["volume"].iloc[-21:-1].mean()
    if avg_vol <= 0:
        return 0.0
    return float(last_bar["volume"] / avg_vol)


def in_high_vol_session(now_utc: datetime) -> bool:
    """
    London: 07–10 UTC, NY: 13–16 UTC (approx).
    """
    h = now_utc.hour
    is_london = 7 <= h < 10
    is_ny = 13 <= h < 16
    return is_london or is_ny


def get_market_context(
    btc_df: pd.DataFrame,
    alts_data: Dict[str, pd.DataFrame],
    now_utc: datetime,
) -> MarketContext:
    """
    Build Bernard-style macro context:
    - BTC 4H trend (approximated)
    - BTC regime (risk_on / risk_off / neutral)
    - Dominance regime (btc_favored / alt_favored / mixed)
    - is_high_vol_session (London/NY)
    """
    df = calculate_indicators(btc_df)
    if len(df) < 210:
        return MarketContext("chop", "neutral", "mixed", False)

    last = df.iloc[-2]

    # 4H trend approximation: 1H EMAs vs SMA200
    trend4 = "chop"
    if last["close"] > last["ema50"] > last["sma200"]:
        trend4 = "up"
    elif last["close"] < last["ema50"] < last["sma200"]:
        trend4 = "down"

    # BTC regime: MACD slope + RSI + trend4
    macd_up = df["macd"].iloc[-2] > df["macd"].iloc[-3]
    rsi_val = float(last["rsi"])

    regime = "neutral"
    if trend4 == "up" and macd_up and rsi_val > 50:
        regime = "risk_on"
    elif trend4 == "down" and (not macd_up) and rsi_val < 50:
        regime = "risk_off"

    # Dominance: BTC vs top alts last 4 bars
    if len(df) < 5:
        dom = "mixed"
    else:
        btc_ret = df["close"].iloc[-1] / df["close"].iloc[-5] - 1.0

        alt_syms = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT"]
        alt_rets = []
        for s in alt_syms:
            if s in alts_data:
                adf = alts_data[s]
                if len(adf) >= 5:
                    r = adf["close"].iloc[-1] / adf["close"].iloc[-5] - 1.0
                    alt_rets.append(r)
        avg_alt = float(np.mean(alt_rets)) if alt_rets else 0.0

        if avg_alt > btc_ret + 0.005:
            dom = "alt_favored"
        elif btc_ret > avg_alt + 0.005:
            dom = "btc_favored"
        else:
            dom = "mixed"

    return MarketContext(
        btc_trend4=trend4,
        btc_regime=regime,
        dom_regime=dom,
        is_high_vol_session=in_high_vol_session(now_utc),
    )


# ---------------------------------------------------------------------
# ZONES
# ---------------------------------------------------------------------

def load_zones(symbol: str) -> pd.DataFrame:
    path = ZONES_DIR / f"{symbol}_zones.parquet"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_parquet(path)
    # Keep only fresh / tested and not broken
    if "status" in df.columns:
        df = df[df["status"].isin(["fresh", "tested"])].copy()

    if "broken_ts" in df.columns:
        df = df[df["broken_ts"].isna()].copy()

    return df


# ---------------------------------------------------------------------
# CORE ENGINE V12
# ---------------------------------------------------------------------

class BernardEngineV12:
    def __init__(self):
        self.state = self._load_state()
        self.model, self.feat_names = self._load_model()
        self.zone_cache: Dict[str, int] = {}  # zone exhaustion counter
        self.supabase: Optional["Client"] = self._init_supabase()
        self.gpt_client: Optional["OpenAI"] = self._init_gpt_client()

    # ------------------ STATE & MODEL ------------------

    def _load_state(self) -> Dict[str, Any]:
        if LIVE_STATE_PATH.exists():
            try:
                with open(LIVE_STATE_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"last_processed_ts": ""}

    def _save_state(self):
        with open(LIVE_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def _load_model(self):
        if not META_MODEL_PATH.exists():
            logger.warning(f"[META] Model file not found: {META_MODEL_PATH}")
            return None, []

        obj = joblib.load(META_MODEL_PATH)
        if isinstance(obj, dict):
            model = obj.get("model", None)
            feat_names = obj.get("feature_names", None)
        else:
            model = obj
            feat_names = getattr(obj, "feature_names_in_", None)

        if model is None:
            model = obj

        if feat_names is None:
            # fallback: manual list used during V12 training
            feat_names = [
                "entry",
                "sl",
                "tp",
                "zone_low",
                "zone_high",
                "touches",
                "zone_thickness_pct",
                "zone_pos",
                "side_code",
                "zone_type_code",
                "trend4_btc_code",
                "btc_regime_code",
                "dom_regime_code",
            ]

        logger.info(f"[META] Loaded model. Features: {list(feat_names)}")
        return model, list(feat_names)

    def _init_supabase(self) -> Optional["Client"]:
        if not SUPABASE_URL or not SUPABASE_KEY:
            logger.info("[SUPABASE] URL/key not provided, Supabase disabled.")
            return None
        if create_client is None:
            logger.warning("[SUPABASE] supabase-py not installed, Supabase disabled.")
            return None
        try:
            client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("[SUPABASE] Client initialised.")
            return client
        except Exception as e:
            logger.warning(f"[SUPABASE] Failed to initialise client: {e}")
            return None

    def _init_gpt_client(self) -> Optional["OpenAI"]:
        if not OPENAI_API_KEY:
            logger.info("[GPT] OPENAI_API_KEY not set, GPT check disabled.")
            return None
        if OpenAI is None:
            logger.warning("[GPT] openai package not installed, GPT check disabled.")
            return None
        try:
            client = OpenAI()
            logger.info("[GPT] Client initialised.")
            return client
        except Exception as e:
            logger.warning(f"[GPT] Failed to initialise client: {e}")
            return None

    # ------------------ SIGNAL LOGIC ------------------

    def _analyze_symbol(
        self,
        symbol: str,
        df: pd.DataFrame,
        zones: pd.DataFrame,
        ctx: MarketContext,
        closed_ts: Optional[pd.Timestamp],
    ) -> List[Dict[str, Any]]:
        """
        Return a list of signal dicts (TRIGGER + EARLY) for this symbol.
        """
        if df is None or df.empty or zones is None or zones.empty:
            return []

        df = calculate_indicators(df)
        if len(df) < 3:
            return []

        # last closed bar = index[-2], current live bar = index[-1]
        last_bar = df.iloc[-2]
        curr_bar = df.iloc[-1]

        atr = float(last_bar["atr14"]) if math.isfinite(last_bar.get("atr14", np.nan)) else None
        rvol = calc_rvol(df)

        signals: List[Dict[str, Any]] = []

        for _, z in zones.iterrows():
            z_top = float(z["zone_top"])
            z_btm = float(z["zone_bottom"])
            z_type = z["zone_type"]

            zone_id = f"{symbol}_{z_type}_{z_top:.6f}_{z_btm:.6f}"
            touches_hist = int(z.get("touches", 0))
            extra_touches = self.zone_cache.get(zone_id, 0)
            total_touches = touches_hist + extra_touches
            if total_touches >= MAX_ZONE_TOUCHES:
                continue  # exhausted

            zone_h = z_top - z_btm
            if zone_h <= 0:
                continue
            mid = (z_top + z_btm) / 2.0

            # ---------------- TRIGGER LOGIC (on closed bar) ----------------
            if closed_ts is not None and df.index[-2] == closed_ts:
                in_zone = (last_bar["high"] >= z_btm) and (last_bar["low"] <= z_top)
                if in_zone:
                    o = float(last_bar["open"])
                    c = float(last_bar["close"])
                    hi = float(last_bar["high"])
                    lo = float(last_bar["low"])

                    body = abs(c - o)
                    rng = hi - lo
                    if rng <= 0:
                        continue

                    # Doji filter (skip super tiny bodies)
                    if body < 0.2 * rng:
                        continue

                    # Volume filter
                    if rvol < RVOL_THRESHOLD:
                        continue

                    side: Optional[str] = None

                    if z_type == "demand" and c > o:
                        lower_wick = min(o, c) - lo
                        if lower_wick > 0.5 * body:
                            side = "LONG"
                    elif z_type == "supply" and c < o:
                        upper_wick = hi - max(o, c)
                        if upper_wick > 0.5 * body:
                            side = "SHORT"

                    # Context filter
                    if side == "LONG":
                        if ctx.btc_regime == "risk_off" or ctx.dom_regime == "btc_favored":
                            side = None
                    elif side == "SHORT":
                        if ctx.btc_regime == "risk_on" or ctx.dom_regime == "alt_favored":
                            side = None

                    if side:
                        # record extra touch
                        self.zone_cache[zone_id] = extra_touches + 1
                        sig = self._build_signal(
                            symbol=symbol,
                            side=side,
                            kind="TRIGGER",
                            price=c,
                            zone_top=z_top,
                            zone_btm=z_btm,
                            touches=total_touches,
                            ctx=ctx,
                        )
                        signals.append(sig)

            # ---------------- EARLY ALERT (current price near zone) ----------------
            curr_price = float(curr_bar["close"])
            if curr_price > z_top:
                dist = curr_price - z_top
            elif curr_price < z_btm:
                dist = z_btm - curr_price
            else:
                dist = 0.0

            if dist <= 0.5 * zone_h:
                side = "LONG" if z_type == "demand" else "SHORT"
                valid_ctx = True
                if side == "LONG" and (ctx.btc_regime == "risk_off" or ctx.dom_regime == "btc_favored"):
                    valid_ctx = False
                if side == "SHORT" and (ctx.btc_regime == "risk_on" or ctx.dom_regime == "alt_favored"):
                    valid_ctx = False

                if valid_ctx:
                    sig = self._build_signal(
                        symbol=symbol,
                        side=side,
                        kind="EARLY",
                        price=curr_price,
                        zone_top=z_top,
                        zone_btm=z_btm,
                        touches=total_touches,
                        ctx=ctx,
                    )
                    signals.append(sig)

        return signals

    def _build_signal(
        self,
        symbol: str,
        side: str,
        kind: str,
        price: float,
        zone_top: float,
        zone_btm: float,
        touches: int,
        ctx: MarketContext,
    ) -> Dict[str, Any]:
        """
        Build a signal dict and apply META scoring + thresholds.
        """
        zone_h = zone_top - zone_btm
        mid = (zone_top + zone_btm) / 2.0
        thickness_pct = (zone_h / mid) * 100.0 if mid > 0 else 0.0
        zone_pos = (price - zone_btm) / zone_h if zone_h > 0 else 0.5
        zone_pos = max(0.0, min(1.0, zone_pos))

        # SL/TP with 2R structure
        if side.upper() == "LONG":
            sl = zone_btm - 0.2 * zone_h
            dist = price - sl
            if dist <= 0:
                sl = price * 0.99
                dist = price - sl
            tp = price + 2.0 * dist
        else:
            sl = zone_top + 0.2 * zone_h
            dist = sl - price
            if dist <= 0:
                sl = price * 1.01
                dist = sl - price
            tp = price - 2.0 * dist

        t_c, b_c, d_c, side_c, z_c = encode_features(
            ctx.btc_trend4,
            ctx.btc_regime,
            ctx.dom_regime,
            side,
            "demand" if side.upper() == "LONG" else "supply",
        )

        feat_vec = {
            "entry": price,
            "sl": sl,
            "tp": tp,
            "zone_low": zone_btm,
            "zone_high": zone_top,
            "touches": touches,
            "zone_thickness_pct": thickness_pct,
            "zone_pos": zone_pos,
            "side_code": side_c,
            "zone_type_code": z_c,
            "trend4_btc_code": t_c,
            "btc_regime_code": b_c,
            "dom_regime_code": d_c,
        }

        score = 0.5
        if self.model is not None and self.feat_names:
            try:
                X = np.array([[feat_vec[c] for c in self.feat_names]])
                score = float(self.model.predict_proba(X)[:, 1][0])
            except Exception as e:
                logger.warning(f"[META] Scoring error: {e}")

        # dynamic threshold: session vs base
        threshold = META_TRIGGER_SESSION if ctx.is_high_vol_session else META_TRIGGER_BASE

        status = "DISCARD"
        if kind == "TRIGGER":
            if score >= threshold:
                status = "ACTIVE"
        else:  # EARLY
            if score >= (threshold - 0.05):
                status = "WATCH"

        return {
            "symbol": symbol,
            "side": side,
            "kind": kind,
            "status": status,
            "price": round(price, 5),
            "sl": round(sl, 5),
            "tp": round(tp, 5),
            "score": round(score, 2),
            "zone_top": zone_top,
            "zone_btm": zone_btm,
            "touches": touches,
            "btc_trend4": ctx.btc_trend4,
            "btc_regime": ctx.btc_regime,
            "dom_regime": ctx.dom_regime,
            "session_boost": ctx.is_high_vol_session,
            "ts_kl": datetime.now(KL_TZ).isoformat(timespec="seconds"),
            # GPT overlay (filled later)
            "gpt_approved": None,
            "gpt_quality": None,
            "gpt_confidence": None,
            "gpt_comment": None,
        }

    # ------------------ GPT VIRTUAL CHECK-UP ------------------

    @staticmethod
    def _prepare_ohlc_for_gpt(df: Optional[pd.DataFrame], max_bars: int = 60) -> List[Dict[str, Any]]:
        if df is None or df.empty:
            return []
        df_tail = df.tail(max_bars)
        records: List[Dict[str, Any]] = []
        for ts, row in df_tail.iterrows():
            records.append(
                {
                    "ts": ts.isoformat(),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                    "volume": float(row["volume"]),
                }
            )
        return records

    def _gpt_enrich_signals(
        self,
        triggers: List[Dict[str, Any]],
        watch_list: List[Dict[str, Any]],
        ctx: MarketContext,
        btc_df: pd.DataFrame,
        alts_data: Dict[str, pd.DataFrame],
    ) -> (List[Dict[str, Any]], List[Dict[str, Any]]):
        """
        For each signal, call GPT to act as Bernard discretionary eye.
        Does NOT change base engine; only adds gpt_* fields.
        """
        if self.gpt_client is None:
            return triggers, watch_list

        btc_ohlc = self._prepare_ohlc_for_gpt(btc_df, max_bars=80)

        def enrich(sig: Dict[str, Any]) -> Dict[str, Any]:
            symbol = sig["symbol"]
            sym_df = alts_data.get(symbol)
            sym_ohlc = self._prepare_ohlc_for_gpt(sym_df, max_bars=80)

            payload = {
                "signal": sig,
                "btc_context": {
                    "btc_trend4": ctx.btc_trend4,
                    "btc_regime": ctx.btc_regime,
                    "dom_regime": ctx.dom_regime,
                    "session_boost": ctx.is_high_vol_session,
                },
                "btc_ohlc": btc_ohlc,
                "symbol_ohlc": sym_ohlc,
            }

            system_prompt = (
                "You are Bernard, an expert zone trader. "
                "You receive crypto zone-based signals from a fixed engine. "
                "The engine already filters by structure, volume, and BTC context. "
                "Your job is a final discretionary check: "
                "look at the provided OHLC data + signal info and decide if the trade "
                "aligns with Bernard-style clean structure, reaction, and context.\n\n"
                "Rules (do NOT change them, only apply them):\n"
                "- Trade only if price reacts cleanly at demand/supply zones.\n"
                "- Avoid overextended moves into a zone or messy chop right before entry.\n"
                "- Respect BTC regime: avoid longs in hard risk_off, avoid shorts in hard risk_on.\n"
                "- Thin, well-defined zones are stronger; very thick zones are weaker.\n"
                "- SL/TP are already structure-based; you do NOT adjust them, only approve or reject.\n\n"
                "Respond ONLY with a compact JSON object with keys: "
                '{"approve": bool, "quality": "A+|A|B|C", "confidence": float, "comment": str}.'
            )

            try:
                resp = self.gpt_client.chat.completions.create(
                    model=GPT_MODEL,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": (
                                "Here is the full signal + recent candles in JSON. "
                                "Return ONLY the JSON decision object, no extra text.\n\n"
                                f"{json.dumps(payload, separators=(',', ':'))}"
                            ),
                        },
                    ],
                )
                content = resp.choices[0].message.content.strip()
                # Attempt to parse JSON; tolerate wrapped code fences
                if content.startswith("```"):
                    content = content.strip("`")
                    # handle formats like ```json\n...\n```
                    parts = content.split("\n", 1)
                    if len(parts) == 2:
                        content = parts[1]
                decision = json.loads(content)
                sig["gpt_approved"] = bool(decision.get("approve"))
                sig["gpt_quality"] = decision.get("quality")
                sig["gpt_confidence"] = float(decision.get("confidence", 0.0))
                sig["gpt_comment"] = decision.get("comment")
            except Exception as e:
                logger.warning(f"[GPT] Failed to enrich signal {symbol}: {e}")
                # leave gpt_* as None to indicate 'not checked'
                pass

            return sig

        enriched_triggers = [enrich(s) for s in triggers]
        enriched_watch = [enrich(s) for s in watch_list]
        return enriched_triggers, enriched_watch

    # ------------------ SUPABASE PUSH ------------------

    def _push_to_supabase(
        self,
        triggers: List[Dict[str, Any]],
        watch_list: List[Dict[str, Any]],
        ctx: MarketContext,
        btc_price: float,
    ):
        """
        Push current signals into Supabase tables:
        - signals_triggers
        - signals_alerts
        - meta_logs (light)
        """
        if self.supabase is None:
            return

        ts_kl_now = datetime.now(KL_TZ).isoformat(timespec="seconds")

        try:
            # TRIGGERS -> signals_triggers
            if triggers:
                rows_trig = []
                for s in triggers:
                    rows_trig.append(
                        {
                            "symbol": s["symbol"],
                            "side": s["side"],
                            "kind": s["kind"],
                            "status": s["status"],
                            "entry": float(s["price"]),
                            "sl": float(s["sl"]),
                            "tp": float(s["tp"]),
                            "meta_score": float(s["score"]),
                            "zone_low": float(s["zone_btm"]),
                            "zone_high": float(s["zone_top"]),
                            "touches": int(s["touches"]),
                            "btc_trend4": s["btc_trend4"],
                            "btc_regime": s["btc_regime"],
                            "dom_regime": s["dom_regime"],
                            "session_boost": bool(s["session_boost"]),
                            "gpt_approved": s["gpt_approved"],
                            "gpt_quality": s["gpt_quality"],
                            "gpt_confidence": s["gpt_confidence"],
                            "gpt_comment": s["gpt_comment"],
                            "ts_kl": s["ts_kl"],
                            "btc_price": btc_price,
                            "created_kl": ts_kl_now,
                        }
                    )
                self.supabase.table("signals_triggers").insert(rows_trig).execute()

            # WATCH LIST -> signals_alerts
            if watch_list:
                rows_alert = []
                for s in watch_list:
                    rows_alert.append(
                        {
                            "symbol": s["symbol"],
                            "side": s["side"],
                            "kind": s["kind"],
                            "status": s["status"],
                            "entry": float(s["price"]),
                            "sl": float(s["sl"]),
                            "tp": float(s["tp"]),
                            "meta_score": float(s["score"]),
                            "zone_low": float(s["zone_btm"]),
                            "zone_high": float(s["zone_top"]),
                            "touches": int(s["touches"]),
                            "btc_trend4": s["btc_trend4"],
                            "btc_regime": s["btc_regime"],
                            "dom_regime": s["dom_regime"],
                            "session_boost": bool(s["session_boost"]),
                            "gpt_approved": s["gpt_approved"],
                            "gpt_quality": s["gpt_quality"],
                            "gpt_confidence": s["gpt_confidence"],
                            "gpt_comment": s["gpt_comment"],
                            "ts_kl": s["ts_kl"],
                            "btc_price": btc_price,
                            "created_kl": ts_kl_now,
                        }
                    )
                self.supabase.table("signals_alerts").insert(rows_alert).execute()

            # META LOGS (one row per signal)
            meta_rows = []
            for s in triggers + watch_list:
                meta_rows.append(
                    {
                        "symbol": s["symbol"],
                        "kind": s["kind"],
                        "side": s["side"],
                        "meta_score": float(s["score"]),
                        "gpt_approved": s["gpt_approved"],
                        "gpt_quality": s["gpt_quality"],
                        "gpt_confidence": s["gpt_confidence"],
                        "ts_kl": s["ts_kl"],
                        "btc_trend4": s["btc_trend4"],
                        "btc_regime": s["btc_regime"],
                        "dom_regime": s["dom_regime"],
                        "session_boost": bool(s["session_boost"]),
                        "created_kl": ts_kl_now,
                    }
                )
            if meta_rows:
                self.supabase.table("meta_logs").insert(meta_rows).execute()

            logger.info(
                f"[SUPABASE] Pushed {len(triggers)} triggers and {len(watch_list)} alerts."
            )

        except Exception as e:
            logger.warning(f"[SUPABASE] Failed to push signals: {e}")

    # ------------------ MAIN LOOP ------------------

    def run(self):
        logger.info("[EXCHANGE] Loading Binance markets once...")
        exchange = ccxt.binance({"enableRateLimit": True})
        try:
            exchange.load_markets()
        except Exception as e:
            logger.warning(f"[EXCHANGE] Failed to load markets: {e}")

        logger.info("=== Starting Bernard V12 Live Engine (sync) ===")
        logger.info(f"Watchlist symbols: {len(WATCHLIST)}")

        while True:
            sleep_until_next_quarter()

            now_utc = datetime.now(timezone.utc)
            kl_now = now_utc.astimezone(KL_TZ)
            logger.info(
                f"[LOOP] UTC: {now_utc.isoformat()} | KL: {kl_now.isoformat()}"
            )

            # 1. Fetch BTC (for context + closed candle timing)
            logger.info("[FETCH] Fetching OHLCV for BTC + watchlist...")
            df_btc = fetch_ohlcv_with_retry(exchange, "BTC/USDT", limit=300)
            if df_btc is None or len(df_btc) < 3:
                logger.error("[ERROR] Missing or insufficient BTC 1h data, skipping this loop.")
                continue

            # We'll treat index[-2] as "last closed"
            closed_ts = df_btc.index[-2]

            last_processed = self.state.get("last_processed_ts", "")
            is_new_candle = (str(closed_ts) != last_processed)

            # 2. Fetch alt data (for dominance + per-symbol analysis)
            alts_data: Dict[str, pd.DataFrame] = {}
            for sym in WATCHLIST:
                ccxt_sym = sym.replace("USDT", "/USDT")
                df = fetch_ohlcv_with_retry(exchange, ccxt_sym, limit=300)
                if df is not None and len(df) >= 3:
                    alts_data[sym] = df

            if "BTCUSDT" not in alts_data:
                alts_data["BTCUSDT"] = df_btc

            # 3. Build macro context
            ctx = get_market_context(df_btc, alts_data, now_utc)
            logger.info(
                f"[CTX] trend4={ctx.btc_trend4}, regime={ctx.btc_regime}, dom={ctx.dom_regime}, "
                f"high_vol_session={ctx.is_high_vol_session}"
            )

            # 4. Per-symbol analysis
            all_signals: List[Dict[str, Any]] = []
            for sym, df in alts_data.items():
                if sym == "BTCUSDT":
                    continue
                zones = load_zones(sym)
                if zones.empty:
                    continue

                # ensure symbol shares same closed_ts as BTC
                if closed_ts not in df.index:
                    continue

                sigs = self._analyze_symbol(sym, df, zones, ctx, closed_ts if is_new_candle else None)
                all_signals.extend(sigs)

            # filter only ACTIVE or WATCH
            active = [s for s in all_signals if s["status"] != "DISCARD"]
            triggers = [s for s in active if s["kind"] == "TRIGGER" and s["status"] == "ACTIVE"]
            watch_list = [s for s in active if s["kind"] == "EARLY" and s["status"] == "WATCH"]

            logger.info(
                f"[RESULT] total_signals={len(all_signals)}, triggers={len(triggers)}, watch={len(watch_list)}"
            )

            # 4b. GPT virtual check (discretionary Bernard eye)
            if self.gpt_client is not None and (triggers or watch_list):
                triggers, watch_list = self._gpt_enrich_signals(
                    triggers, watch_list, ctx, df_btc, alts_data
                )

            # 4c. Push to Supabase for online dashboard / webhooks
            btc_price = float(df_btc["close"].iloc[-1])
            self._push_to_supabase(
                triggers=triggers,
                watch_list=watch_list,
                ctx=ctx,
                btc_price=btc_price,
            )

            # 5. Append to history parquet (fix: ts_kl as string)
            if active:
                df_new = pd.DataFrame(active)

                # ensure ts_kl is consistent string to avoid ArrowTypeError
                df_new["ts_kl"] = df_new["ts_kl"].astype(str)

                if LIVE_HISTORY_PATH.exists():
                    df_old = pd.read_parquet(LIVE_HISTORY_PATH)
                    if "ts_kl" in df_old.columns:
                        df_old["ts_kl"] = df_old["ts_kl"].astype(str)
                    df_hist = pd.concat([df_old, df_new], ignore_index=True)
                else:
                    df_hist = df_new

                df_hist.to_parquet(LIVE_HISTORY_PATH, index=False)

            # 6. Write dashboard JSON snapshot (Cloudflare / TradingView reader)
            dashboard_data = {
                "updated_kl": datetime.now(KL_TZ).isoformat(timespec="seconds"),
                "btc_price": btc_price,
                "btc_trend4": ctx.btc_trend4,
                "btc_regime": ctx.btc_regime,
                "dom_regime": ctx.dom_regime,
                "session_boost": ctx.is_high_vol_session,
                "triggers": triggers,
                "watch_list": watch_list,
            }

            with open(DASHBOARD_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(dashboard_data, f, indent=2)

            logger.info(
                f"[DASH] wrote dashboard JSON: triggers={len(triggers)}, watch={len(watch_list)}"
            )

            # 7. Save state when a new closed candle was processed
            if is_new_candle:
                self.state["last_processed_ts"] = str(closed_ts)
                self._save_state()


if __name__ == "__main__":
    engine = BernardEngineV12()
    engine.run()
