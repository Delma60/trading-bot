# risk_manager.py
from typing import Dict, Any, Optional
import MetaTrader5 as mt5
from datetime import datetime, timezone
import math
import pandas as pd
import time
import threading
from collections import Counter
from manager.market_sessions import MarketSessionManager
from manager.profile_manager import profile as _profile

def _pip_value_usd(symbol: str, lots: float) -> float:
    """Return the USD value of exactly one pip for the given symbol and lot size."""
    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            return 0.10

        tick_value = float(info.trade_tick_value)
        tick_size = float(info.trade_tick_size)
        point = float(info.point)
        digits = int(info.digits)

        if tick_value <= 0 or tick_size <= 0 or point <= 0:
            return 0.10

        if digits in (5, 3):
            pip_size = 10.0 * point
        else:
            pip_size = point

        pip_value_per_lot = (pip_size / tick_size) * tick_value
        return max(round(pip_value_per_lot * lots, 5), 1e-5)
    except Exception:
        return 0.10


# ─────────────────────────────────────────────────────────────────────────────
# NEW: MarketConditionFilter (Feature #2)
# ─────────────────────────────────────────────────────────────────────────────

class MarketConditionFilter:
    """Blocks trading during known unfavorable conditions (volatility spikes, dead volume)."""
    def __init__(self, broker):
        self.broker = broker

    def is_market_suitable(self, symbol: str) -> tuple:
        """Check if market conditions are favorable for trading."""
        df = self.broker.get_historical_rates(symbol, "H1", 50)
        if df is None or df.empty or len(df) < 50:
            return True, "Not enough data, skipping filter."

        df = df.copy()
        df['tr'] = df['high'] - df['low']
        recent_atr = df['tr'].tail(3).mean()
        normal_atr = df['tr'].tail(40).mean()

        # 1. Volatility spike (news event)
        if recent_atr > normal_atr * 2.5:
            return False, "Volatility spike detected — likely news event. Waiting."

        # 2. Dead volume (market sleeping)
        if 'tick_volume' in df.columns:
            recent_vol = df['tick_volume'].tail(3).mean()
            avg_vol = df['tick_volume'].tail(40).mean()
            if recent_vol < avg_vol * 0.3:
                return False, "Volume too low — market is sleeping. No entries."

        return True, "Market conditions suitable."


# ─────────────────────────────────────────────────────────────────────────────
# NEW: SmartReEntrySystem (Feature #9)
# ─────────────────────────────────────────────────────────────────────────────

class SmartReEntrySystem:
    """Tracks stop-outs to allow re-entry if price sweeps liquidity and recovers."""
    def __init__(self):
        self.stopped_out_trades = {}  # {symbol: {"time": datetime, "price": float, "direction": str}}

    def record_stop_out(self, symbol: str, price: float, direction: str):
        """Record when a position was stopped out."""
        self.stopped_out_trades[symbol] = {
            "time": datetime.now(),
            "price": price,
            "direction": direction,
            "re_entered": False
        }

    def check_reentry_validity(self, symbol: str, current_price: float, new_signal_direction: str) -> bool:
        """Check if re-entry conditions are met after a stop-out."""
        record = self.stopped_out_trades.get(symbol)
        if not record or record["re_entered"]:
            return False
            
        # If signal is same direction and within 4 hours, and price is better or equal
        time_elapsed = (datetime.now() - record["time"]).total_seconds() / 3600
        if time_elapsed <= 4.0 and new_signal_direction == record["direction"]:
            if new_signal_direction == "BUY" and current_price <= record["price"]:
                return True
            if new_signal_direction == "SELL" and current_price >= record["price"]:
                return True
        return False


# ─────────────────────────────────────────────────────────────────────────────
# LockBalanceGuard
# ─────────────────────────────────────────────────────────────────────────────

class LockBalanceGuard:
    """
    Protects a ring-fenced portion of the account from any trading activity.

    The lock can be specified as:
        - A fixed dollar amount  (lock_amount=300)
        - A percentage of balance (lock_pct=0.30 → 30%)
        - Both — the LARGER of the two is applied

    All downstream risk calculations (position sizing, daily loss limit,
    drawdown watermarks) must use `tradeable_balance()` instead of the
    raw account balance.
    """

    def __init__(
        self,
        lock_amount: float = 0.0,   # fixed $ amount to protect
        lock_pct:    float = 0.0,   # fraction of balance to protect (0.0–1.0)
    ):
        self.lock_amount = max(0.0, lock_amount)
        self.lock_pct    = max(0.0, min(lock_pct, 0.99))   # cap at 99%

    def effective_lock(self, balance: float) -> float:
        """
        Returns the actual dollar amount that is locked.
        If both lock_amount and lock_pct are set, the larger is used.
        """
        if balance <= 0:
            return 0.0
        pct_lock = balance * self.lock_pct
        return max(self.lock_amount, pct_lock)

    def tradeable_balance(self, balance: float) -> float:
        """
        Returns the portion of balance available for trading.
        This is what ALL risk calculations must use.
        """
        return max(0.0, balance - self.effective_lock(balance))

    def is_locked_out(self, balance: float) -> bool:
        """True if the lock consumes the entire balance — no trading allowed."""
        return self.tradeable_balance(balance) <= 0

    def status_str(self, balance: float) -> str:
        """Human-readable lock status for chat responses."""
        lock   = self.effective_lock(balance)
        trade  = self.tradeable_balance(balance)
        if lock <= 0:
            return f"No lock balance set. Full balance ${balance:,.2f} is tradeable."
        pct_of_balance = (lock / balance * 100) if balance > 0 else 0
        return (
            f"Lock balance: ${lock:,.2f} ({pct_of_balance:.1f}% of ${balance:,.2f}). "
            f"Tradeable: ${trade:,.2f}."
        )

    def update(self, lock_amount: float = None, lock_pct: float = None):
        """Update lock parameters at runtime (e.g. from chat command)."""
        if lock_amount is not None:
            self.lock_amount = max(0.0, lock_amount)
        if lock_pct is not None:
            self.lock_pct = max(0.0, min(lock_pct, 0.99))


# ─────────────────────────────────────────────────────────────────────────────
# BalancePipSizer
# ─────────────────────────────────────────────────────────────────────────────

class BalancePipSizer:
    """
    Derives an appropriate SL pip distance from the TRADEABLE account balance.

    Rationale
    ---------
    A fixed SL (e.g. "always 20 pips") ignores account size entirely.
    A $200 account risks ruin on a 50-pip stop; a $5,000 account can
    absorb wide stops without threatening its daily limit.

    This class computes a balance-tier base pip, then bounds the live
    ATR within [50%, 150%] of that base — so the stop adapts to
    volatility but stays within a range appropriate for the account size.

    Tiers (fully tunable via constructor)
    --------------------------------------
    tradeable < $200    →  8 pips
    $200 – $499         → 12 pips
    $500 – $999         → 18 pips
    $1,000 – $2,499     → 25 pips
    $2,500 – $4,999     → 35 pips
    ≥ $5,000            → 50 pips
    """

    # Default tiers: (min_balance, pip_base)
    # Sorted ascending — first match wins.
    DEFAULT_TIERS = [
        (0,     8),
        (200,  12),
        (500,  18),
        (1000, 25),
        (2500, 35),
        (5000, 50),
    ]

    def __init__(
        self,
        tiers:           list[tuple[float, float]] = None,
        atr_floor_ratio: float = 0.50,   # ATR must be ≥ base × this
        atr_ceil_ratio:  float = 1.50,   # ATR must be ≤ base × this
    ):
        self.tiers           = sorted(tiers or self.DEFAULT_TIERS, key=lambda t: t[0])
        self.atr_floor_ratio = atr_floor_ratio
        self.atr_ceil_ratio  = atr_ceil_ratio

    def base_pips(self, tradeable_balance: float) -> float:
        """
        Lookup the balance-tier pip base for this account size.
        Returns the pip count appropriate for the tradeable balance.
        """
        result = self.tiers[0][1]   # default to lowest tier
        for min_bal, pips in self.tiers:
            if tradeable_balance >= min_bal:
                result = pips
        return float(result)

    def get_sl_pips(self, tradeable_balance: float, atr_pips: float = 0.0) -> float:
        """
        Returns the final SL pip distance to use for this trade.

        Logic
        -----
        1. Determine the balance-tier base pip.
        2. Compute the ATR-bounded range: [base × floor, base × ceil].
        3. Clamp the live ATR within that range.
        4. If no valid ATR is provided (0 or missing), use the base directly.

        Parameters
        ----------
        tradeable_balance : float
            Account balance minus the lock. NOT the raw balance.
        atr_pips : float
            ATR expressed in pips (from DynamicRiskTargeter).
            Pass 0 if ATR data is unavailable.
        """
        base = self.base_pips(tradeable_balance)

        if atr_pips <= 0:
            # No ATR data — use balance-tier base directly
            return base

        floor_pips = base * self.atr_floor_ratio
        ceil_pips  = base * self.atr_ceil_ratio

        # Clamp the live ATR within the balance-appropriate range
        final = max(floor_pips, min(atr_pips, ceil_pips))
        return round(final, 1)

    def describe(self, tradeable_balance: float, atr_pips: float = 0.0) -> str:
        """Human-readable explanation — useful for debug/ARIA responses."""
        base  = self.base_pips(tradeable_balance)
        final = self.get_sl_pips(tradeable_balance, atr_pips)
        atr_str = f", ATR {atr_pips:.1f}p" if atr_pips > 0 else " (no ATR)"
        return (
            f"Balance ${tradeable_balance:,.0f} → base {base:.0f}p"
            f"{atr_str} → SL {final:.1f}p "
            f"[range {base * self.atr_floor_ratio:.0f}–{base * self.atr_ceil_ratio:.0f}p]"
        )


class RiskManager:
    """The Defense Engine: Handles exposure, drawdown limits, and position sizing."""
    
    def __init__(self, broker, cache=None, max_open_trades: int = 3, min_margin_level: float = 150.0, notify_callback=print,
                 pyramid_min_pips: float = 1.0, spread_tolerance_pips: float = 1.0):
        self.broker = broker
        self.cache = cache
        self.max_open_trades = max_open_trades
        self.min_margin_level = min_margin_level
        self.notify = notify_callback
        self.pyramid_min_pips = pyramid_min_pips
        self.spread_tolerance_pips = spread_tolerance_pips
        
        # Atomic Loss Lock (prevents concurrent threads from bypassing daily loss limit)
        self._loss_lock = threading.Lock()
        
        self.daily_high_watermark = 0.0 
        self.daily_low_watermark = 0.0
        self.watermark_date=None

        # Risk calculators — instantiated once to avoid fresh objects on every call
        self.targeter = DynamicRiskTargeter(broker)
        self.reentry_system = SmartReEntrySystem()

        # ── Balance-based pip sizing & lock balance ───────────────────────
        r = _profile.risk()
        self.lock_guard = LockBalanceGuard(
            lock_amount = r.lock_amount,
            lock_pct    = r.lock_pct_decimal,   # always 0–1, bug fixed
        )
        self.balance_pip_sizer = BalancePipSizer()
    
    def _load_profile(self) -> dict:
        from pathlib import Path
        import json
        try:
            return json.loads(Path("data/profile.json").read_text())
        except Exception:
            return {}
    
    def record_stop_out_position(self, symbol: str, close_price: float, position_type: int):
        """Record a stop-out when a position closes (called from PortfolioManager or chat layer).
        position_type: 0 = BUY, 1 = SELL (from mt5.POSITION_TYPE_BUY/SELL)"""
        direction = "BUY" if position_type == 0 else "SELL"
        self.reentry_system.record_stop_out(symbol, close_price, direction)

    def _get_realized_daily_loss(self) -> float:
        """
        Fetches all closed deals for today and calculates realized P&L.
        Returns the absolute loss amount (positive = loss).
        """
        try:
            # Get today's start timestamp
            today_start = int(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
            today_end = int(datetime.now().timestamp())
            
            # Fetch all deals closed today
            deals = mt5.history_deals_get(today_start, today_end)
            if deals is None or len(deals) == 0:
                return 0.0
            
            realized_pnl = 0.0
            for deal in deals:
                if deal.profit < 0:  # Negative profit = realized loss
                    realized_pnl += abs(deal.profit)
            
            return realized_pnl
        except Exception as e:
            self.notify(f"[Risk Manager]: Warning - could not fetch realized losses: {e}")
            return 0.0

    def _pyramid_min_usd(self, symbol: str, lots: float) -> float:
        if self.cache is not None:
            return self.cache.get_pip_value(symbol, lots) * self.pyramid_min_pips
        return _pip_value_usd(symbol, lots) * self.pyramid_min_pips

    def _spread_tolerance_usd(self, symbol: str, lots: float) -> float:
        if self.cache is not None:
            return -abs(self.cache.get_pip_value(symbol, lots) * self.spread_tolerance_pips)
        return -abs(_pip_value_usd(symbol, lots) * self.spread_tolerance_pips)

    def _check_existing_positions(self, symbol: str, symbol_positions: list) -> tuple[bool, str]:
        if not symbol_positions:
            return True, "No existing positions on this symbol."

        for pos in symbol_positions:
            lots = float(pos.volume)
            profit = float(pos.profit)
            spread_tol = self._spread_tolerance_usd(symbol, lots)
            pyramid_min = self._pyramid_min_usd(symbol, lots)

            if profit < spread_tol:
                return False, (
                    f"{symbol} has a losing position (${profit:.2f} < tolerance ${spread_tol:.2f} "
                    f"= -{self.spread_tolerance_pips} pip on {lots}L). No new entries while in the red."
                )

            if profit < pyramid_min:
                return False, (
                    f"{symbol} not yet profitable enough to pyramid (${profit:.2f} < ${pyramid_min:.2f} "
                    f"= {self.pyramid_min_pips} pip on {lots}L). Wait for it to prove itself."
                )

        return True, "Existing positions profitable — pyramiding approved."

    def is_trading_allowed(self, symbol: str, max_daily_loss: float, portfolio_size: int) -> tuple[bool, str]:
        """A detailed check verifying global exposure, symbol exposure, and profitability. 
        Atomic daily loss check (wrapped with _loss_lock) prevents concurrent threads from bypassing limit."""
        
        account = self.cache.get_account() if self.cache is not None else self.broker.getAccountInfo()
        if account is None:
            account = self.broker.getAccountInfo()

        if not account:
            return False, "Could not fetch account data from broker."

        positions = self.cache.get_positions() if self.cache is not None else self.broker.getPositions()
        if positions is None:
            positions = self.broker.getPositions()

        # Lock balance gate
        tradeable = self.lock_guard.tradeable_balance(account.balance)
        if tradeable <= 0:
            return False, (
                f"Lock balance (${self.lock_guard.effective_lock(account.balance):,.2f}) "
                f"consumes the full account — no tradeable balance."
            )

        # 1. Global Exposure Limit (Check against max_open_trades, not portfolio_size)
        current_open_trades = len(positions) if positions else 0
        if current_open_trades >= self.max_open_trades:
            return False, f"Global exposure reached ({current_open_trades}/{self.max_open_trades} trades)."

        # 2. Check Specific Symbol Limits
        if positions:
            symbol_positions = [p for p in positions if p.symbol == symbol]
            symbol_trade_count = len(symbol_positions)

            # Rule A: Max 3 trades per symbol
            if symbol_trade_count >= 3:
                return False, f"Max symbol exposure reached for {symbol} ({symbol_trade_count}/3 trades)."

            allowed_b, reason_b = self._check_existing_positions(symbol, symbol_positions)
            if not allowed_b:
                return False, reason_b
        
        # 3. Check Margin Level
        if account.margin_level and account.margin_level < self.min_margin_level:
            return False, f"Margin level too low ({account.margin_level:.1f}%). Minimum is {self.min_margin_level}%."

        # === ATOMIC DAILY LOSS CHECK (wrapped with _loss_lock) ===
        # This is CRITICAL: prevents concurrent threads from both checking and bypassing the limit simultaneously
        with self._loss_lock:
            current_equity = account.equity
            today = datetime.now().date()
            
            # Reset watermark at midnight, or initialize on first run
            if self.watermark_date != today:
                self.daily_high_watermark = max(account.balance, current_equity)
                self.daily_low_watermark = min(account.balance, current_equity)
                self.watermark_date = today
            elif current_equity > self.daily_high_watermark:
                # Update the high watermark as the portfolio grows!
                self.daily_high_watermark = current_equity
            elif current_equity < self.daily_low_watermark:
                # Update the low watermark as the portfolio shrinks!
                self.daily_low_watermark = current_equity

            # Calculate the drawdown from the PEAK, not the starting balance
            trailing_drawdown = self.daily_high_watermark - current_equity
            
            if max_daily_loss > 0 and trailing_drawdown >= max_daily_loss:
                return False, f"FATAL: Trailing drawdown limit hit! Peak: ${self.daily_high_watermark:,.2f}, Dropped: ${trailing_drawdown:.2f} (Limit: ${max_daily_loss})"

        return True, "System healthy."
    
    def calculate_safe_trade(self, symbol: str, base_risk_pct: float, stop_loss_pips: float, max_daily_loss: float, portfolio_size: int) -> Dict[str, Any]:
        """Calculates exact lot size with ATR-adjusted SL and dynamic drawdown recovery scaling (Features #2, #6)."""
        
        # 1. First, check if we are even allowed to trade
        allowed, reason = self.is_trading_allowed(symbol, max_daily_loss, portfolio_size)
        if not allowed:
            return {"approved": False, "reason": reason}

        account = self.cache.get_account() if self.cache is not None else self.broker.getAccountInfo()
        if account is None:
            account = self.broker.getAccountInfo()

        if account is None:
            return {"approved": False, "reason": "Account data not available."}

        current_equity = account.equity
        trailing_drawdown = self.daily_high_watermark - current_equity
        actual_risk_pct = base_risk_pct
        bounce_from_low = (current_equity - self.daily_low_watermark) / self.daily_low_watermark if self.daily_low_watermark > 0 else 0.0

        # --- DRAWDOWN RECOVERY MODE (Feature #2) ---
        if max_daily_loss > 0:
            dd_ratio = trailing_drawdown / max_daily_loss
            if dd_ratio > 0.8:
                actual_risk_pct = base_risk_pct * 0.25  # Deep DD: Cut size by 75%
                self.notify(f"⚠️ Critical Drawdown ({dd_ratio:.0%}). Recovery Mode: Risk cut to {actual_risk_pct}%.")
            elif dd_ratio > 0.5:
                actual_risk_pct = base_risk_pct * 0.5   # Moderate DD: Cut size by 50%
                self.notify(f"⚠️ Elevated Drawdown ({dd_ratio:.0%}). Recovery Mode: Risk cut to {actual_risk_pct}%.")

        # ── Lock balance: risk only from tradeable portion ────────────────
        tradeable = self.lock_guard.tradeable_balance(account.balance)
        if tradeable <= 0:
            return {
                "approved": False,
                "reason": (
                    f"Lock balance (${self.lock_guard.effective_lock(account.balance):,.2f}) "
                    f"consumes the entire account. No tradeable balance remaining."
                )
            }

        max_risk_usd = tradeable * (actual_risk_pct / 100)

        # ── Balance-based pip sizing (replaces raw ATR × 1.5) ─────────────
        dynamic_targets = self.targeter.calculate_targets(symbol)
        atr_pips        = dynamic_targets.get("atr_pips", 0.0)
        safe_sl_pips    = self.balance_pip_sizer.get_sl_pips(
            tradeable_balance = tradeable,
            atr_pips          = atr_pips,
        )
        self.notify(
            f"[RiskManager] {self.balance_pip_sizer.describe(tradeable, atr_pips)}",
            # Remove this notify line in production if it's too verbose.
            # It's useful during tuning to see exactly what pip was chosen.
        )
        
        symbol_info = self.cache.get_symbol_info(symbol) if self.cache is not None else mt5.symbol_info(symbol)
        if symbol_info is None:
            return {"approved": False, "reason": "Symbol info missing."}

        pip_multiplier = 1.0 if any(x in symbol for x in ["BTC", "ETH"]) else 10.0
        safe_sl_points = int(safe_sl_pips * pip_multiplier)
        
        optimal_lots = self.calculate_position_size(symbol, max_risk_usd, safe_sl_points)
        
        if optimal_lots == 0.0:
             return {"approved": False, "reason": "Volatility/Spread too high for minimum lot size."}

        return {
            "approved": True,
            "reason": "Clearance granted.",
            "symbol": symbol,
            "lots": optimal_lots,
            "risk_usd": max_risk_usd,
            "applied_risk_pct": actual_risk_pct,
            "stop_loss_pips": safe_sl_pips
        }
    
    def calculate_micro_lot(self, symbol: Optional[str] = None) -> float:
        """Return the smallest tradable volume for the symbol, falling back to 0.01 if unknown."""
        if symbol:
            symbol_info = self.cache.get_symbol_info(symbol) if self.cache is not None else mt5.symbol_info(symbol)
            if symbol_info is not None and symbol_info.get("volume_min", 0) and symbol_info.get("volume_min", 0) > 0:
                return round(symbol_info["volume_min"], 2)

        return 0.01  # Default micro-lot for MT5 if symbol-specific data is unavailable

    def calculate_position_size(self, symbol: str, risk_amount_usd: float, stop_loss_points: int) -> float:
        """
        Dynamically calculates the correct lot size based on the specific asset's tick value.
        
        :param symbol: The ticker (e.g., 'EURUSD', 'XAUUSD')
        :param risk_amount_usd: The maximum amount of money willing to lose on this trade.
        :param stop_loss_points: The stop loss distance in POINTS (not pips). 
                                Points = the smallest price movement (tick_size).
        """
        symbol_info = self.cache.get_symbol_info(symbol) if self.cache is not None else mt5.symbol_info(symbol)
        if symbol_info is None:
            # [Debug] Failed to get symbol info
            return 0.0

        # 1. Fetch dynamic tick data from the broker or cache
        tick_value = float(symbol_info.get("trade_tick_value", 0.0))
        tick_size = float(symbol_info.get("trade_tick_size", 0.0))
        min_lot = float(symbol_info.get("volume_min", 0.0))
        max_lot = float(symbol_info.get("volume_max", 0.0))
        step_lot = float(symbol_info.get("volume_step", 0.0))

        if tick_value == 0 or tick_size == 0 or step_lot == 0:
            # [Debug] Tick value/size/step is invalid; cannot calculate risk
            return 0.0

        # 2. Calculate the monetary risk for 1 standard lot
        # If 1 lot moves by the stop loss distance, how much money do we lose?
        stop_loss_points = abs(stop_loss_points)
        if stop_loss_points == 0:
            return 0.0

        risk_per_1_lot = stop_loss_points * tick_value
        if risk_per_1_lot == 0.0:
            return 0.0

        # 3. Calculate exact required lot size to match our target risk
        raw_lot_size = risk_amount_usd / risk_per_1_lot

        # 4. Normalize the lot size to broker rules (min, max, and step)
        min_lot = float(symbol_info.get("volume_min", 0.0))
        max_lot = float(symbol_info.get("volume_max", 0.0))
        step_lot = float(symbol_info.get("volume_step", 0.0))

        if step_lot <= 0 or min_lot <= 0:
            return 0.0

        # Round down to the nearest allowed step (e.g., 0.01 micro lots)
        # Using floor with step normalization ensures precision
        clean_lot_size = math.floor(raw_lot_size / step_lot) * step_lot

        # 5. Clamp limits to prevent broker rejection errors
        if clean_lot_size < min_lot:
            # [Silent] Risk too small; rejecting trade
            return 0.0 
            
        if clean_lot_size > max_lot:
            # [Silent] Clamping to maximum allowed
            return max_lot

        # Return rounded to 2 decimal places to avoid MT5 floating point rejection errors
        return round(clean_lot_size, 2)


class DynamicRiskTargeter:
    """
    An independent engine that calculates structural Stop Loss and Take Profit
    targets using ATR and recent swing highs/lows.
    """

    def __init__(self, broker):
        self.broker = broker

    def calculate_targets(self, symbol: str) -> dict:
        if not symbol:
            return {
                "atr_pips": 0.0,
                "sl_buy_pips": 0.0,
                "tp_buy_pips": 0.0,
                "sl_sell_pips": 0.0,
                "tp_sell_pips": 0.0,
            }

        try:
            df = self.broker.get_historical_rates(symbol, timeframe="H1", count=25)
            if df is None or df.empty or len(df) < 14:
                return {}

            df = df.copy()
            df["prev_close"] = df["close"].shift(1)
            df["tr1"] = df["high"] - df["low"]
            df["tr2"] = (df["high"] - df["prev_close"]).abs()
            df["tr3"] = (df["low"] - df["prev_close"]).abs()
            df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
            atr = df["tr"].rolling(window=14, min_periods=14).mean().iloc[-1]
            if pd.isna(atr):
                return {}

            if "XAU" in symbol.upper() or "XAG" in symbol.upper():
                pip_multiplier = 100.0
            elif "JPY" in symbol.upper():
                pip_multiplier = 100.0
            else:
                pip_multiplier = 10000.0
            atr_pips = atr * pip_multiplier
            current_price = df["close"].iloc[-1]
            recent_high = df["high"].rolling(window=20, min_periods=1).max().iloc[-1]
            recent_low = df["low"].rolling(window=20, min_periods=1).min().iloc[-1]

            sl_buy_pips = max((current_price - recent_low) * pip_multiplier, atr_pips)
            tp_buy_pips = max((recent_high - current_price) * pip_multiplier * 1.5, atr_pips)
            sl_sell_pips = max((recent_high - current_price) * pip_multiplier, atr_pips)
            tp_sell_pips = max((current_price - recent_low) * pip_multiplier * 1.5, atr_pips)

            return {
                "atr_pips": round(atr_pips, 1),
                "sl_buy_pips": round(sl_buy_pips, 1),
                "tp_buy_pips": round(tp_buy_pips, 1),
                "sl_sell_pips": round(sl_sell_pips, 1),
                "tp_sell_pips": round(tp_sell_pips, 1),
            }
        except Exception:
            return {}


class TrailingStopManager:
    """
    Background worker that continuously monitors open positions, 
    secures breakeven, and locks in 50% of peak profits.
    """

    def __init__(self, broker, targeter, trail_atr_multiplier: float = 1.5):
        self.broker = broker
        self.targeter = targeter
        self.trail_multiplier = trail_atr_multiplier
        self.running = False
        self._thread = None
        self.peak_prices = {}

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("[System] 🛡️ Advanced Trailing Stop & Profit Locker online.")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)

    def _verify_order_levels(self, ticket: int, symbol: str, expected_sl: float, expected_tp: float = None, tolerance: float = 1e-5) -> bool:
        positions = self.broker.getPositions() or []
        pos = next((p for p in positions if p.ticket == ticket), None)
        if pos is None:
            print(f"[Profit Lock] ❌ Verification failed for {symbol} ticket {ticket}: position not found after order-level update.")
            return False

        actual_sl = float(pos.sl or 0.0)
        if abs(actual_sl - expected_sl) > tolerance:
            print(f"[Profit Lock] ❌ SL enforcement error for {symbol} ticket {ticket}: expected {expected_sl}, got {actual_sl}.")
            return False

        if expected_tp is not None:
            actual_tp = float(pos.tp or 0.0)
            if abs(actual_tp - expected_tp) > tolerance:
                print(f"[Profit Lock] ❌ TP enforcement error for {symbol} ticket {ticket}: expected {expected_tp}, got {actual_tp}.")
                return False

        return True

    def _monitor_loop(self):
        while self.running:
            if not self.broker.connected:
                time.sleep(5)
                continue

            positions = self.broker.getPositions()
            if positions:
                self._process_positions(positions)

            time.sleep(5)

    def _process_positions(self, positions):
        # 1. Clean up closed trades from our memory to prevent memory leaks
        active_tickets = [p.ticket for p in positions]
        keys_to_remove = [k for k in self.peak_prices.keys() if k not in active_tickets]
        for k in keys_to_remove:
            del self.peak_prices[k]

        # 2. Process each active trade
        for pos in positions:
            symbol = pos.symbol
            ticket = pos.ticket
            order_type = pos.type
            current_sl = float(pos.sl or 0.0)
            open_price = float(pos.price_open)
            current_price = float(pos.price_current)

            risk_data = self.targeter.calculate_targets(symbol)
            atr_pips = risk_data.get("atr_pips")
            if not atr_pips or atr_pips <= 0:
                continue

            if "XAU" in symbol.upper() or "XAG" in symbol.upper():
                pip_multiplier = 100.0
            elif "JPY" in symbol.upper():
                pip_multiplier = 100.0
            else:
                pip_multiplier = 10000.0
            breakeven_trigger_distance = (atr_pips * 0.8) / pip_multiplier

            if ticket not in self.peak_prices:
                self.peak_prices[ticket] = current_price

            if order_type == 0:
                if current_price > self.peak_prices[ticket]:
                    self.peak_prices[ticket] = current_price

                peak = self.peak_prices[ticket]
                profit_distance = peak - open_price

                if current_price > (open_price + breakeven_trigger_distance):
                    if current_sl < open_price:
                        be_price = round(open_price + (2.0 / pip_multiplier), 5)
                        if self.broker.modify_position(ticket, symbol, be_price):
                            if self._verify_order_levels(ticket, symbol, be_price):
                                print(f"[Profit Lock] 🛡️ {symbol} is now RISK FREE at {be_price}")
                                current_sl = be_price
                            else:
                                print(f"[Profit Lock] ❌ Failed to enforce breakeven SL for {symbol} ticket {ticket}.")
                        else:
                            print(f"[Profit Lock] ❌ Broker failed to apply breakeven SL for {symbol} ticket {ticket}.")

                if profit_distance > (breakeven_trigger_distance * 1.5):
                    seventy_five_percent_mark = peak - (profit_distance * 0.75)
                    if seventy_five_percent_mark > current_sl:
                        new_sl = round(seventy_five_percent_mark, 5)
                        if self.broker.modify_position(ticket, symbol, new_sl):
                            if self._verify_order_levels(ticket, symbol, new_sl):
                                print(f"[Profit Lock] 💰 Raised 75% Profit Lock on {symbol} to {new_sl}")
                            else:
                                print(f"[Profit Lock] ❌ Failed to verify 75% SL lock for {symbol} ticket {ticket}.")
                        else:
                            print(f"[Profit Lock] ❌ Broker failed to apply 75% SL lock for {symbol} ticket {ticket}.")

            elif order_type == 1:
                if current_price < self.peak_prices[ticket]:
                    self.peak_prices[ticket] = current_price

                peak = self.peak_prices[ticket]
                profit_distance = open_price - peak

                if current_price < (open_price - breakeven_trigger_distance):
                    if current_sl == 0.0 or current_sl > open_price:
                        be_price = round(open_price - (2.0 / pip_multiplier), 5)
                        if self.broker.modify_position(ticket, symbol, be_price):
                            if self._verify_order_levels(ticket, symbol, be_price):
                                print(f"[Profit Lock] 🛡️ {symbol} is now RISK FREE at {be_price}")
                                current_sl = be_price
                            else:
                                print(f"[Profit Lock] ❌ Failed to enforce breakeven SL for {symbol} ticket {ticket}.")
                        else:
                            print(f"[Profit Lock] ❌ Broker failed to apply breakeven SL for {symbol} ticket {ticket}.")

                if profit_distance > (breakeven_trigger_distance * 1.5):
                    seventy_five_percent_mark = peak + (profit_distance * 0.75)
                    if current_sl == 0.0 or seventy_five_percent_mark < current_sl:
                        new_sl = round(seventy_five_percent_mark, 5)
                        if self.broker.modify_position(ticket, symbol, new_sl):
                            if self._verify_order_levels(ticket, symbol, new_sl):
                                print(f"[Profit Lock] 💰 Lowered 75% Profit Lock on {symbol} to {new_sl}")
                            else:
                                print(f"[Profit Lock] ❌ Failed to verify 75% SL lock for {symbol} ticket {ticket}.")
                        else:
                            print(f"[Profit Lock] ❌ Broker failed to apply 75% SL lock for {symbol} ticket {ticket}.")


# ─────────────────────────────────────────────────────────────────────────────
# Add this class to the bottom of manager/risk_manager.py
# ─────────────────────────────────────────────────────────────────────────────

class ProfitGuard:
    """
    Protects floating profit from full reversal.

    Problem solved
    --------------
    The bot opens a trade, it drifts into profit, then slowly gives every
    penny back before the fixed SL finally fires — net result: a loss on
    what was a winning trade.

    Solution
    --------
    Track each ticket's *peak* floating profit.  When profit retraces
    a tier-specific percentage from that peak, close the position and
    lock in whatever is left.

    Tiers (retracement % that triggers close)
    ------------------------------------------
    peak < $1.00   →  70 %   (tiny profit — let it breathe, noise is high)
    peak $1 – $5   →  55 %
    peak $5 – $20  →  45 %
    peak > $20     →  35 %   (large winner — protect aggressively)

    Breakeven lock
    --------------
    Once peak profit reaches `BREAKEVEN_TRIGGER` ($2 by default), the
    position's SL is moved to the entry price.  This ensures the trade
    can never become a full loss once it has been meaningfully profitable.

    Damage-control exit
    -------------------
    If a trade whose peak was ≥ $5 crosses back through zero and reaches
    −$0.50, it is closed immediately.  The SL would eventually do this
    anyway, but the guard catches it faster and prevents a larger loss on
    a trade that was already a solid winner.

    Usage
    -----
        guard = ProfitGuard(broker, notify_callback=agent_notify)
        guard.start()   # background thread, checks every 5 s

    Called from chat.py in ARIA.__init__:
        self.profit_guard = ProfitGuard(broker, notify_callback)
        self.profit_guard.start()
    """

    ACTIVATE_PIPS: float = 0.5
    BREAKEVEN_PIPS: float = 2.0
    DAMAGE_CONTROL_PIPS: float = 5.0
    DAMAGE_LOSS_PIPS: float = 1.0
    CHECK_INTERVAL: int = 5

    # (peak_pips_ceiling, retracement_threshold)
    TIERS: list[tuple[float, float]] = [
        (1.0,          0.70),
        (5.0,          0.55),
        (20.0,         0.45),
        (float("inf"), 0.35),
    ]

    def __init__(self, broker, notify_callback=print):
        self.broker = broker
        self.notify = notify_callback

        self._peak: dict[int, float] = {}
        self._peak_pips: dict[int, float] = {}
        self._pip_val: dict[int, float] = {}
        self._peak_date: dict[int, datetime] = {}  # Daily reset tracking
        self._breakeven_set: set[int] = set()
        self._be_attempted: set[int] = set()
        self._closed_this_cycle: set[int] = set()

        self.running = False
        self._thread = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(
            target=self._guard_loop, daemon=True, name="ProfitGuard"
        )
        self._thread.start()
        print("[System] Profit Guard online.")

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=3)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def _guard_loop(self):
        while self.running:
            try:
                if self.broker.connected:
                    self._closed_this_cycle.clear()
                    self._check_all_positions()
            except Exception as exc:
                # Never crash the guard thread
                print(f"[ProfitGuard] Error: {exc}")
            time.sleep(self.CHECK_INTERVAL)

    def _check_all_positions(self):
        positions = self.broker.getPositions()

        if not positions:
            self._peak.clear()
            self._peak_pips.clear()
            self._pip_val.clear()
            self._breakeven_set.clear()
            self._be_attempted.clear()
            return

        open_tickets = {p.ticket for p in positions}

        # Purge tracking data for closed positions
        for stale in (set(self._peak) - open_tickets):
            self._peak.pop(stale, None)
            self._peak_pips.pop(stale, None)
            self._pip_val.pop(stale, None)
            self._breakeven_set.discard(stale)
            self._be_attempted.discard(stale)

        for pos in positions:
            self._evaluate(pos)


    # ── Per-position evaluation ───────────────────────────────────────────────

    def _evaluate(self, pos):
        ticket = pos.ticket
        symbol = pos.symbol
        lots = float(pos.volume)
        profit = float(pos.profit)

        # Daily reset: clear peak tracking at midnight so thresholds reset daily
        today = datetime.now().date()
        if self._peak_date.get(ticket) != today:
            self._peak.pop(ticket, None)
            self._peak_pips.pop(ticket, None)
            self._breakeven_set.discard(ticket)
            self._be_attempted.discard(ticket)
            self._peak_date[ticket] = today

        if ticket not in self._pip_val:
            self._pip_val[ticket] = _pip_value_usd(symbol, lots)
        pip_val = self._pip_val[ticket]

        activate_usd = pip_val * self.ACTIVATE_PIPS
        breakeven_usd = pip_val * self.BREAKEVEN_PIPS
        damage_control_usd = pip_val * self.DAMAGE_CONTROL_PIPS
        damage_loss_usd = -(pip_val * self.DAMAGE_LOSS_PIPS)

        current_peak = self._peak.get(ticket, 0.0)
        if profit > current_peak:
            self._peak[ticket] = profit
            current_peak = profit

        current_peak_pips = current_peak / pip_val if pip_val > 0 else 0.0
        self._peak_pips[ticket] = current_peak_pips

        if current_peak >= breakeven_usd and ticket not in self._breakeven_set:
            self._set_breakeven(pos)

        if current_peak < activate_usd:
            return

        if current_peak >= damage_control_usd and profit <= damage_loss_usd:
            self._close(pos, reason=(
                f"peaked at +${current_peak:.2f} ({current_peak_pips:.1f} pips), "
                f"now ${profit:.2f} — damage-control exit"
            ))
            return

        if profit <= 0:
            return

        retracement = (current_peak - profit) / current_peak
        threshold = self._threshold_for(current_peak_pips)

        if retracement >= threshold:
            self._close(pos, reason=(
                f"peaked at +${current_peak:.2f} ({current_peak_pips:.1f} pips), "
                f"now +${profit:.2f} ({retracement:.0%} retrace ≥ {threshold:.0%} threshold)"
            ))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _threshold_for(self, peak_pips: float) -> float:
        for ceiling, threshold in self.TIERS:
            if peak_pips < ceiling:
                return threshold
        return 0.35

    def _set_breakeven(self, pos):
        """Move the stop-loss to the trade's entry price."""
        ticket     = pos.ticket
        symbol     = pos.symbol
        entry      = float(pos.price_open)
        current_sl = float(pos.sl or 0.0)

        if ticket in self._be_attempted:
            return

        # Don't move if SL is already at or better than breakeven
        if pos.type == 0:   # BUY — sl must be ≤ entry to improve; we want ≥ entry
            if current_sl >= entry:
                self._breakeven_set.add(ticket)
                self._be_attempted.add(ticket)
                return
        else:               # SELL — sl must be ≥ entry; we want ≤ entry
            if 0 < current_sl <= entry:
                self._breakeven_set.add(ticket)
                self._be_attempted.add(ticket)
                return

        new_sl = round(entry, 5)
        ok = self.broker.modify_position(ticket, symbol, new_sl)
        self._be_attempted.add(ticket)
        if ok:
            self._breakeven_set.add(ticket)
            pip_val = self._pip_val.get(ticket, _pip_value_usd(symbol, float(pos.volume)))
            self.notify(
                f"🔐 ProfitGuard: {symbol} SL moved to breakeven @ {new_sl} "
                f"(triggered at {self.BREAKEVEN_PIPS} pips = ${pip_val * self.BREAKEVEN_PIPS:.2f})",
                priority="normal",
            )

    def _close(self, pos, reason: str):
        ticket = pos.ticket
        if ticket in self._closed_this_cycle:
            return   # already attempted this scan
        self._closed_this_cycle.add(ticket)

        self.notify(
            f"🔒 ProfitGuard: Closing {pos.symbol} — {reason}",
            priority="trade_executed",
        )
        self.broker.close_position(pos.symbol)
        self._peak.pop(ticket, None)
        self._peak_pips.pop(ticket, None)
        self._pip_val.pop(ticket, None)
        self._breakeven_set.discard(ticket)
        self._be_attempted.discard(ticket)

    # ── Inspection / chat interface ───────────────────────────────────────────

    def status(self) -> str:
        """Return a human-readable status string for use in chat responses."""
        if not self._peak:
            return "No positions currently being guarded."

        lines = ["ProfitGuard tracking:"]
        for ticket, peak in self._peak.items():
            peak_pips = self._peak_pips.get(ticket, 0.0)
            pip_val = self._pip_val.get(ticket, 0.0)
            threshold = self._threshold_for(peak_pips)
            be_locked = "✅ BE locked" if ticket in self._breakeven_set else "⚠️  no BE lock"
            lines.append(
                f"  #{ticket} | peak +${peak:.2f} ({peak_pips:.1f} pips) | "
                f"1 pip = ${pip_val:.4f} | close on {threshold:.0%} retrace | {be_locked}"
            )
        return "\n".join(lines)
    
class TradeGatekeeper:
    """
    Pre-trade gate that checks spread and session quality.

    Call gate() before any order submission.  Returns (True, "OK") when
    conditions are acceptable, or (False, reason) to block the trade.

    Parameters
    ----------
    max_spread_pips_forex   : Maximum allowed spread for forex/metals (pips).
    max_spread_pips_crypto  : Maximum allowed spread for crypto (pips).
                              Crypto spreads are naturally much wider in nominal
                              pip terms — BTC at $0.50 spread = ~5 pips, not 5000.
    avoid_asian_session     : Block trades between 00:00–07:00 UTC for non-Asian
                              forex pairs (wide spreads, thin liquidity).
    avoid_friday_close      : Block new non-crypto trades after 20:00 UTC Friday
                              (weekend gap risk).
    """

    def __init__(
        self,
        max_spread_pips:        float = 3.0,    # kept for backward compat — used as forex limit
        max_spread_pips_forex:  float = None,   # overrides max_spread_pips when set
        max_spread_pips_crypto: float = 50.0,   # crypto-specific limit in pips
        avoid_asian_session:    bool  = True,
        avoid_friday_close:     bool  = True,
    ):
        # Backward-compatible: if someone passes max_spread_pips only, use it for forex
        self.max_spread_forex   = max_spread_pips_forex if max_spread_pips_forex is not None else max_spread_pips
        self.max_spread_crypto  = max_spread_pips_crypto
        self.avoid_asian_session = avoid_asian_session
        self.avoid_friday_close  = avoid_friday_close

    def gate(self, symbol: str, broker=None) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
        Call this before submitting any order.
        """
        from datetime import datetime, timezone
        from manager.market_sessions import MarketSessionManager

        now_utc = datetime.now(timezone.utc)
        hour    = now_utc.hour
        weekday = now_utc.weekday()

        # ── Session filter ───────────────────────────────────────────────────
        mgr      = MarketSessionManager()
        category = mgr.get_symbol_category(symbol)
        tradeable, session_reason = mgr.is_symbol_tradeable(symbol, now_utc)

        if not tradeable:
            next_open = mgr.get_next_open_time(symbol, now_utc)
            return False, f"{session_reason} Opens: {next_open}."

        if category != "crypto" and self.avoid_friday_close and weekday == 4 and hour >= 20:
            return False, (
                "Friday after 20:00 UTC — weekend gap risk. "
                "No new non-crypto positions before market close."
            )

        if category == "forex" and self.avoid_asian_session and 0 <= hour < 7:
            asian_active = {"USDJPY", "AUDUSD", "NZDUSD", "CADJPY", "CHFJPY", "GBPJPY"}
            if symbol.upper() not in asian_active:
                return False, (
                    f"Asian session ({hour:02d}:00 UTC) — low liquidity for {symbol}. "
                    f"Waiting for London open (07:00 UTC)."
                )

        # ── Spread filter ────────────────────────────────────────────────────
        spread_pips = self._get_spread_pips(symbol, broker, category)
        if spread_pips is None:
            return False, f"Could not fetch tick data for {symbol}."

        # Select the appropriate limit for this instrument category
        if category == "crypto":
            spread_limit = self.max_spread_crypto
        else:
            spread_limit = self.max_spread_forex

        if spread_pips > spread_limit:
            return False, (
                f"Spread too wide: {spread_pips:.1f} pips "
                f"(limit {spread_limit:.1f} pips for {category}). "
                f"Waiting for tighter conditions."
            )

        return True, "OK"

    def _get_spread_pips(self, symbol: str, broker=None, category: str = None) -> float | None:
        """
        Calculate the current spread in pips, correctly normalised per instrument type.

        Pip sizes by category
        ---------------------
        Standard forex (5-digit, e.g. EURUSD at 1.08500):
            point = 0.00001  →  1 pip = 10 points
        JPY pairs (3-digit, e.g. USDJPY at 155.000):
            point = 0.001    →  1 pip = 10 points  (same formula)
        Metals (e.g. XAUUSD at 2300.00):
            point = 0.01     →  1 pip = 1 point    (digits = 2)
        Crypto (e.g. BTCUSD at 65000.00):
            point = 0.01     →  1 pip = 1 point    (digits = 2)
        Indices (e.g. US30 at 38000.0):
            point = 0.1      →  1 pip = 1 point

        The rule: digits 5 or 3 → divide raw spread_points by 10.
                  everything else → spread_points already represents pips.
        """
        import MetaTrader5 as mt5

        try:
            tick = None

            # Try broker helper first (avoids direct MT5 call in tests)
            if broker is not None and hasattr(broker, "get_tick_data"):
                tick_data = broker.get_tick_data(symbol)
                if tick_data and "ask" in tick_data and "bid" in tick_data:
                    tick = type("Tick", (), tick_data)()

            if tick is None:
                tick = mt5.symbol_info_tick(symbol)

            info = mt5.symbol_info(symbol)
            if not tick or not info:
                return None

            spread_points = (tick.ask - tick.bid) / info.point

            # 5-digit forex (EURUSD) and 3-digit JPY pairs both have 10 points per pip
            if info.digits in (5, 3):
                return spread_points / 10.0

            # Everything else (metals, crypto, indices): 1 point = 1 pip
            return spread_points

        except Exception:
            return None

class CorrelationGuard:
    """
    Prevents over-concentration in a single underlying currency.
 
    Works by decomposing each 6-character forex symbol into its two legs
    (e.g. EURUSD → EUR, USD) and counting how many open positions share a
    leg with the proposed new trade.
 
    Parameters
    ----------
    max_shared_legs : int
        Maximum number of times one currency may appear across open positions
        (including the proposed new trade).  Default 2 — allows one existing
        position on the same currency before blocking a new entry.
    """
 
    def __init__(self, max_shared_legs: int = 2):
        self.max_shared_legs = max_shared_legs
 
    def check(self, symbol: str, broker) -> tuple[bool, str]:
        """
        Returns (allowed: bool, reason: str).
 
        Call before opening a new position.
        """
        positions = broker.getPositions() or []
        if not positions:
            return True, "No existing positions."
 
        # Extract legs from all open positions
        open_legs: list[str] = []
        for pos in positions:
            open_legs.extend(self._legs(pos.symbol))
 
        leg_counts = Counter(open_legs)
 
        # Check proposed symbol's legs against existing counts
        proposed_legs = self._legs(symbol)
        for leg in proposed_legs:
            # Count would become leg_counts[leg] + 1 after this trade
            if leg_counts[leg] + 1 > self.max_shared_legs:
                concentrated = [
                    pos.symbol for pos in positions
                    if leg in self._legs(pos.symbol)
                ]
                return False, (
                    f"Correlation limit: {leg} already appears in "
                    f"{leg_counts[leg]} open position(s) "
                    f"({', '.join(concentrated)}). "
                    f"Max shared legs = {self.max_shared_legs}."
                )
 
        return True, "OK"
 
    @staticmethod
    def _legs(symbol: str) -> list[str]:
        """Split a 6-char forex symbol into its two 3-char currency legs."""
        sym = symbol.upper()
        # Skip non-forex instruments (metals, crypto, indices)
        if any(sym.startswith(p) for p in ["XAU", "XAG", "BTC", "ETH", "US", "GER"]):
            return []
        if len(sym) >= 6:
            return [sym[:3], sym[3:6]]
        return []
 