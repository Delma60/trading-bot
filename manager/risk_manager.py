from typing import Dict, Any, Optional, List, Tuple
import MetaTrader5 as mt5
from datetime import datetime, timezone, timedelta
import math
import pandas as pd
import time
import threading
from collections import Counter
from manager.market_sessions import MarketSessionManager
from manager.profile_manager import profile as _profile
from manager.expectancy_guard import ExpectancyGuard

def _pip_value_usd(symbol: str, lots: float) -> float:
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

class MarketConditionFilter:
    def __init__(self, broker):
        self.broker = broker

    def is_market_suitable(self, symbol: str) -> Tuple[bool, str]:
        df = self.broker.get_historical_rates(symbol, "H1", 50)
        if df is None or df.empty or len(df) < 50:
            return True, "Not enough data, skipping filter."

        df = df.copy()
        df['tr'] = df['high'] - df['low']
        recent_atr = df['tr'].tail(3).mean()
        normal_atr = df['tr'].tail(40).mean()

        if recent_atr > normal_atr * 2.5:
            return False, "Volatility spike detected — likely news event. Waiting."

        if 'tick_volume' in df.columns:
            recent_vol = df['tick_volume'].tail(3).mean()
            avg_vol = df['tick_volume'].tail(40).mean()
            if recent_vol < avg_vol * 0.3:
                return False, "Volume too low — market is sleeping. No entries."

        return True, "Market conditions suitable."

class SmartReEntrySystem:
    def __init__(self):
        self.stopped_out_trades: Dict[str, Dict[str, Any]] = {}

    def record_stop_out(self, symbol: str, price: float, direction: str):
        self.stopped_out_trades[symbol] = {
            "time": datetime.now(),
            "price": price,
            "direction": direction,
            "re_entered": False
        }

    def mark_reentered(self, symbol: str):
        if symbol in self.stopped_out_trades:
            self.stopped_out_trades[symbol]["re_entered"] = True

    def check_reentry_validity(self, symbol: str, current_price: float, new_signal_direction: str) -> bool:
        record = self.stopped_out_trades.get(symbol)
        if not record or record["re_entered"]:
            return False
            
        time_elapsed = (datetime.now() - record["time"]).total_seconds() / 3600
        if time_elapsed <= 4.0 and new_signal_direction == record["direction"]:
            if new_signal_direction == "BUY" and current_price <= record["price"]:
                return True
            if new_signal_direction == "SELL" and current_price >= record["price"]:
                return True
        return False

class LockBalanceGuard:
    def __init__(self, lock_amount: float = 0.0, lock_pct: float = 0.0):
        self.lock_amount = max(0.0, lock_amount)
        self.lock_pct    = max(0.0, min(lock_pct, 0.99))

    def effective_lock(self, balance: float) -> float:
        if balance <= 0:
            return 0.0
        pct_lock = balance * self.lock_pct
        return max(self.lock_amount, pct_lock)

    def tradeable_balance(self, balance: float) -> float:
        return max(0.0, balance - self.effective_lock(balance))

    def is_locked_out(self, balance: float) -> bool:
        return self.tradeable_balance(balance) <= 0

    def status_str(self, balance: float) -> str:
        lock   = self.effective_lock(balance)
        trade  = self.tradeable_balance(balance)
        if lock <= 0:
            return f"No lock balance set. Full balance ${balance:,.2f} is tradeable."
        pct_of_balance = (lock / balance * 100) if balance > 0 else 0
        return (
            f"Lock balance: ${lock:,.2f} ({pct_of_balance:.1f}% of ${balance:,.2f}). "
            f"Tradeable: ${trade:,.2f}."
        )

    def update(self, lock_amount: Optional[float] = None, lock_pct: Optional[float] = None):
        if lock_amount is not None:
            self.lock_amount = max(0.0, lock_amount)
        if lock_pct is not None:
            self.lock_pct = max(0.0, min(lock_pct, 0.99))

class BalancePipSizer:
    DEFAULT_TIERS: List[Tuple[float, float]] = [
        (0,    12),
        (200,  15),
        (500,  18),
        (1000, 25),
        (2500, 35),
        (5000, 50),
    ]

    def __init__(
        self,
        tiers:           Optional[List[Tuple[float, float]]] = None,
        atr_floor_ratio: float = 0.50,
        atr_ceil_ratio:  float = 1.50,
    ):
        self.tiers           = sorted(tiers or self.DEFAULT_TIERS, key=lambda t: t[0])
        self.atr_floor_ratio = atr_floor_ratio
        self.atr_ceil_ratio  = atr_ceil_ratio

    def base_pips(self, tradeable_balance: float) -> float:
        result = self.tiers[0][1]
        for min_bal, pips in self.tiers:
            if tradeable_balance >= min_bal:
                result = pips
        return float(result)

    def get_sl_pips(self, tradeable_balance: float, atr_pips: float = 0.0) -> float:
        base = self.base_pips(tradeable_balance)
        if atr_pips <= 0:
            return base

        floor_pips = base * self.atr_floor_ratio
        ceil_pips  = base * self.atr_ceil_ratio

        final = max(floor_pips, min(atr_pips, ceil_pips))
        return round(final, 1)

    def describe(self, tradeable_balance: float, atr_pips: float = 0.0) -> str:
        base  = self.base_pips(tradeable_balance)
        final = self.get_sl_pips(tradeable_balance, atr_pips)
        atr_str = f", ATR {atr_pips:.1f}p" if atr_pips > 0 else " (no ATR)"
        return (
            f"Balance ${tradeable_balance:,.0f} → base {base:.0f}p"
            f"{atr_str} → SL {final:.1f}p "
            f"[range {base * self.atr_floor_ratio:.0f}–{base * self.atr_ceil_ratio:.0f}p]"
        )

class RiskManager:
    LOSS_STREAK_LIMIT = 2
    LOSS_STREAK_PAUSE_MINUTES = 60

    def __init__(self, broker, cache=None, max_open_trades: int = 3, min_margin_level: float = 150.0, notify_callback=print,
                 pyramid_min_pips: float = 1.0, spread_tolerance_pips: float = 1.0):
        self.broker = broker
        self.cache = cache
        self.max_open_trades = max_open_trades
        self.min_margin_level = min_margin_level
        self.notify = notify_callback
        self.pyramid_min_pips = pyramid_min_pips
        self.spread_tolerance_pips = spread_tolerance_pips

        self._loss_lock = threading.Lock()
        self._sync_lock = threading.Lock() # Global operations level thread barrier

        self.daily_high_watermark = None
        self.daily_low_watermark = 0.0
        self.watermark_date = None

        self.targeter = DynamicRiskTargeter(broker)
        self.reentry_system = SmartReEntrySystem()

        r = _profile.risk()
        self.lock_guard = LockBalanceGuard(
            lock_amount = r.lock_amount,
            lock_pct    = r.lock_pct_decimal,
        )
        self.balance_pip_sizer = BalancePipSizer()
        self._loss_cooldown_until = {}  
        self._consecutive_losses = {}  

    @staticmethod
    def get_asset_class(symbol: str):
        from manager.profile_manager import profile
        return profile.get_asset_class(symbol)

    def compute_correlation_matrix(self, symbol_registry, window: int = 100) -> dict:
        import pandas as pd
        symbols = symbol_registry.get_all_symbols()
        price_data = {}
        for sym in symbols:
            df = symbol_registry.get_ohlcv(sym, limit=window)
            if df is not None and 'close' in df.columns:
                price_data[sym] = df['close'].values[-window:]
        if len(price_data) < 2:
            return {}
        df_prices = pd.DataFrame(price_data)
        corr = df_prices.corr()
        result = {}
        for s1 in symbols:
            for s2 in symbols:
                if s1 != s2:
                    result[(s1, s2)] = corr.loc[s1, s2]
        return result

    def is_correlation_safe(self, symbol, direction, open_positions, symbol_registry, threshold=0.85) -> tuple:
        corr_matrix = self.compute_correlation_matrix(symbol_registry)
        for pos in open_positions:
            if pos['direction'] == direction:
                pair = (symbol, pos['symbol'])
                if pair in corr_matrix and abs(corr_matrix[pair]) >= threshold:
                    return (False, f"{symbol} and {pos['symbol']} are highly correlated ({corr_matrix[pair]:.2f}) in the same direction.")
        return (True, "OK")

    def record_loss(self, symbol: str, notify_callback=print):
        from manager.profile_manager import profile
        asset_class = profile.get_asset_class(symbol)
        key = asset_class or symbol
        self._consecutive_losses[key] = self._consecutive_losses.get(key, 0) + 1
        if self._consecutive_losses[key] >= self.LOSS_STREAK_LIMIT:
            pause_until = datetime.now() + timedelta(minutes=self.LOSS_STREAK_PAUSE_MINUTES)
            self._loss_cooldown_until[key] = pause_until
            self._consecutive_losses[key] = 0
            notify_callback(f"⏸ {key}: {self.LOSS_STREAK_LIMIT} consecutive losses — paused for {self.LOSS_STREAK_PAUSE_MINUTES}m")

    def record_win(self, symbol: str):
        from manager.profile_manager import profile
        asset_class = profile.get_asset_class(symbol)
        key = asset_class or symbol
        self._consecutive_losses[key] = 0

    def is_loss_paused(self, symbol: str) -> bool:
        from manager.profile_manager import profile
        asset_class = profile.get_asset_class(symbol)
        key = asset_class or symbol
        until = self._loss_cooldown_until.get(key)
        return bool(until and datetime.now() < until)
    
    def record_stop_out_position(self, symbol: str, close_price: float, position_type: int):
        direction = "BUY" if position_type == 0 else "SELL"
        self.reentry_system.record_stop_out(symbol, close_price, direction)

    def _get_realized_daily_loss(self) -> float:
        try:
            today_start = int(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
            today_end = int(datetime.now().timestamp())
            
            deals = mt5.history_deals_get(today_start, today_end)
            if deals is None or len(deals) == 0:
                return 0.0
            
            realized_pnl = 0.0
            for deal in deals:
                if deal.profit < 0:
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

    def _check_existing_positions(self, symbol: str, symbol_positions: list) -> Tuple[bool, str]:
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

    def is_trading_allowed(self, symbol: str, max_daily_loss: float, portfolio_size: int) -> Tuple[bool, str]:
        from manager.profile_manager import profile
        account = self.cache.get_account() if self.cache is not None else self.broker.getAccountInfo()
        if account is None:
            account = self.broker.getAccountInfo()

        if not account:
            return False, "Could not fetch account data from broker."

        positions = self.cache.get_positions() if self.cache is not None else self.broker.getPositions()
        if positions is None:
            positions = self.broker.getPositions()

        tradeable = self.lock_guard.tradeable_balance(account.balance)
        if tradeable <= 0:
            return False, (
                f"Lock balance (${self.lock_guard.effective_lock(account.balance):,.2f}) "
                f"consumes the full account — no tradeable balance."
            )

        asset_class = profile.get_asset_class(symbol)
        class_max_trades = profile.max_open_trades(symbol)
        if asset_class:
            open_trades_in_class = sum(1 for p in positions if profile.get_asset_class(p.symbol) == asset_class)
            if open_trades_in_class >= class_max_trades:
                return False, f"{asset_class.title()} exposure reached ({open_trades_in_class}/{class_max_trades} trades)."

        current_open_trades = len(positions) if positions else 0
        if current_open_trades >= self.max_open_trades:
            return False, f"Global exposure reached ({current_open_trades}/{self.max_open_trades} trades)."

        if positions:
            symbol_positions = [p for p in positions if p.symbol == symbol]
            symbol_trade_count = len(symbol_positions)

            if symbol_trade_count >= 3:
                return False, f"Max symbol exposure reached for {symbol} ({symbol_trade_count}/3 trades)."

            allowed_b, reason_b = self._check_existing_positions(symbol, symbol_positions)
            if not allowed_b:
                return False, reason_b
        
        if account.margin_level and account.margin_level < self.min_margin_level:
            return False, f"Margin level too low ({account.margin_level:.1f}%). Minimum is {self.min_margin_level}%."

        with self._loss_lock:
            current_equity = account.equity
            today = datetime.now().date()

            # FIXED: Robust daily reference retrieval prevents drawdown watermarks from defaulting to depleted account balances
            if self.watermark_date != today:
                today_start = int(datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp())
                today_end = int(datetime.now().timestamp())
                deals = mt5.history_deals_get(today_start, today_end)

                net_today_pnl = 0.0
                if deals:
                    for deal in deals:
                        net_today_pnl += deal.profit + getattr(deal, 'commission', 0.0) + getattr(deal, 'fee', 0.0)

                start_of_day_balance = account.balance - net_today_pnl

                self.daily_high_watermark = max(start_of_day_balance, account.balance, current_equity)
                self.daily_low_watermark = min(start_of_day_balance, account.balance, current_equity)
                self.watermark_date = today
                self._start_of_day_balance = start_of_day_balance
            else:
                start_of_day_balance = getattr(self, '_start_of_day_balance', account.balance)
                if current_equity > self.daily_high_watermark:
                    self.daily_high_watermark = current_equity
                elif current_equity < self.daily_low_watermark:
                    self.daily_low_watermark = current_equity

            daily_loss_from_start = start_of_day_balance - current_equity

            if max_daily_loss > 0 and daily_loss_from_start >= max_daily_loss:
                return False, f"Daily loss limit reached. Net decline from baseline: ${daily_loss_from_start:,.2f}."
        return True, "System healthy."
   # ── Excerpt drop-in for manager/risk_manager.py ───────────────────────────────

    def calculate_safe_trade(self, symbol: str, base_risk_pct: float, stop_loss_pips: float, max_daily_loss: float, portfolio_size: int) -> dict:
        """Evaluates live market indicators and active drawdown levels to authorize trade parameters."""
        allowed, reason = self.is_trading_allowed(symbol, max_daily_loss, portfolio_size)
        with self._sync_lock:
            if not allowed:
                return {"approved": False, "reason": reason}

            # ... account data retrieval and active drawdown protection layers ...
            account = self.cache.get_account() if self.cache is not None else self.broker.getAccountInfo()
            if account is None:
                account = self.broker.getAccountInfo()

            if account is None:
                return {"approved": False, "reason": "Account data not available."}
            
            if self.daily_high_watermark is None:
                self.daily_high_watermark = account.equity
                
            current_equity = account.equity
            trailing_drawdown = max(0.0, self.daily_high_watermark - current_equity)
            actual_risk_pct = base_risk_pct
            

            if max_daily_loss > 0:
                dd_ratio = trailing_drawdown / max_daily_loss
                if dd_ratio > 0.8:
                    actual_risk_pct = base_risk_pct * 0.25
                    self.notify(f"⚠️ Critical Drawdown ({dd_ratio:.0%}). Recovery Mode: Risk cut to {actual_risk_pct}%.")
                elif dd_ratio > 0.5:
                    actual_risk_pct = base_risk_pct * 0.5
                    self.notify(f"⚠️ Elevated Drawdown ({dd_ratio:.0%}). Recovery Mode: Risk cut to {actual_risk_pct}%.")


            tradeable = self.lock_guard.tradeable_balance(account.balance)
            if tradeable <= 0:
                return {
                    "approved": False,
                    "reason": (
                        f"Lock balance (${self.lock_guard.effective_lock(account.balance):,.2f}) "
                        f"consumes the entire account. No tradeable balance remaining."
                    )
                }


            dynamic_targets = self.targeter.calculate_targets(symbol)
            atr_pips        = dynamic_targets.get("atr_pips", 0.0)
            safe_sl_pips    = self.balance_pip_sizer.get_sl_pips(tradeable, atr_pips)
            max_risk_usd = tradeable * (actual_risk_pct / 100)
            
            
            # Bug 5 Fixed: Resolve valid Take Profit targets from dynamic feature extraction or core defaults
            safe_tp_pips    = dynamic_targets.get("tp_buy_pips") or (_profile.risk(symbol).take_profit_pips)
            
            symbol_info = self.cache.get_symbol_info(symbol) if self.cache else mt5.symbol_info(symbol)
            if not symbol_info:
                return {"approved": False, "reason": "Symbol info missing."}

            pip_multiplier = 1.0 if any(x in symbol for x in ["BTC", "ETH"]) else 10.0
            safe_sl_points = int(safe_sl_pips * pip_multiplier)
            optimal_lots = self.calculate_position_size(symbol, max_risk_usd, safe_sl_points)
            
            if optimal_lots == 0.0:
                return {"approved": False, "reason": "Spread or volatility too high for operational minimum lots."}

            return {
                "approved": True,
                "reason": "Clearance granted.",
                "symbol": symbol,
                "lots": optimal_lots,
                "risk_usd": max_risk_usd,
                "applied_risk_pct": actual_risk_pct,
                "stop_loss_pips": safe_sl_pips,
                # Bug 5 Fixed: Included take_profit_pips directly inside output validation mapping
                "take_profit_pips": safe_tp_pips
            }
    def calculate_micro_lot(self, symbol: Optional[str] = None) -> float:
        if symbol:
            symbol_info = self.cache.get_symbol_info(symbol) if self.cache is not None else mt5.symbol_info(symbol)
            if symbol_info is not None and symbol_info.get("volume_min", 0) and symbol_info.get("volume_min", 0) > 0:
                return round(symbol_info["volume_min"], 2)
        return 0.01

    def calculate_position_size(self, symbol: str, risk_amount_usd: float, stop_loss_points: int) -> float:
        # 1. Fetch account state securely to enforce absolute equity limits
        account = self.cache.get_account() if self.cache is not None else self.broker.getAccountInfo()
        if account is None:
            account = self.broker.getAccountInfo()

        # 2. Enforce Hard Absolute Risk Cap: No individual position can ever risk > 3.0% of available equity
        if account and getattr(account, 'equity', 0.0) > 0:
            max_absolute_risk_cap = account.equity * 0.03
            risk_amount_usd = min(risk_amount_usd, max_absolute_risk_cap)

        # 3. Retrieve symbol specifications
        symbol_info = self.cache.get_symbol_info(symbol) if self.cache is not None else mt5.symbol_info(symbol)
        if symbol_info is None:
            return 0.0

        tick_value = float(symbol_info.get("trade_tick_value", 0.0))
        tick_size  = float(symbol_info.get("trade_tick_size", 0.0))
        min_lot    = float(symbol_info.get("volume_min", 0.0))
        max_lot    = float(symbol_info.get("volume_max", 0.0))
        step_lot   = float(symbol_info.get("volume_step", 0.0))

        if tick_value == 0 or tick_size == 0 or step_lot == 0:
            return 0.0

        stop_loss_points = abs(stop_loss_points)
        if stop_loss_points == 0:
            return 0.0

        risk_per_1_lot = stop_loss_points * tick_value
        if risk_per_1_lot == 0.0:
            return 0.0

        raw_lot_size = risk_amount_usd / risk_per_1_lot

        if step_lot <= 0 or min_lot <= 0:
            return 0.0

        clean_lot_size = math.floor(raw_lot_size / step_lot) * step_lot

        # Strictly reject any trade where the minimum lot size exceeds the capped dollar risk limit
        if clean_lot_size < min_lot:
            risk_at_min_lot = min_lot * risk_per_1_lot
            if risk_at_min_lot > risk_amount_usd:
                return 0.0 
            return min_lot 
            
        if clean_lot_size > max_lot:
            return max_lot

        return round(clean_lot_size, 2)
    
class DynamicRiskTargeter:
    def __init__(self, broker):
        self.broker = broker

    def calculate_targets(self, symbol: str) -> dict:
        if not symbol:
            return {
                "atr_pips": 0.0, "sl_buy_pips": 0.0, "tp_buy_pips": 0.0,
                "sl_sell_pips": 0.0, "tp_sell_pips": 0.0,
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
    MAX_MODIFY_ATTEMPTS = 5

    def __init__(self, broker, targeter, trail_atr_multiplier: float = 1.5, _trail_lock=None):
        self.broker = broker
        self.targeter = targeter
        self.trail_multiplier = trail_atr_multiplier
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self.peak_prices: Dict[int, float] = {}
        self._failed_attempts: dict[int, int] = {}
        self._trail_lock = _trail_lock or threading.Lock()

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

    def _verify_order_levels(self, ticket: int, symbol: str, expected_sl: float, expected_tp: Optional[float] = None, tolerance: float = 1e-5) -> bool:
        positions = self.broker.getPositions() or []
        pos = next((p for p in positions if p.ticket == ticket), None)
        if pos is None:
            return False
        actual_sl = float(pos.sl or 0.0)
        if abs(actual_sl - expected_sl) > tolerance:
            return False
        if expected_tp is not None:
            actual_tp = float(pos.tp or 0.0)
            if abs(actual_tp - expected_tp) > tolerance:
                return False
        return True

    def _monitor_loop(self):
        while self.running:
            if not self.broker.connected:
                time.sleep(5)
                continue
            with self._trail_lock:
                positions = self.broker.getPositions()
                if positions:
                    self._process_positions(positions)
            time.sleep(5)

    def _min_stop_distance(self, symbol: str) -> float:
        info = getattr(self.broker, "get_symbol_info", lambda s: None)(symbol)
        if info and hasattr(info, "stops_level") and hasattr(info, "point"):
            return info.stops_level * info.point
        if "XAU" in symbol.upper() or "XAG" in symbol.upper():
            return 0.5
        elif "JPY" in symbol.upper():
            return 0.05
        else:
            return 0.0005

    def _try_modify(self, ticket, symbol, new_sl, label):
        if self._failed_attempts.get(ticket, 0) >= self.MAX_MODIFY_ATTEMPTS:
            return False 
        ok = self.broker.modify_position(ticket, symbol, new_sl)
        if not ok:
            self._failed_attempts[ticket] = self._failed_attempts.get(ticket, 0) + 1
        else:
            self._failed_attempts.pop(ticket, None) 
        return ok

    def _process_positions(self, positions):
        active_tickets = [p.ticket for p in positions]
        keys_to_remove = [k for k in self.peak_prices.keys() if k not in active_tickets]
        for k in keys_to_remove:
            del self.peak_prices[k]
            self._failed_attempts.pop(k, None)

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
            min_stop = self._min_stop_distance(symbol)

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
                        if (be_price - open_price) >= min_stop:
                            self._try_modify(ticket, symbol, be_price, "breakeven")

                if profit_distance > (breakeven_trigger_distance * 1.5):
                    seventy_five_percent_mark = peak - (profit_distance * 0.75)
                    if seventy_five_percent_mark > current_sl:
                        if (seventy_five_percent_mark - open_price) >= min_stop:
                            self._try_modify(ticket, symbol, round(seventy_five_percent_mark, 5), "75% lock")

            elif order_type == 1:
                if current_price < self.peak_prices[ticket]:
                    self.peak_prices[ticket] = current_price
                peak = self.peak_prices[ticket]
                profit_distance = open_price - peak

                if current_price < (open_price - breakeven_trigger_distance):
                    if current_sl == 0.0 or current_sl > open_price:
                        be_price = round(open_price - (2.0 / pip_multiplier), 5)
                        if (open_price - be_price) >= min_stop:
                            self._try_modify(ticket, symbol, be_price, "breakeven")

                if profit_distance > (breakeven_trigger_distance * 1.5):
                    seventy_five_percent_mark = peak + (profit_distance * 0.75)
                    if current_sl == 0.0 or seventy_five_percent_mark < current_sl:
                        if (open_price - seventy_five_percent_mark) >= min_stop:
                            self._try_modify(ticket, symbol, round(seventy_five_percent_mark, 5), "75% lock")

class ProfitGuard:
    BREAKEVEN_NORM_PCT: float      = 0.002   
    DAMAGE_CONTROL_NORM_PCT: float = 0.005  
    DAMAGE_LOSS_NORM_PCT: float    = 0.001   
    CHECK_INTERVAL: int            = 5

    TIERS: List[Tuple[float, float]] = [
        (0.001,        0.70),  
        (0.005,        0.55),  
        (0.02,         0.45),  
        (float("inf"), 0.35),  
    ]

    def __init__(self, broker, notify_callback=print, api_lock=None):
        self.broker = broker
        self.notify = notify_callback

        self._peak: Dict[int, float] = {}
        self._peak_pips: Dict[int, float] = {}
        self._pip_val: Dict[int, float] = {}
        self._peak_date: Dict[int, datetime] = {}
        self._breakeven_set: set[int] = set()
        self._be_attempted: set[int] = set()
        self._closed_this_cycle: set[int] = set()

        self._api_lock = api_lock or threading.Lock()
        self.running = False
        self._thread: Optional[threading.Thread] = None
        self._exp_guard = ExpectancyGuard(notify_callback=notify_callback)
        
        # Initialize internal dynamic targeter instance for real-time ATR parsing
        self.targeter = DynamicRiskTargeter(broker)

    def start(self):
        if self.running:
            return
        self.running = True
        self._exp_guard.start()
        self._thread = threading.Thread(
            target=self._guard_loop, daemon=True, name="ProfitGuard"
        )
        self._thread.start()
        print("[System] 🛡️ Advanced Dynamic ProfitGuard online.")

    def stop(self):
        self.running = False
        self._exp_guard.stop()  
        if self._thread:
            self._thread.join(timeout=3)

    def _get_dynamic_activation_usd(self, equity_base: float, pos) -> float:
        """
        Derives activation dynamically against real-time structural asset volatility (ATR).
        Falls back smoothly to continuous balance interpolation if live target structures fail.
        """
        symbol = pos.symbol
        lots = float(pos.volume)
        
        try:
            # Query active dynamic target constraints directly from local scope
            target_data = self.targeter.calculate_targets(symbol)
            atr_pips = target_data.get("atr_pips", 0.0)

            # Resolve cached or runtime pip value for precise translation
            pip_val = self._pip_val.get(pos.ticket) or _pip_value_usd(symbol, lots)
            
            # Evaluate absolute dollar cost of a 0.5 ATR directional breakout
            atr_activation_usd = (atr_pips * 0.5) * pip_val

            if atr_activation_usd > 0:
                # Ensure the target clears minimum operational spread/drag floors
                min_floor_usd = max(2.0, equity_base * 0.0005)
                return max(atr_activation_usd, min_floor_usd)
        except Exception:
            pass

        # Failsafe Fallback: Continuous balance interpolation curve (eliminates cliff effects)
        equity = max(equity_base, 100.0)
        if equity <= 500.0:
            return equity * 0.005
        elif equity <= 5000.0:
            return 2.50 + (equity - 500.0) * (7.50 / 4500.0)
        elif equity <= 20000.0:
            return 10.00 + (equity - 5000.0) * (10.00 / 15000.0)
        return 20.00 + (equity - 20000.0) * 0.0005
    
    def _guard_loop(self):
        while self.running:
            try:
                if self.broker.connected:
                    self._closed_this_cycle.clear()
                    self._check_all_positions()
            except Exception as exc:
                print(f"[ProfitGuard] Structural Daemon Exception: {exc}")
            time.sleep(self.CHECK_INTERVAL)

    def _get_equity_base(self) -> float:
        acc = self.broker.getAccountInfo()
        if not acc or acc.equity <= 0:
            return 10000.0 
            
        r = _profile.risk()
        guard = LockBalanceGuard(lock_amount=r.lock_amount, lock_pct=r.lock_pct_decimal)
        effective_equity = guard.tradeable_balance(acc.equity)
        return max(effective_equity, 100.0)

    def _check_all_positions(self):
        with self._api_lock:
            positions = self.broker.getPositions()

            if not positions:
                self._peak.clear()
                self._peak_pips.clear()
                self._pip_val.clear()
                self._breakeven_set.clear()
                self._be_attempted.clear()
                return

            open_tickets = {p.ticket for p in positions}
            for stale in (set(self._peak) - open_tickets):
                self._peak.pop(stale, None)
                self._peak_pips.pop(stale, None)
                self._pip_val.pop(stale, None)
                self._breakeven_set.discard(stale)
                self._be_attempted.discard(stale)

        eq_base = self._get_equity_base()
        for pos in positions:
            self._evaluate(pos, eq_base)

    def _evaluate(self, pos, equity_base: float):
        ticket = pos.ticket
        symbol = pos.symbol
        lots = float(pos.volume)
        profit = float(pos.profit)

        today = datetime.now().date()
        with self._api_lock:
            if self._peak_date.get(ticket) != today:
                self._peak.pop(ticket, None)
                self._peak_pips.pop(ticket, None)
                self._breakeven_set.discard(ticket)
                self._be_attempted.discard(ticket)
                self._peak_date[ticket] = today

            if ticket not in self._pip_val:
                self._pip_val[ticket] = _pip_value_usd(symbol, lots)
            pip_val = self._pip_val[ticket]

            current_peak = self._peak.get(ticket, 0.0)
            if profit > current_peak:
                self._peak[ticket] = profit
                current_peak = profit

            current_peak_pips = current_peak / pip_val if pip_val > 0 else 0.0
            self._peak_pips[ticket] = current_peak_pips

            # ── VOLATILITY-DRIVEN DYNAMISM (ATR ANCHORING) ──
            # Calculate real-time target structures directly from internal targeter engine
            target_data = self.targeter.calculate_targets(symbol)
            atr_pips = target_data.get("atr_pips", 0.0)
            
            # Evaluate the active dollar value of a single ATR movement for this position size
            atr_usd_value = atr_pips * pip_val if pip_val > 0 else 0.0

            if atr_usd_value > 0:
                # Volatility-Anchored Dynamic Thresholds
                # Scales exactly to the pair's current market behavior and live position exposure
                activate_usd       = self._get_dynamic_activation_usd(equity_base, pos)
                breakeven_usd      = atr_usd_value * 1.0    # Lock breakeven at a full 1.0 ATR push
                damage_control_usd = atr_usd_value * 1.5    # Establish damage control at a 1.5 ATR peak
                damage_loss_usd    = -(atr_usd_value * 0.3) # Allow room to absorb pullbacks up to 0.3 ATR below entry
            else:
                # Safe Fallback to standard internal Profile/Equity normalization if targets fail to parse
                activate_usd       = self._get_dynamic_activation_usd(equity_base, pos)
                breakeven_usd      = equity_base * self.BREAKEVEN_NORM_PCT
                damage_control_usd = equity_base * self.DAMAGE_CONTROL_NORM_PCT
                damage_loss_usd    = -(equity_base * self.DAMAGE_LOSS_NORM_PCT)

            if current_peak >= breakeven_usd and ticket not in self._breakeven_set:
                self._set_breakeven_atomic(pos, breakeven_usd)

            if current_peak < activate_usd:
                exp_verdict = self._exp_guard.evaluate(ticket, profit)
                if exp_verdict == "close":
                    stats = self._exp_guard.stats
                    threshold = stats.loss_close_threshold if stats else 0.0
                    self._close_atomic(
                        pos,
                        reason=(
                            f"loss ${abs(profit):.2f} exceeded expectancy close "
                            f"threshold ${threshold:.2f} "
                            f"({ExpectancyGuard.__module__}.LOSS_MULT_CLOSE × avg loss)"
                        )
                    )
                    return
                elif exp_verdict == "alert":
                    stats = self._exp_guard.stats
                    threshold = stats.loss_alert_threshold if stats else 0.0
                    self.notify(
                        f"⚠️ Expectancy alert — {symbol} is down ${abs(profit):.2f} "
                        f"(avg loss is ${stats.avg_loss:.2f}, "
                        f"alert threshold ${threshold:.2f}). "
                        f"Consider closing manually.",
                        priority="normal",
                    )
                return

            if current_peak >= damage_control_usd and profit <= damage_loss_usd:
                self._close_atomic(pos, reason=(
                    f"peaked at +${current_peak:.2f} ({current_peak_pips:.1f} pips), "
                    f"reverted to ${profit:.2f} — dynamic damage-control termination"
                ))
                return

            if profit <= 0:
                exp_verdict = self._exp_guard.evaluate(ticket, profit)
                if exp_verdict == "close":
                    stats = self._exp_guard.stats
                    threshold = stats.loss_close_threshold if stats else 0.0
                    self._close_atomic(
                        pos,
                        reason=(
                            f"loss ${abs(profit):.2f} exceeded expectancy close "
                            f"threshold ${threshold:.2f} after peak "
                            f"+${current_peak:.2f}"
                        ),
                    )
                    return
                elif exp_verdict == "alert":
                    stats = self._exp_guard.stats
                    self.notify(
                        f"⚠️ Expectancy alert — {symbol} reversed from "
                        f"+${current_peak:.2f} to -${abs(profit):.2f}. "
                        f"Avg loss is ${stats.avg_loss:.2f}.",
                        priority="normal",
                    )
                return

            retracement = (current_peak - profit) / current_peak
            peak_eq_pct = current_peak / equity_base
            threshold   = self._threshold_for_normalized(peak_eq_pct)

            if retracement >= threshold:
                self._close_atomic(pos, reason=(
                    f"peaked at +${current_peak:.2f} ({current_peak_pips:.1f} pips), "
                    f"reverted to +${profit:.2f} ({retracement:.0%} retrace ≥ {threshold:.0%} normalized limit)"
                ))

    def _threshold_for_normalized(self, peak_equity_pct: float) -> float:
        for ceiling, threshold in self.TIERS:
            if peak_equity_pct < ceiling:
                return threshold
        return 0.35

    def _set_breakeven_atomic(self, pos, trigger_amt: float):
        ticket     = pos.ticket
        symbol     = pos.symbol
        entry      = float(pos.price_open)
        current_sl = float(pos.sl or 0.0)

        if ticket in self._be_attempted:
            return

        if pos.type == 0:
            if current_sl >= entry:
                self._breakeven_set.add(ticket)
                self._be_attempted.add(ticket)
                return
        else:
            if 0 < current_sl <= entry:
                self._breakeven_set.add(ticket)
                self._be_attempted.add(ticket)
                return

        new_sl = round(entry, 5)
        self._be_attempted.add(ticket)
        
        ok = self.broker.modify_position(ticket, symbol, new_sl)
        if ok:
            self._breakeven_set.add(ticket)
            self.notify(
                f"🔐 Dynamic ProfitGuard: {symbol} SL locked to entry @ {new_sl} "
                f"(triggered at structural threshold ${trigger_amt:.2f})",
                priority="normal",
            )

    def _close_atomic(self, pos, reason: str):
        ticket = pos.ticket
        if ticket in self._closed_this_cycle:
            return
        self._closed_this_cycle.add(ticket)

        self.notify(
            f"🔒 Dynamic ProfitGuard: Securing {pos.symbol} position — {reason}",
            priority="trade_executed",
        )
        self.broker.close_position(pos.symbol)
        
        self._peak.pop(ticket, None)
        self._peak_pips.pop(ticket, None)
        self._pip_val.pop(ticket, None)
        self._breakeven_set.discard(ticket)
        self._be_attempted.discard(ticket)
        self._exp_guard.clear_ticket(ticket)

    def status(self) -> str:
        with self._api_lock:
            if not self._peak:
                base = "No positions currently under active guard."
            else:
                eq_base = self._get_equity_base()
                lines = [f"Dynamic ProfitGuard status (Equity Base: ${eq_base:,.2f}):"]
                for ticket, peak in self._peak.items():
                    peak_pips = self._peak_pips.get(ticket, 0.0)
                    pip_val   = self._pip_val.get(ticket, 0.0)
                    peak_pct  = peak / eq_base
                    threshold = self._threshold_for_normalized(peak_pct)
                    be_locked = "✅ BE Locked" if ticket in self._breakeven_set else "⚠️ Pending BE"
                    lines.append(
                        f"  #{ticket} | Peak +${peak:.2f} ({peak_pips:.1f} pips, {peak_pct:.2%} Eq) | "
                        f"1 pip = ${pip_val:.4f} | Secure on {threshold:.0%} retrace | {be_locked}"
                    )
                base = "\n".join(lines)

        exp_summary = self._exp_guard.summary()
        return f"{base}\n\n{exp_summary}"

class TradeGatekeeper:
    """
    Pre-trade session and spread guard.

    Checks (in order):
    1. Friday-close blackout  (last 2 h of Friday, UTC)
    2. Asian session block    (configurable pairs + time window)
    3. Live spread ceiling
    """

    # UTC hour ranges (inclusive start, exclusive end)
    FRIDAY_CLOSE_START_UTC = 20   # 20:00 UTC Friday
    ASIAN_SESSION_START_UTC = 0   # 00:00 UTC
    ASIAN_SESSION_END_UTC   = 8   # 08:00 UTC

    def __init__(
        self,
        max_spread_pips:        float = 3.0,
        max_spread_pips_forex:  Optional[float] = None,
        max_spread_pips_crypto: float = 50.0,
        avoid_asian_session:    bool  = True,
        avoid_friday_close:     bool  = True,
    ):
        self.max_spread_forex   = max_spread_pips_forex if max_spread_pips_forex is not None else max_spread_pips
        self.max_spread_crypto  = max_spread_pips_crypto
        self.avoid_asian_session = avoid_asian_session
        self.avoid_friday_close  = avoid_friday_close

    def gate(self, symbol: str, broker) -> tuple[bool, str]:
        from manager.profile_manager import profile

        now_utc  = datetime.now(timezone.utc)
        weekday  = now_utc.weekday()   # 0=Mon … 4=Fri … 6=Sun
        hour_utc = now_utc.hour

        session_cfg = profile.sessions()

        # ── FIX #19: Check ACTUAL clock time, not just pair membership ──

        # 1. Friday close blackout
        if self.avoid_friday_close and weekday == 4 and hour_utc >= self.FRIDAY_CLOSE_START_UTC:
            return False, (
                f"Friday close blackout — no new entries after "
                f"{self.FRIDAY_CLOSE_START_UTC}:00 UTC on Friday."
            )

        # 2. Asian session block (time-gated, then pair-filtered)
        if self.avoid_asian_session:
            in_asian_hours = (
                self.ASIAN_SESSION_START_UTC <= hour_utc < self.ASIAN_SESSION_END_UTC
            )
            asian_pairs = getattr(session_cfg, "asian_session_pairs", [])
            if in_asian_hours and symbol in asian_pairs:
                return False, (
                    f"Asian session ({self.ASIAN_SESSION_START_UTC:02d}:00–"
                    f"{self.ASIAN_SESSION_END_UTC:02d}:00 UTC): "
                    f"trading {symbol} is blocked."
                )

        #3. Dynamic Spread Check via Profile Single Source of Truth
        spread = self._get_spread_pips(symbol, broker)
        ceiling = profile.risk(symbol).max_spread_pips
        
        if spread is not None and spread > ceiling:
            return False, f"Spread too high: {spread:.1f} pips (ceiling {ceiling:.1f})."

        return True, "OK"

    def _get_spread_pips(self, symbol: str, broker=None, category: Optional[str] = None) -> Optional[float]:
        import MetaTrader5 as mt5
        try:
            tick = None
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
            if info.digits in (5, 3):
                return spread_points / 10.0
            return spread_points
        except Exception:
            return None

class CorrelationGuard:
    def __init__(self, max_shared_legs: int = 2):
        self.max_shared_legs = max_shared_legs
 
    def check(self, symbol: str, broker) -> Tuple[bool, str]:
        positions = broker.getPositions() or []
        if not positions:
            return True, "No existing positions."
 
        open_legs: List[str] = []
        for pos in positions:
            open_legs.extend(self._legs(pos.symbol))
 
        leg_counts = Counter(open_legs)
        proposed_legs = self._legs(symbol)
        for leg in proposed_legs:
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
    def _legs(symbol: str) -> List[str]:
        if _profile.get_asset_class(symbol) != "forex":
            return []
            
        sym = symbol.upper()
        if len(sym) >= 6:
            return [sym[:3], sym[3:6]]
        return []