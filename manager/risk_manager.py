# risk_manager.py
from typing import Dict, Any, Optional
import MetaTrader5 as mt5
from datetime import datetime
import math

class RiskManager:
    """The Defense Engine: Handles exposure, drawdown limits, and position sizing."""
    
    def __init__(self, broker, max_open_trades: int = 3, min_margin_level: float = 150.0, notify_callback=print):
        self.broker = broker
        self.max_open_trades = max_open_trades
        self.min_margin_level = min_margin_level
        self.notify = notify_callback
        
        self.daily_high_watermark = 0.0 
        self.daily_low_watermark = 0.0
        self.watermark_date=None
    
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

    def is_trading_allowed(self, symbol: str, max_daily_loss: float, portfolio_size: int) -> tuple[bool, str]:
        """A detailed check verifying global exposure, symbol exposure, and profitability."""
        account = self.broker.getAccountInfo()
        if not account:
            return False, "Could not fetch account data from broker."

        positions = self.broker.getPositions()
        
        # 1. Global Exposure Limit (Max trades = Number of symbols in portfolio)
        current_open_trades = len(positions) if positions else 0
        if current_open_trades >= portfolio_size:
            return False, f"Global exposure reached ({current_open_trades}/{portfolio_size} trades)."

        # 2. Check Specific Symbol Limits
        if positions:
            symbol_positions = [p for p in positions if p.symbol == symbol]
            symbol_trade_count = len(symbol_positions)

            # Rule A: Max 3 trades per symbol
            if symbol_trade_count >= 3:
                return False, f"Max symbol exposure reached for {symbol} ({symbol_trade_count}/3 trades)."

            # Rule B: Are any of the current positions for this symbol losing money?
            # If so, do not enter again (No averaging down)
            for pos in symbol_positions:
                if pos.profit < 0:
                    return False, f"{symbol} currently has a losing position (${pos.profit}). No new entries allowed."

        # 3. Check Margin Level
        if account.margin_level and account.margin_level < self.min_margin_level:
            return False, f"Margin level too low ({account.margin_level:.1f}%). Minimum is {self.min_margin_level}%."

        # === 3. THE TRAILING EQUITY STOP ===
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
        """Calculates exact lot size and dynamically scales risk if taking losses."""
        
        # 1. First, check if we are even allowed to trade
        allowed, reason = self.is_trading_allowed(symbol, max_daily_loss, portfolio_size)
        if not allowed:
            return {"approved": False, "reason": reason}

        account = self.broker.getAccountInfo()
        current_equity = account.equity
        trailing_drawdown = self.daily_high_watermark - current_equity
        actual_risk_pct = base_risk_pct
        bounce_from_low = (current_equity - self.daily_low_watermark) / self.daily_low_watermark if self.daily_low_watermark > 0 else 0.0

        # 2. Dynamic Risk Scaling (If down 60% from the peak, cut risk in half to survive)
        if max_daily_loss > 0 and trailing_drawdown > (max_daily_loss * 0.6):
            actual_risk_pct = base_risk_pct / 2.0
            self.notify(f"[Risk Manager]: ⚠️ Drawdown from peak is ${trailing_drawdown:.2f}. Scaling risk down to {actual_risk_pct}%.")
        elif bounce_from_low > 0 and bounce_from_low < (max_daily_loss * 0.3):
            actual_risk_pct = base_risk_pct / 1.5
            self.notify(f"[Risk Manager]: 📈 Bouncing from daily low. Adjusting risk to {actual_risk_pct}%.")

        # 3. Calculate Risk in Dollars
        max_risk_usd = account.balance * (actual_risk_pct / 100)
        
        # 4. Calculate Position Size (Lots)
        safe_sl_pips = stop_loss_pips if stop_loss_pips > 0 else 20.0 
        
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return {"approved": False, "reason": f"Could not fetch symbol info for {symbol}"}

        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return {"approved": False, "reason": f"Could not fetch tick data for {symbol}"}

        point = symbol_info.point
        spread_points = int(round((tick.ask - tick.bid) / point))
        min_stop_level = symbol_info.trade_stops_level or 0

        # Determine pip multiplier based on symbol type
        if "BTC" in symbol or "ETH" in symbol:
            pip_multiplier = 1.0
        else:
            pip_multiplier = 10.0

        base_sl_points = int(safe_sl_pips * pip_multiplier)
        safe_distance_points = spread_points + min_stop_level
        safe_sl_points = max(base_sl_points, safe_distance_points)
        actual_sl_pips = safe_sl_points / pip_multiplier

        # Call the new function directly! It handles all the MT5 tick math and min/max limits.
        optimal_lots = self.calculate_position_size(symbol, max_risk_usd, safe_sl_points)
        
        # If the risk is too small to meet the broker's minimum lot size, abort the trade.
        if optimal_lots == 0.0:
             return {
                "approved": False,
                "reason": f"Spread is too large ({spread_points} points). Target risk (${max_risk_usd:.2f}) drops volume below minimum lot size.",
            }

        return {
            "approved": True,
            "reason": "Clearance granted.",
            "symbol": symbol,
            "lots": optimal_lots,
            "risk_usd": max_risk_usd,
            "applied_risk_pct": actual_risk_pct,
            "stop_loss_pips": actual_sl_pips
        }
    
    def calculate_micro_lot(self, symbol: Optional[str] = None) -> float:
        """Return the smallest tradable volume for the symbol, falling back to 0.01 if unknown."""
        if symbol:
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is not None and symbol_info.volume_min and symbol_info.volume_min > 0:
                return round(symbol_info.volume_min, 2)

        return 0.01  # Default micro-lot for MT5 if symbol-specific data is unavailable

    def calculate_position_size(self, symbol: str, risk_amount_usd: float, stop_loss_points: int) -> float:
        """
        Dynamically calculates the correct lot size based on the specific asset's tick value.
        
        :param symbol: The ticker (e.g., 'EURUSD', 'XAUUSD')
        :param risk_amount_usd: The maximum amount of money willing to lose on this trade.
        :param stop_loss_points: The stop loss distance in POINTS (not pips). 
                                Points = the smallest price movement (tick_size).
        """
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            # [Debug] Failed to get symbol info
            return 0.0

        # 1. Fetch dynamic tick data from the broker
        tick_value = symbol_info.trade_tick_value 
        tick_size = symbol_info.trade_tick_size

        if tick_value == 0 or tick_size == 0:
            # [Debug] Tick value/size is 0; cannot calculate risk
            return 0.0

        # 2. Calculate the monetary risk for 1 standard lot
        # If 1 lot moves by the stop loss distance, how much money do we lose?
        risk_per_1_lot = stop_loss_points * tick_value

        # 3. Calculate exact required lot size to match our target risk
        raw_lot_size = risk_amount_usd / risk_per_1_lot

        # 4. Normalize the lot size to broker rules (min, max, and step)
        min_lot = symbol_info.volume_min
        max_lot = symbol_info.volume_max
        step_lot = symbol_info.volume_step

        # Round down to the nearest allowed step (e.g., 0.01 micro lots)
        # Using round() with division and multiplication ensures precision
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