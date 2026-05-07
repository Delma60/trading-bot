# risk_manager.py
from typing import Dict, Any
import MetaTrader5 as mt5
from datetime import datetime

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

    def is_trading_allowed(self, max_daily_loss: float) -> tuple[bool, str]:
        """A hard check to see if the system is allowed to take ANY trades right now."""
        account = self.broker.getAccountInfo()
        if not account:
            return False, "Could not fetch account data from broker."

        # 1. Check Max Open Trades
        positions = self.broker.getPositions()
        current_open_trades = len(positions) if positions else 0
        if current_open_trades >= self.max_open_trades:
            return False, f"Max exposure reached ({current_open_trades}/{self.max_open_trades} trades)."

        # 2. Check Margin Level
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
    
    def calculate_safe_trade(self, symbol: str, base_risk_pct: float, stop_loss_pips: float, max_daily_loss: float) -> Dict[str, Any]:
        """Calculates exact lot size and dynamically scales risk if taking losses."""
        
        # 1. First, check if we are even allowed to trade
        allowed, reason = self.is_trading_allowed(max_daily_loss)
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
        
        # Convert Pips to Points (1 Standard Forex Pip = 10 MT5 Points)
        safe_sl_points = int(safe_sl_pips * 10)
        
        # Call the new function directly! It handles all the MT5 tick math and min/max limits.
        optimal_lots = self.calculate_position_size(symbol, max_risk_usd, safe_sl_points)
        
        # If the risk is too small to meet the broker's minimum lot size, abort the trade.
        if optimal_lots == 0.0:
             return {
                "approved": False,
                "reason": f"Target risk (${max_risk_usd:.2f}) is too small to meet the minimum lot size for {symbol}.",
            }

        return {
            "approved": True,
            "reason": "Clearance granted.",
            "symbol": symbol,
            "lots": optimal_lots,
            "risk_usd": max_risk_usd,
            "applied_risk_pct": actual_risk_pct,
            "stop_loss_pips": safe_sl_pips
        }
    
    def calculate_micro_lot(self) -> float:
        return 0.01 # MT5 standard micro-lot

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
            print(f"[Risk Manager] Failed to get symbol info for {symbol}")
            return 0.0

        # 1. Fetch dynamic tick data from the broker
        tick_value = symbol_info.trade_tick_value 
        tick_size = symbol_info.trade_tick_size

        if tick_value == 0 or tick_size == 0:
            print(f"[Risk Manager] Tick value/size is 0 for {symbol}. Cannot calculate risk.")
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
        clean_lot_size = (int(raw_lot_size / step_lot)) * step_lot

        # 5. Clamp limits to prevent broker rejection errors
        if clean_lot_size < min_lot:
            print(f"[Risk Manager] Risk too small. Calculated {clean_lot_size}, but broker min is {min_lot}")
            # Return 0.0 to abort the trade, or return min_lot if you accept the higher risk
            return 0.0 
            
        if clean_lot_size > max_lot:
            print(f"[Risk Manager] Warning: Clamping lot size to broker max ({max_lot})")
            return max_lot

        # Return rounded to 2 decimal places to avoid MT5 floating point rejection errors
        return round(clean_lot_size, 2)