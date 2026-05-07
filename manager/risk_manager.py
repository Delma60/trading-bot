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
            self.watermark_date = today
        elif current_equity > self.daily_high_watermark:
            # Update the high watermark as the portfolio grows!
            self.daily_high_watermark = current_equity
            
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

        # 2. Dynamic Risk Scaling (If down 60% from the peak, cut risk in half to survive)
        if max_daily_loss > 0 and trailing_drawdown > (max_daily_loss * 0.6):
            actual_risk_pct = base_risk_pct / 2.0
            self.notify(f"[Risk Manager]: ⚠️ Drawdown from peak is ${trailing_drawdown:.2f}. Scaling risk down to {actual_risk_pct}%.")
            
        # 3. Calculate Risk in Dollars
        max_risk_usd = account.balance * (actual_risk_pct / 100)
        
        # 4. Calculate Position Size (Lots)
        estimated_pip_value_per_lot = 10.0 
        safe_sl_pips = stop_loss_pips if stop_loss_pips > 0 else 20.0 
        
        optimal_lots = max_risk_usd / (safe_sl_pips * estimated_pip_value_per_lot)
        optimal_lots = max(0.01, round(optimal_lots, 2)) # Enforce MT5 minimums

        return {
            "approved": True,
            "reason": "Clearance granted.",
            "symbol": symbol,
            "lots": optimal_lots,
            "risk_usd": max_risk_usd,
            "applied_risk_pct": actual_risk_pct,
            "stop_loss_pips": safe_sl_pips
        }
        
        """Calculates exact lot size and dynamically scales risk if taking losses."""
        
        # 1. First, check if we are even allowed to trade
        allowed, reason = self.is_trading_allowed(max_daily_loss)
        if not allowed:
            return {"approved": False, "reason": reason}

        account = self.broker.getAccountInfo()
        floating_loss = account.balance - account.equity
        realized_loss = self._get_realized_daily_loss()
        current_daily_loss = floating_loss + realized_loss
        actual_risk_pct = base_risk_pct

        # 2. Dynamic Risk Scaling (If down 60% of daily limit, cut risk in half)
        if max_daily_loss > 0 and current_daily_loss > (max_daily_loss * 0.6):
            actual_risk_pct = base_risk_pct / 2.0
            self.notify(f"[Risk Manager]: ⚠️ Approaching daily loss limit. Risk scaled down to {actual_risk_pct}%.")
        # 3. Calculate Risk in Dollars
        max_risk_usd = account.balance * (actual_risk_pct / 100)
        
        # 4. Calculate Position Size (Lots)
        # Using a standard fallback: $10 per pip for 1 standard lot
        estimated_pip_value_per_lot = 10.0 
        safe_sl_pips = stop_loss_pips if stop_loss_pips > 0 else 20.0 
        
        optimal_lots = max_risk_usd / (safe_sl_pips * estimated_pip_value_per_lot)
        optimal_lots = max(0.01, round(optimal_lots, 2)) # Enforce MT5 minimums

        return {
            "approved": True,
            "reason": "Clearance granted.",
            "symbol": symbol,
            "lots": optimal_lots,
            "risk_usd": max_risk_usd,
            "applied_risk_pct": actual_risk_pct,
            "stop_loss_pips": safe_sl_pips
        }