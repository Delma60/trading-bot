import json
from pathlib import Path
from typing import Any

class ProfileManager:
    """Handles all File I/O, saving, and loading of user configuration."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.profile_file = self.data_dir / "profile.json"
        self.history_file = self.data_dir / "history.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.config = {}
        self.load_config()

    def _read_json(self, filepath: Path, default: Any = None) -> Any:
        if not filepath.exists(): return default if default is not None else {}
        try:
            with filepath.open("r") as f: return json.load(f)
        except (json.JSONDecodeError, IOError): return default if default is not None else {}

    def _write_json(self, filepath: Path, data: Any):
        with filepath.open("w") as f: json.dump(data, f, indent=4)

    def load_config(self) -> bool:
        self.config = self._read_json(self.profile_file)
        return "trading_symbols" in self.config

    def save_config(self, trading_symbols, risk, target, max_loss, timeframes):
        self.config.update({
            "trading_symbols": trading_symbols,
            "risk_percentage": risk,
            "target_profit": target,
            "max_daily_loss": max_loss,
            "preferred_timeframes": timeframes
        })
        self._write_json(self.profile_file, self.config)

    def save_credentials(self, login, password, server):
        self.config.update({"login": login, "password": password, "server": server})
        self._write_json(self.profile_file, self.config)

    def _format_account_data(self):
        """Fetches and formats live MT5 account data."""
        if not self.broker.connected:
            return "⚠️ Cannot fetch account data. MT5 is not connected."
            
        account = self.broker.getAccountInfo()
        if not account:
            return "⚠️ Failed to retrieve account information from broker."
        
        high_mark = getattr(self.risk_manager, 'daily_high_watermark', account.equity)
        low_mark = getattr(self.risk_manager, 'daily_low_watermark', account.equity)
        
        # If low watermark is infinity (no ticks yet), format it cleanly
        if low_mark == float('inf'):
            low_mark = account.equity
            
        return {
            "Balance": f"${account.balance:,.2f}",
            "Equity": f"${account.equity:,.2f}",
            "Floating PnL": f"${account.profit:,.2f}",
            "Margin Level": f"{account.margin_level:.2f}%" if account.margin_level else "N/A",
            "Daily High Watermark": f"${high_mark:,.2f}",
            "Daily Low Watermark": f"${low_mark:,.2f}",
            "Daily Drawdown": f"${(high_mark - low_mark):,.2f}"
            
        }

    def _format_positions_data(self):
        """Fetches and formats open MT5 trades."""
        if not self.broker.connected:
            return "⚠️ Cannot fetch positions. MT5 is not connected."
            
        positions = self.broker.getPositions()
        if not positions:
            return "You currently have no active trades."
            
        pos_strings = []
        for p in positions:
            # MT5 position types: 0 = BUY, 1 = SELL
            action = "BUY" if p.type == 0 else "SELL"
            pos_strings.append(f"• {p.symbol}: {action} {p.volume} lots | Open: {p.price_open} | Profit: ${p.profit:,.2f}")
            
        return "\n".join(pos_strings)

    def _format_settings_data(self):
        """Displays the current risk and portfolio configuration."""
        return {
            "Watchlist": ", ".join(self.trading_symbols) if self.trading_symbols else "Empty",
            "Risk Per Trade": f"{self.risk_percentage}%",
            "Target Profit": f"${self.target_profit}",
            "Max Daily Loss Limit": f"${self.max_daily_loss}"
        }

    def log_interaction(self, user_input: str, intent: str, status: str = "completed"):
        
        from datetime import datetime
        entry = {
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "intent": intent,
            "status": status
        }
        history = self._read_json(self.history_file, [])
        history.append(entry)
        self._write_json(self.history_file, history[-200:])