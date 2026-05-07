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