import json
from pathlib import Path
from typing import Any

class ProfileManager:
    """Handles all file-based configuration and credentials storage."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.config_file = self.data_dir / "profile.json"
        self.credentials_file = self.data_dir / "credentials.json"
        self.history_file = self.data_dir / "history.json"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.config = self._read_json(self.config_file, {})

    def _read_json(self, filepath: Path, default: Any = None) -> Any:
        if not filepath.exists():
            return default if default is not None else {}
        try:
            with filepath.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return default if default is not None else {}

    def _write_json(self, filepath: Path, data: Any):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with filepath.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load_config(self) -> bool:
        self.config = self._read_json(self.config_file, {})
        return bool(self.config)

    def get_config(self) -> dict:
        return self.config.copy()

    def save_config(self, trading_symbols, risk, target, max_loss, timeframes):
        self.config.update({
            "trading_symbols": trading_symbols,
            "risk_percentage": risk,
            "target_profit": target,
            "max_daily_loss": max_loss,
            "preferred_timeframes": timeframes,
        })
        self._write_json(self.config_file, self.config)

    def load_credentials(self) -> dict:
        return self._read_json(self.credentials_file, {})

    def save_credentials(self, login, password, server):
        credentials = {"login": login, "password": password, "server": server}
        self._write_json(self.credentials_file, credentials)
        return credentials

    def log_interaction(self, user_input: str, intent: str, status: str = "completed"):
        from datetime import datetime

        entry = {
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
            "intent": intent,
            "status": status,
        }
        history = self._read_json(self.history_file, [])
        history.append(entry)
        self._write_json(self.history_file, history[-200:])
