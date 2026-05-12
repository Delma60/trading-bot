import threading
import time, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from strategies.features.feature_engineer import FeatureEngineer
from manager.profile_manager import profile as _profile

class ProgressSpinner:
    """Reusable progress indicator with spinner animation for terminal output."""
    
    def __init__(self, total: int, prefix: str = "Loading", suffix: str = ""):
        self.total = total
        self.prefix = prefix
        self.suffix = suffix
        self._lock = threading.Lock()
        self._count = 0
        self._spinner_chars = ['|', '/', '-', '\\']
        self._spinner_index = 0
    
    def update(self, item_name: str = "") -> None:
        """Update progress with next item. Thread-safe."""
        with self._lock:
            self._count += 1
            spinner = self._spinner_chars[self._spinner_index % len(self._spinner_chars)]
            self._spinner_index += 1
            
            msg = f"[{self.prefix}] {spinner} {self._count}/{self.total}"
            if item_name:
                msg += f" {item_name}"
            if self.suffix:
                msg += f" {self.suffix}"
            
            print(f"\r{msg}", end="", flush=True)
    
    def finish(self, completion_msg: str = "Complete.") -> None:
        """Clear progress line and show completion message."""
        print("\r" + " " * 60 + f"\r{completion_msg}", flush=True)


class LocalCache:
    """In-memory cache for account, positions, symbol info, OHLCV, and engineered features."""
    def _save_symbol_features(self, symbol: str, df: pd.DataFrame) -> None:
        try:
            path = self.history_dir / f"{symbol.upper()}_{self._timeframe}_features.parquet"
            df.to_parquet(path)
        except Exception as exc:
            pass
            # self.notify(f"[LocalCache] Failed to save features for {symbol}: {exc}")

    def _save_positions(self):
        try:
            import json
            path = self.history_dir / "positions.json"
            with path.open("w", encoding="utf-8") as f:
                # Only save basic info to avoid serialization errors
                json.dump([
                    {"symbol": p.symbol, "type": getattr(p, "type", None), "volume": getattr(p, "volume", None), "price_open": getattr(p, "price_open", None), "profit": getattr(p, "profit", None)}
                    for p in self._positions
                ], f, indent=2)
        except Exception as exc:
            pass
            # self.notify(f"[LocalCache] Failed to save positions: {exc}")

    def _save_account(self):
        try:
            import json
            path = self.history_dir / "account.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump(self._account, f, indent=2, default=str)
        except Exception as exc:
            pass
            # self.notify(f"[LocalCache] Failed to save account: {exc}")

    def __init__(
        self,
        broker,
        symbols: Optional[List[str]] = None,
        notify_callback=print,
        history_dir: Optional[Path] = None,
    ):
        self.broker = broker
        self.symbols = [s.upper() for s in (symbols or [])]
        self.notify = notify_callback
        self.history_dir = Path(history_dir or Path("data/cache"))
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self._timeframe = _profile.scanner().timeframe

        self._lock = threading.RLock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_snapshot_refresh = 0.0
        self._last_feature_refresh = 0.0

        self._account: Optional[Any] = None
        self._positions: List[Any] = []
        self._symbol_info: Dict[str, dict] = {}
        self._ticks: Dict[str, dict] = {}
        self._ohlcv: Dict[str, pd.DataFrame] = {}
        self._features: Dict[str, pd.DataFrame] = {}

    @property
    def timeframe(self) -> str:
        """The primary timeframe this cache is populated with."""
        return self._timeframe
    
    def warm_up(self):
        """Warm the cache with historical OHLCV, symbol info, and market snapshots."""
        self.notify("[LocalCache] Warming up cache...")
        self._refresh_symbol_info()
        self._refresh_snapshot()
        self._refresh_features(is_initial=True)
        self.notify("[LocalCache] Cache warm-up complete.")

    def start(self):
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._background_loop, daemon=True, name="LocalCache")
        self._thread.start()
        self.notify("[LocalCache] Background cache refresh started.")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self.notify("[LocalCache] Background cache refresh stopped.")

    def _background_loop(self):
        while self._running:
            now = time.time()
            if now - self._last_snapshot_refresh >= 2.0:
                self._refresh_snapshot()
            if now - self._last_feature_refresh >= 60.0:
                self._refresh_features(is_initial=False)
            time.sleep(0.5)

    def _refresh_snapshot(self):
        if not self.broker.connected:
            return

        with self._lock:
            account = self.broker.getAccountInfo()
            if account is not None:
                self._account = account

            positions = self.broker.getPositions() or []
            self._positions = list(positions)

            for symbol in self.symbols:
                if symbol not in self._symbol_info:
                    symbol_info = self.broker.get_symbol_info(symbol)
                    if symbol_info is not None:
                        self._symbol_info[symbol] = symbol_info

                tick_data = self.broker.get_tick_data(symbol)
                if tick_data is not None:
                    self._ticks[symbol] = tick_data

            self._last_snapshot_refresh = time.time()

    def _refresh_symbol_info(self):
        if not self.broker.connected:
            return

        with self._lock:
            for symbol in self.symbols:
                symbol_info = self.broker.get_symbol_info(symbol)
                if symbol_info is not None:
                    self._symbol_info[symbol] = symbol_info

                tick_data = self.broker.get_tick_data(symbol)
                if tick_data is not None:
                    self._ticks[symbol] = tick_data


    def _get_history_path(self, symbol: str) -> Path:
        # 1. Strip out any dangerous characters (allow only uppercase A-Z and 0-9)
        safe_symbol = re.sub(r'[^A-Z0-9]', '', symbol.upper())
        if not safe_symbol:
            raise ValueError("Invalid symbol provided for caching.")
        
        filename = f"{safe_symbol}_{self._timeframe}_ohlcv.parquet"
        target_path = self.history_dir / filename
        
        # 2. Enforce absolute path boundary resolution
        resolved_base = self.history_dir.resolve()
        resolved_target = target_path.resolve()
        
        if not resolved_target.is_relative_to(resolved_base):
            raise PermissionError("Path traversal attempt detected.")
            
        return target_path
    
    def _load_symbol_history(self, symbol: str) -> Optional[pd.DataFrame]:
        path = self._get_history_path(symbol)
        if not path.exists():
            return None

        try:
            df = pd.read_parquet(path)
            if not df.empty:
                return df
        except Exception as exc:
            self.notify(f"[LocalCache] Failed to load cached history for {symbol}: {exc}")
        return None

    def _save_symbol_history(self, symbol: str, df: pd.DataFrame) -> None:
        try:
            df.to_parquet(self._get_history_path(symbol))
        except Exception as exc:
            pass
            # self.notify(f"[LocalCache] Failed to save history for {symbol}: {exc}")

    def _refresh_features(self, is_initial: bool = False):
        if not self.broker.connected:
            return

        # Initialize progress spinner only for initial load
        progress = ProgressSpinner(total=len(self.symbols), prefix="[LocalCache]") if is_initial else None

        with ThreadPoolExecutor(max_workers=4) as pool:
            pool.map(self._load_symbol_features, [(symbol, progress) for symbol in self.symbols])

        # Save all features after refresh
        for symbol, feat_df in self._features.items():
            if feat_df is not None:
                self._save_symbol_features(symbol, feat_df)

        if is_initial and progress:
            progress.finish("[LocalCache] Cache loading complete.")
        self._last_feature_refresh = time.time()

        # Save positions and account snapshot
        self._save_positions()
        self._save_account()

    def _load_symbol_features(self, args):
        symbol, progress = args
        symbol = symbol.upper()

        # Update progress with spinner (only if progress object exists)
        if progress:
            progress.update(item_name=symbol)

        df = self._load_symbol_history(symbol)
        if df is None:
            df = self.broker.ohclv_data(symbol, timeframe=self._timeframe, num_bars=1000)
            if df is None or df.empty:
                return
            with self._lock:
                self._ohlcv[symbol] = df
            self._save_symbol_history(symbol, df)
        else:
            with self._lock:
                self._ohlcv[symbol] = df

        # Save OHLCV history after update
        if df is not None:
            self._save_symbol_history(symbol, df)

        if symbol not in self._features or self._features[symbol] is None:
            feat_df = FeatureEngineer.compute(df)
            if feat_df is not None and not feat_df.empty:
                with self._lock:
                    self._features[symbol] = feat_df
                self._save_symbol_features(symbol, feat_df)

    def get_raw_ohlcv(self, symbol: str, timeframe: str = None) -> Optional[pd.DataFrame]:
        if timeframe is None:
            timeframe = self._timeframe
        with self._lock:
            return self._ohlcv.get(symbol.upper())

    def get_features(self, symbol: str) -> Optional[pd.DataFrame]:
        with self._lock:
            return self._features.get(symbol.upper())

    def get_account(self) -> Optional[Any]:
        with self._lock:
            return self._account

    def get_positions(self) -> List[Any]:
        with self._lock:
            return list(self._positions)

    def get_symbol_info(self, symbol: str) -> Optional[dict]:
        with self._lock:
            info = self._symbol_info.get(symbol.upper())
        if info is not None:
            return info

        if self.broker.connected:
            info = self.broker.get_symbol_info(symbol)
            if info is not None:
                with self._lock:
                    self._symbol_info[symbol.upper()] = info
            return info
        return None

    def get_tick_data(self, symbol: str) -> Optional[dict]:
        with self._lock:
            return self._ticks.get(symbol.upper())

    def get_pip_value(self, symbol: str, lots: float) -> float:
        info = self.get_symbol_info(symbol)
        if info is None:
            return 0.10

        try:
            tick_value = float(info.get("trade_tick_value", 0.0))
            tick_size = float(info.get("trade_tick_size", 0.0))
            point = float(info.get("point", 0.0))
            digits = int(info.get("digits", 5))

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

    def get_positions_snapshot(self) -> List[Any]:
        return self.get_positions()
