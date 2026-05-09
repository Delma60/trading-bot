"""
manager/profile_manager.py — Single source of truth for bot configuration.

Every module imports from here. No module reads profile.json directly.

Usage
-----
    from manager.profile_manager import profile

    # Global risk settings
    risk = profile.risk()

    # Symbol-aware (merges global defaults + any symbol override)
    risk = profile.risk("XAUUSD")

    # Other sections
    portfolio = profile.portfolio()
    broker    = profile.broker()
    sessions  = profile.sessions()
    scanner   = profile.scanner()

    # Convenience accessors
    symbols   = profile.symbols()
    strategy  = profile.strategy_for("XAUUSD")
"""

import json
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Typed config dataclasses ──────────────────────────────────────────────────

@dataclass
class RiskConfig:
    risk_pct:             float
    stop_loss_pips:       float
    take_profit_pips:     float
    max_daily_loss:       float
    daily_goal:           float
    cooldown_minutes:     int
    lock_amount:          float   # fixed $ to ring-fence
    lock_pct:             float   # fraction of balance to ring-fence (0.0–0.99)
    # Drawdown recovery thresholds
    moderate_threshold:   float = 0.50
    moderate_scale:       float = 0.50
    critical_threshold:   float = 0.80
    critical_scale:       float = 0.25

    @property
    def lock_pct_decimal(self) -> float:
        """Always a 0–1 fraction regardless of how it was stored."""
        return self.lock_pct if self.lock_pct <= 1.0 else self.lock_pct / 100.0


@dataclass
class PortfolioConfig:
    symbols:              list[str]
    preferred_timeframes: list[str]
    asset_classes:        dict[str, list[str]]
    strategy_mapping:     dict


@dataclass
class BrokerConfig:
    max_open_trades:       int
    min_margin_level:      float
    spread_tolerance_pips: float
    magic_number:          int
    slippage_points:       int


@dataclass
class SessionConfig:
    avoid_asian_session:  bool
    avoid_friday_close:   bool
    asian_session_pairs:  list[str]


@dataclass
class ScannerConfig:
    interval_seconds:      int
    dry_run:               bool
    mtf_min_alignment:     float
    volatility_spike_atr:  float
    dead_volume_ratio:     float


# ── Validation ────────────────────────────────────────────────────────────────

_REQUIRED_SECTIONS = ["portfolio", "risk", "broker", "sessions", "scanner"]

_REQUIRED_RISK_KEYS = [
    "risk_pct", "stop_loss_pips", "take_profit_pips",
    "max_daily_loss", "daily_goal", "cooldown_minutes",
    "lock_amount", "lock_pct",
]

_REQUIRED_BROKER_KEYS = [
    "max_open_trades", "min_margin_level",
    "spread_tolerance_pips", "magic_number", "slippage_points",
]


def _validate(raw: dict, path: Path) -> None:
    """Raise ValueError with a clear message if the config is malformed."""
    for section in _REQUIRED_SECTIONS:
        if section not in raw:
            raise ValueError(
                f"[ProfileManager] Missing section '{section}' in {path}. "
                f"Add it or re-run first-time setup."
            )

    defaults = raw["risk"].get("defaults", {})
    for key in _REQUIRED_RISK_KEYS:
        if key not in defaults:
            raise ValueError(
                f"[ProfileManager] Missing risk.defaults.{key} in {path}."
            )

    for key in _REQUIRED_BROKER_KEYS:
        if key not in raw["broker"]:
            raise ValueError(
                f"[ProfileManager] Missing broker.{key} in {path}."
            )

    symbols = raw["portfolio"].get("symbols", [])
    if not symbols:
        raise ValueError(
            f"[ProfileManager] portfolio.symbols is empty in {path}. "
            f"Add at least one symbol."
        )

    r = raw["risk"]["defaults"]
    if not 0.0 < r["risk_pct"] <= 10.0:
        raise ValueError(
            f"[ProfileManager] risk.defaults.risk_pct must be 0–10 "
            f"(got {r['risk_pct']})."
        )
    if r["max_daily_loss"] <= 0:
        raise ValueError(
            f"[ProfileManager] risk.defaults.max_daily_loss must be > 0."
        )


# ── ProfileManager ────────────────────────────────────────────────────────────

class ProfileManager:
    """
    Thread-safe config loader. Call profile.reload() to hot-reload
    from disk without restarting the bot.
    """

    PROFILE_PATH = Path("data/profile.json")

    def __init__(self, path: Path = None):
        self._path  = path or self.PROFILE_PATH
        self._lock  = threading.RLock()
        self._raw:  dict = {}
        self._load_and_validate()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load_and_validate(self) -> None:
        if not self._path.exists():
            raise FileNotFoundError(
                f"[ProfileManager] {self._path} not found. "
                f"Run first-time setup or copy the template."
            )
        with self._lock:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            _validate(raw, self._path)
            self._raw = raw

    def reload(self) -> None:
        """Hot-reload config from disk. Safe to call from any thread."""
        self._load_and_validate()

    def save(self) -> None:
        """Persist the current in-memory config back to disk."""
        with self._lock:
            self._path.write_text(
                json.dumps(self._raw, indent=4, ensure_ascii=False),
                encoding="utf-8",
            )

    # ── Section accessors ─────────────────────────────────────────────────────

    def risk(self, symbol: str = None) -> RiskConfig:
        """
        Returns risk config for symbol, merging global defaults with
        any per-symbol override. If symbol is None, returns global defaults.
        """
        with self._lock:
            base  = deepcopy(self._raw["risk"]["defaults"])
            overrides = self._raw["risk"].get("symbol_overrides", {})
            if symbol and symbol.upper() in overrides:
                base.update(overrides[symbol.upper()])

            dr = self._raw["risk"].get("drawdown_recovery", {})
            return RiskConfig(
                risk_pct             = float(base["risk_pct"]),
                stop_loss_pips       = float(base["stop_loss_pips"]),
                take_profit_pips     = float(base["take_profit_pips"]),
                max_daily_loss       = float(base["max_daily_loss"]),
                daily_goal           = float(base["daily_goal"]),
                cooldown_minutes     = int(base["cooldown_minutes"]),
                lock_amount          = float(base["lock_amount"]),
                lock_pct             = float(base["lock_pct"]),
                moderate_threshold   = float(dr.get("moderate_threshold", 0.50)),
                moderate_scale       = float(dr.get("moderate_scale",     0.50)),
                critical_threshold   = float(dr.get("critical_threshold", 0.80)),
                critical_scale       = float(dr.get("critical_scale",     0.25)),
            )

    def portfolio(self) -> PortfolioConfig:
        with self._lock:
            p = self._raw["portfolio"]
            return PortfolioConfig(
                symbols              = list(p.get("symbols", [])),
                preferred_timeframes = list(p.get("preferred_timeframes", ["H1"])),
                asset_classes        = dict(p.get("asset_classes", {})),
                strategy_mapping     = deepcopy(p.get("strategy_mapping", {})),
            )

    def broker(self) -> BrokerConfig:
        with self._lock:
            b = self._raw["broker"]
            return BrokerConfig(
                max_open_trades       = int(b["max_open_trades"]),
                min_margin_level      = float(b["min_margin_level"]),
                spread_tolerance_pips = float(b["spread_tolerance_pips"]),
                magic_number          = int(b["magic_number"]),
                slippage_points       = int(b["slippage_points"]),
            )

    def sessions(self) -> SessionConfig:
        with self._lock:
            s = self._raw["sessions"]
            return SessionConfig(
                avoid_asian_session = bool(s.get("avoid_asian_session", True)),
                avoid_friday_close  = bool(s.get("avoid_friday_close",  True)),
                asian_session_pairs = list(s.get("asian_session_pairs", [])),
            )

    def scanner(self) -> ScannerConfig:
        with self._lock:
            sc = self._raw["scanner"]
            return ScannerConfig(
                interval_seconds     = int(sc.get("interval_seconds",     3)),
                dry_run              = bool(sc.get("dry_run",             False)),
                mtf_min_alignment    = float(sc.get("mtf_min_alignment",  0.50)),
                volatility_spike_atr = float(sc.get("volatility_spike_atr", 2.5)),
                dead_volume_ratio    = float(sc.get("dead_volume_ratio",  0.3)),
            )

    # ── Convenience helpers ───────────────────────────────────────────────────

    def symbols(self) -> list[str]:
        with self._lock:
            return list(self._raw["portfolio"].get("symbols", []))

    def strategy_for(self, symbol: str) -> str:
        """Returns the best strategy name for a given symbol."""
        with self._lock:
            mapping = self._raw["portfolio"].get("strategy_mapping", {})
            overrides = mapping.get("symbol_overrides", {})
            if symbol.upper() in overrides:
                return overrides[symbol.upper()]

            asset_classes = self._raw["portfolio"].get("asset_classes", {})
            for asset_class, syms in asset_classes.items():
                if symbol.upper() in [s.upper() for s in syms]:
                    class_defaults = mapping.get("asset_class_defaults", {})
                    if asset_class in class_defaults:
                        return class_defaults[asset_class]

            return mapping.get("default", "Mean_Reversion")

    # ── Runtime updates (chat commands like "change risk to 2") ──────────────

    def update_risk(self, symbol: str = None, **kwargs) -> None:
        """
        Update one or more risk fields at runtime and persist.

        profile.update_risk(risk_pct=2.0)                     # global
        profile.update_risk("XAUUSD", stop_loss_pips=100.0)   # per-symbol
        """
        with self._lock:
            if symbol:
                sym = symbol.upper()
                overrides = self._raw["risk"].setdefault("symbol_overrides", {})
                overrides.setdefault(sym, {}).update(kwargs)
            else:
                self._raw["risk"]["defaults"].update(kwargs)
            self.save()

    def update_broker(self, **kwargs) -> None:
        with self._lock:
            self._raw["broker"].update(kwargs)
            self.save()

    def add_symbol(self, symbol: str, asset_class: str = None) -> bool:
        """Add a symbol to the portfolio. Returns False if already present."""
        with self._lock:
            sym = symbol.upper()
            syms = self._raw["portfolio"]["symbols"]
            if sym in syms:
                return False
            syms.append(sym)
            if asset_class:
                classes = self._raw["portfolio"].setdefault("asset_classes", {})
                classes.setdefault(asset_class, []).append(sym)
            self.save()
            return True

    def remove_symbol(self, symbol: str) -> bool:
        """Remove a symbol from the portfolio and all asset class lists."""
        with self._lock:
            sym = symbol.upper()
            syms = self._raw["portfolio"]["symbols"]
            if sym not in syms:
                return False
            syms.remove(sym)
            for class_list in self._raw["portfolio"].get("asset_classes", {}).values():
                if sym in class_list:
                    class_list.remove(sym)
            self.save()
            return True

    # ── Credential separation (kept out of profile.json) ─────────────────────

    def load_credentials(self) -> dict:
        """
        Loads MT5 credentials. Checks environment variables first,
        then falls back to data/credentials.json.
        """
        import os
        if os.getenv("MT5_LOGIN"):
            return {
                "login":    int(os.getenv("MT5_LOGIN")),
                "password": os.getenv("MT5_PASSWORD", ""),
                "server":   os.getenv("MT5_SERVER", "MetaQuotes-Demo"),
            }
        cred_path = self._path.parent / "credentials.json"
        if cred_path.exists():
            return json.loads(cred_path.read_text(encoding="utf-8"))
        return {}

    def save_credentials(self, login: int, password: str, server: str) -> None:
        cred_path = self._path.parent / "credentials.json"
        cred_path.write_text(
            json.dumps({"login": login, "password": password, "server": server},
                       indent=4),
            encoding="utf-8",
        )


# ── Module-level singleton ────────────────────────────────────────────────────
# Import this everywhere: `from manager.profile_manager import profile`
profile = ProfileManager()
