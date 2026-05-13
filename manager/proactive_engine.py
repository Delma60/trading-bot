
"""
manager/proactive_engine.py — ARIA's Proactive Insights

Generates unprompted insights and observations.
Makes the bot feel alive rather than reactive.
"""

import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Callable

from manager.working_memory import WorkingMemory


class ProactiveEngine:
    """
    Generates unprompted insights and observations.
    
    This is what makes the bot feel alive rather than reactive.
    It watches for:
    - Market conditions changing while user is idle
    - Approaching risk limits before they're hit
    - Patterns in open positions
    - Time-based observations (approaching session close, etc.)
    - Things the user said they cared about
    """

    CHECK_INTERVAL = 30  # seconds

    def __init__(self, broker, working_memory:WorkingMemory, episodic_memory, 
                 notify_callback: Callable):
        self.broker = broker
        self.wm = working_memory
        self.em = episodic_memory
        self.notify = notify_callback
        self._running = False
        self._thread = None
        self._last_proactive: Optional[datetime] = None
        self._min_gap = timedelta(minutes=3)  # don't spam
        self._notified_positions: dict[int, set[str]] = {}

    def start(self):
        """Start the proactive watch loop."""
        self._running = True
        self._thread = threading.Thread(
            target=self._watch_loop, daemon=True, name="ProactiveEngine"
        )
        self._thread.start()

    def stop(self):
        """Stop the proactive watch loop."""
        self._running = False

    def _watch_loop(self):
        """Main loop that generates proactive thoughts."""
        while self._running:
            try:
                if self.broker.connected:
                    thought = self._generate_proactive_thought()
                    if thought:
                        self.notify(thought, priority="proactive")
            except Exception:
                pass
            time.sleep(self.CHECK_INTERVAL)

    def _generate_proactive_thought(self) -> Optional[str]:
        """
        Returns a proactive message if something worth saying is detected.
        Returns None if nothing notable.
        """
        # Rate limit
        if self._last_proactive:
            if datetime.now() - self._last_proactive < self._min_gap:
                return None

        # Check open positions for things worth noting
        thought = self._check_positions()
        if thought:
            self._last_proactive = datetime.now()
            return thought

        # Check time-of-day events
        thought = self._check_session_timing()
        if thought:
            self._last_proactive = datetime.now()
            return thought

        # Check if a promised alert should fire
        thought = self._check_promises()
        if thought:
            self._last_proactive = datetime.now()
            return thought

        return None

    # ── Excerpt drop-in for manager/proactive_engine.py ───────────────────────────
    
    def _check_positions(self):
        """Scans floating positions to provide structural, non-intrusive runtime feedback."""
        positions = self.broker.getPositions()
        if not positions:
            return None

        for pos in positions:
            symbol = pos.symbol
            profit = float(pos.profit)
            lots = float(pos.volume)

            # Try to get pip value safely
            pip_val = None
            if hasattr(self.broker, 'cache') and hasattr(self.broker.cache, 'get_pip_value'):
                pip_val = self.broker.cache.get_pip_value(symbol, lots)
            else:
                pip_val = 1.0  # fallback to 1 to avoid division by zero, or set to None

            if not pip_val or pip_val <= 0:
                continue

            floating_pips = profit / pip_val

            # Evaluate scalable performance milestones instead of flat text values
            if 15.0 <= floating_pips <= 25.0:
                return f"{symbol} is approaching standard initial target channels."
            elif floating_pips > 40.0:
                return f"Exceptional performance detected. {symbol} floating higher than 40 pips."
            elif floating_pips < -20.0:
                return f"Warning: {symbol} testing protective Stop Loss bounds."
        return None
    
    def _check_session_timing(self) -> Optional[str]:
        """Check for notable session timing events."""
        now = datetime.utcnow()
        hour = now.hour

        # London close approaching
        if hour == 15 and now.minute < 30:
            try:
                positions = self.broker.getPositions()
                if positions:
                    return "London close in about 30 minutes. Spreads will widen. Any positions you want to close before then?"
            except Exception:
                pass

        # NY session opening
        if hour == 13 and now.minute < 15:
            return "New York session is opening. Usually the most liquid window of the day."

        return None

    def _check_promises(self) -> Optional[str]:
        """Check if any open promises should be actioned."""
        if not self.wm.open_promises:
            return None

        promise = self.wm.open_promises[0]
        
        try:
            # Simple keyword matching against current state
            positions = self.broker.getPositions() or []
            symbols_open = [p.symbol for p in positions]
            
            if "alert" in promise.lower() and "when" in promise.lower():
                # Parse the symbol from the promise
                for sym in self.wm.symbols_discussed:
                    if sym in promise and sym in symbols_open:
                        self.wm.fulfill_promise(promise)
                        return f"Earlier you asked me to watch {sym} — it's showing movement now."
        except Exception:
            pass
        
        return None
