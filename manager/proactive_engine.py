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

    def _check_positions(self) -> Optional[str]:
        """Monitor open positions for notable events without repeating alerts."""
        try:
            positions = self.broker.getPositions()
        except Exception:
            return None
            
        if not positions:
            self._notified_positions.clear()
            return None

        # Clean up stale tickets from tracking to prevent memory leaks
        current_tickets = {getattr(p, 'ticket', None) for p in positions}
        for stale_ticket in list(self._notified_positions.keys()):
            if stale_ticket not in current_tickets:
                del self._notified_positions[stale_ticket]

        for pos in positions:
            try:
                ticket = getattr(pos, 'ticket', None)
                if ticket is None:
                    continue

                if ticket not in self._notified_positions:
                    self._notified_positions[ticket] = set()

                profit = pos.profit
                symbol = pos.symbol
                notified = self._notified_positions[ticket]

                # Position approaching a nice round number
                if 8.0 <= profit <= 12.0:
                    if "near_10" not in notified:
                        notified.add("near_10")
                        return f"Your {symbol} position is sitting at +${profit:.2f}. Getting close to $10 — worth watching."

                if profit > 20.0 and symbol in self.wm.symbols_discussed:
                    if "up_20" not in notified:
                        notified.add("up_20")
                        return f"{symbol} is up ${profit:.2f}. You were watching this one — still want to let it run?"

                # Position moving against for a while
                if profit < -5.0:
                    if "down_5" not in notified:
                        notified.add("down_5")
                        return f"{symbol} is down ${abs(profit):.2f}. Still within plan, but heads up."
            except Exception:
                continue

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
