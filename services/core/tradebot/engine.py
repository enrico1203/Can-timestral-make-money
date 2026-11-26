"""Portfolio engine + paper broker. The SAME code runs backtests and live paper trading.

Timing model, for every bar t (ts = candle open time):
  1. pending market orders (decided at the close of t-1) are filled at OPEN[t] +/- slippage, plus fee;
  2. stop-loss / take-profit are checked against HIGH/LOW[t]; if both are touched in the same bar the
     stop is assumed to have been hit first (pessimistic); a gap through the level fills at the open;
  3. positions are marked to CLOSE[t], equity and drawdown are updated, risk limits are checked;
  4. the strategy sees CLOSE[t] and the forecast made with data up to t, and sets a target;
     orders to reach it are queued for OPEN[t+1].
Nothing in step 4 can be filled at a price the strategy has already seen.

Shorts are simulated like a perpetual future without funding (stated in the report).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .strategy import Bar, Decision, PosView, Strategy


@dataclass
class EngineConfig:
    initial_cash: float = 10_000.0
    fee: float = 0.001
    slippage: float = 0.0003
    rebalance_band: float = 0.05        # min |Δexposure| (fraction of equity) to resize a same-side position
    max_gross: float = 1.0              # no leverage
    daily_loss_limit: float | None = None
    max_drawdown_halt: float | None = None
    min_order_value: float = 10.0


@dataclass
class Records:
    orders: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    equity: list = field(default_factory=list)
    signals: list = field(default_factory=list)
    events: list = field(default_factory=list)

    def drain(self) -> "Records":
        out = Records(self.orders, self.trades, self.equity, self.signals, self.events)
        self.orders, self.trades, self.equity, self.signals, self.events = [], [], [], [], []
        return out


class Engine:
    def __init__(self, strategy: Strategy, cfg: EngineConfig, record_all_signals: bool = False):
        self.strategy = strategy
        self.cfg = cfg
        self.record_all_signals = record_all_signals
        self.cash = cfg.initial_cash
        self.positions: dict[str, dict] = {}
        self.pending: list[dict] = []
        self.last_close: dict[str, float] = {}
        self.peak = cfg.initial_cash
        self.day = None
        self.day_start_equity = cfg.initial_cash
        self.day_blocked = None
        self.halted = False
        self.halt_reason = None
        self.next_order_id = 1
        self.fees_paid = 0.0
        self.last_ts = None
        self.last_eq = cfg.initial_cash
        self.rec = Records()

    # ------------------------------------------------------------ persistence
    def dump_state(self) -> dict:
        return {
            "cash": self.cash, "positions": self.positions, "pending": self.pending,
            "last_close": self.last_close, "peak": self.peak, "day": self.day,
            "day_start_equity": self.day_start_equity, "day_blocked": self.day_blocked,
            "halted": self.halted, "halt_reason": self.halt_reason,
            "next_order_id": self.next_order_id, "fees_paid": self.fees_paid, "last_eq": self.last_eq,
            "last_ts": None if self.last_ts is None else pd.Timestamp(self.last_ts).isoformat(),
            "strategy": self.strategy.dump_state(),
        }

    def load_state(self, s: dict) -> None:
        self.cash = s["cash"]
        self.positions = s["positions"]
        self.pending = s["pending"]
        self.last_close = s["last_close"]
        self.peak = s["peak"]
        self.day = s["day"]
        self.day_start_equity = s["day_start_equity"]
        self.day_blocked = s["day_blocked"]
        self.halted = s["halted"]
        self.halt_reason = s["halt_reason"]
        self.next_order_id = s["next_order_id"]
        self.fees_paid = s.get("fees_paid", 0.0)
        self.last_eq = s.get("last_eq", self.cash)
        self.last_ts = pd.Timestamp(s["last_ts"]) if s.get("last_ts") else None
        self.strategy.load_state(s.get("strategy"))

    # ------------------------------------------------------------ accounting
    def equity(self) -> float:
        return self.cash + sum(p["qty"] * self.last_close.get(sym, p["avg"]) for sym, p in self.positions.items())

    def gross_exposure(self) -> float:
        eq = self.equity()
        if eq <= 0:
            return math.inf
        return sum(abs(p["qty"]) * self.last_close.get(s, p["avg"]) for s, p in self.positions.items()) / eq

    def _fill(self, sym: str, qty: float, ref_price: float, ts, reason: str, order: dict | None = None,
              meta: dict | None = None) -> None:
        """Execute qty (signed) at ref_price with slippage and fee."""
        side = 1 if qty > 0 else -1
        price = ref_price * (1 + side * self.cfg.slippage)
        fee = abs(qty) * price * self.cfg.fee
        slip = abs(qty) * ref_price * self.cfg.slippage
        self.cash -= qty * price + fee
        self.fees_paid += fee
        if order is None:
            order = {"id": self._new_id(), "symbol": sym, "qty": qty, "created_ts": ts, "reason": reason}
        self.rec.orders.append({**order, "status": "filled", "fill_ts": ts, "fill_price": price, "fee": fee,
                                "slippage": slip})

        pos = self.positions.get(sym)
        remaining = qty
        if pos is not None and np.sign(pos["qty"]) != side:
            closing = min(abs(qty), abs(pos["qty"]))
            frac_fee = fee * closing / abs(qty)
            pos["realized"] += closing * (price - pos["avg"]) * np.sign(pos["qty"])
            pos["fees"] += frac_fee
            pos["qty"] += side * closing
            remaining = qty - side * closing
            fee -= frac_fee
            if abs(pos["qty"]) < 1e-12:
                self._close_trade(sym, pos, ts, price, reason)
                del self.positions[sym]
                pos = None
        if abs(remaining) > 1e-12:
            if pos is None:
                pos = {"qty": 0.0, "avg": price, "fees": 0.0, "realized": 0.0, "entry_ts": pd.Timestamp(ts).isoformat(),
                       "bars": 0, "max_qty": 0.0, "stop": None, "tp": None, "dir": side}
                self.positions[sym] = pos
            new_qty = pos["qty"] + remaining
            pos["avg"] = (pos["avg"] * pos["qty"] + price * remaining) / new_qty
            pos["qty"] = new_qty
            pos["fees"] += fee
            pos["max_qty"] = max(pos["max_qty"], abs(new_qty))
        if pos is not None and meta:
            if meta.get("stop") is not None or meta.get("tp") is not None:
                pos["stop"], pos["tp"] = meta.get("stop"), meta.get("tp")

    def _close_trade(self, sym, pos, ts, price, reason):
        self.rec.trades.append({
            "symbol": sym, "side": "long" if pos.get("dir", 1) > 0 else "short",
            "qty": pos["max_qty"], "entry_ts": pos["entry_ts"], "entry_price": pos["avg"],
            "exit_ts": pd.Timestamp(ts).isoformat(), "exit_price": price,
            "pnl": pos["realized"] - pos["fees"], "fees": pos["fees"], "bars": pos["bars"], "exit_reason": reason,
        })

    def _new_id(self) -> int:
        i = self.next_order_id
        self.next_order_id += 1
        return i

    # ------------------------------------------------------------ main step
    def flatten(self, ts, reason: str = "manual flatten") -> None:
        """Queue orders closing every position at the next open."""
        self.pending = []
        for sym, p in self.positions.items():
            o = {"id": self._new_id(), "symbol": sym, "qty": -p["qty"], "created_ts": pd.Timestamp(ts).isoformat(),
                 "reason": reason, "meta": None}
            self.pending.append(o)
            self.rec.orders.append({k: o[k] for k in ("id", "symbol", "qty", "created_ts", "reason")} | {"status": "pending"})

    def reset_halt(self) -> None:
        self.halted = False
        self.halt_reason = None
        self.peak = self.equity()

    def step(self, ts, bars: dict[str, Bar], allow_orders: bool = True) -> None:
        ts = pd.Timestamp(ts)
        self.last_ts = ts
        valid = {s: b for s, b in bars.items() if b is not None and np.isfinite(b.close)}

        # 1. fill pending orders at the open
        still = []
        for o in self.pending:
            b = valid.get(o["symbol"])
            if b is None or not np.isfinite(b.open):
                still.append(o)
                continue
            self._fill(o["symbol"], o["qty"], b.open, ts, o["reason"], order={k: o[k] for k in
                       ("id", "symbol", "qty", "created_ts", "reason")}, meta=o.get("meta"))
        self.pending = still

        # 2. stops / take profits on the bar range
        for sym, b in valid.items():
            pos = self.positions.get(sym)
            if not pos:
                continue
            stop, tp, q = pos.get("stop"), pos.get("tp"), pos["qty"]
            exit_px, why = None, None
            if q > 0:
                if stop is not None and b.low <= stop:
                    exit_px, why = (b.open if b.open <= stop else stop), "stop_loss"
                elif tp is not None and b.high >= tp:
                    exit_px, why = (b.open if b.open >= tp else tp), "take_profit"
            elif q < 0:
                if stop is not None and b.high >= stop:
                    exit_px, why = (b.open if b.open >= stop else stop), "stop_loss"
                elif tp is not None and b.low <= tp:
                    exit_px, why = (b.open if b.open <= tp else tp), "take_profit"
            if exit_px is not None:
                pos["bars"] += 1
                self._fill(sym, -q, exit_px, ts, why)
                # cancel queued orders for this symbol (none at this point, they were filled at the open)

        # 3. mark to market, risk
        for sym, b in valid.items():
            self.last_close[sym] = float(b.close)
            if sym in self.positions:
                self.positions[sym]["bars"] += 1
        eq = self.equity()
        self.peak = max(self.peak, eq)
        dd = eq / self.peak - 1 if self.peak > 0 else 0.0
        day = ts.strftime("%Y-%m-%d")
        if self.day != day:
            # equity at the close of the previous bar is the start-of-day reference
            self.day_start_equity = self.last_eq
            self.day = day
        cfg = self.cfg
        if cfg.max_drawdown_halt is not None and not self.halted and dd <= -cfg.max_drawdown_halt:
            self.halted = True
            self.halt_reason = f"max drawdown {dd:.1%} reached at {ts}"
            self.rec.events.append(("warning", self.halt_reason))
        if (cfg.daily_loss_limit is not None and self.day_blocked != day
                and eq <= self.day_start_equity * (1 - cfg.daily_loss_limit)):
            self.day_blocked = day
            self.rec.events.append(("warning", f"daily loss limit hit on {day}: trading paused until next UTC day"))
        exposure = sum(p["qty"] * self.last_close.get(s, p["avg"]) for s, p in self.positions.items()) / eq if eq > 0 else 0
        self.rec.equity.append({"ts": ts, "cash": self.cash, "positions_value": eq - self.cash, "equity": eq,
                                "drawdown": dd, "exposure": exposure})
        self.last_eq = eq

        # 4. strategy decisions -> orders for next open
        blocked = self.halted or self.day_blocked == day
        for sym, b in valid.items():
            pos = self.positions.get(sym)
            qty = pos["qty"] if pos else 0.0
            frac = qty * b.close / eq if eq > 0 else 0.0
            view = PosView(frac=frac, bars_held=pos["bars"] if pos else 0, side=int(np.sign(qty)))
            d = self.strategy.decide(b, view)
            if blocked:
                d = Decision(0.0, "FLAT", ("HALTED: " + str(self.halt_reason)) if self.halted else "daily loss limit",
                             p_up=d.p_up, p_down=d.p_down, exp_ret=d.exp_ret, width=d.width)
            order_qty = self._order_for_target(sym, d, qty, b.close, eq) if allow_orders else 0.0
            if order_qty != 0.0:
                o = {"id": self._new_id(), "symbol": sym, "qty": order_qty, "created_ts": ts.isoformat(),
                     "reason": f"{d.action}: {d.reason}",
                     "meta": {"stop": d.stop, "tp": d.take_profit} if d.action in ("LONG", "SHORT") else None}
                self.pending.append(o)
                self.rec.orders.append({k: o[k] for k in ("id", "symbol", "qty", "created_ts", "reason")} | {"status": "pending"})
            if self.record_all_signals or order_qty != 0.0:
                self.rec.signals.append({"symbol": sym, "ts": ts, "p_up": d.p_up, "p_down": d.p_down,
                                         "exp_ret": d.exp_ret, "width": d.width, "target": d.target,
                                         "action": d.action,
                                         "reason": d.reason})

    def _order_for_target(self, sym: str, d: Decision, qty: float, price: float, eq: float) -> float:
        if eq <= 0 or not np.isfinite(price) or price <= 0:
            return 0.0
        target_qty = d.target * eq / price
        cur_val, tgt_val = qty * price, target_qty * price
        if d.target == 0.0:
            diff = -qty
        elif np.sign(target_qty) != np.sign(qty):
            diff = target_qty - qty
        elif abs(tgt_val - cur_val) > self.cfg.rebalance_band * eq:
            diff = target_qty - qty
        else:
            diff = 0.0
        # never exceed gross exposure limit
        if diff != 0.0:
            proj: dict[str, float] = {s: p["qty"] for s, p in self.positions.items()}
            for o in self.pending:
                proj[o["symbol"]] = proj.get(o["symbol"], 0.0) + o["qty"]
            other = sum(abs(q) * self.last_close.get(s, price) for s, q in proj.items() if s != sym)
            room = max(0.0, self.cfg.max_gross * eq - other)
            new_val = abs((qty + diff) * price)
            if new_val > room + 1e-9:
                capped = room / price * np.sign(qty + diff)
                diff = capped - qty
        if abs(diff * price) < self.cfg.min_order_value:
            return 0.0
        return float(diff)
