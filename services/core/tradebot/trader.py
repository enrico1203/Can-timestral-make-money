"""Live paper trader. Steps the same Engine used by backtests, one closed candle at a time.

- configuration comes from the `live_config` setting (editable from the UI);
- control flags from the `control` setting: paused, flatten, reset_halt, reset_run;
- engine state is persisted in runs.state after every bar, so restarts resume exactly;
- if the trader was down, missed bars are replayed in order (forecast + step for each).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time

import numpy as np
import pandas as pd
from psycopg.types.json import Jsonb

from . import config, db
from .backtest.generate import generate
from .engine import Engine, EngineConfig, Records
from .features import ForecasterClient
from .strategy import Bar, make_strategy

log = logging.getLogger("trader")
SRC = "trader"
WARMUP_BARS = 720


def cfg_hash(cfg: dict) -> str:
    return hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]


def load_live_config(conn) -> dict:
    cfg = db.get_setting(conn, "live_config")
    if not cfg:
        cfg = json.loads(json.dumps(config.DEFAULT_LIVE_CONFIG))
        db.set_setting(conn, "live_config", cfg)
    return cfg


def build_engine(cfg: dict) -> Engine:
    n = len(cfg["symbols"])
    risk = cfg.get("risk", {})
    max_frac = risk.get("max_frac_per_symbol") or 1.0 / n
    strat = make_strategy(cfg["strategy"]["name"], cfg["fee"] + cfg["slippage"], max_frac, cfg["strategy"].get("params"))
    ecfg = EngineConfig(initial_cash=cfg["initial_cash"], fee=cfg["fee"], slippage=cfg["slippage"],
                        rebalance_band=risk.get("rebalance_band", 0.05), daily_loss_limit=risk.get("daily_loss_limit"),
                        max_drawdown_halt=risk.get("max_drawdown_halt"))
    return Engine(strat, ecfg, record_all_signals=True)


class LiveTrader:
    def __init__(self):
        config.assert_paper_mode()
        self.conn = db.connect()
        db.init_schema(self.conn)
        self.client = ForecasterClient()
        self.tf = config.TIMEFRAME
        self.step_td = pd.Timedelta(seconds=config.TIMEFRAME_SECONDS[self.tf])
        self.cfg = load_live_config(self.conn)
        db.set_setting(self.conn, "trader_heartbeat", {"ts": db.utcnow().isoformat(), "phase": "starting"})
        self.run_id, self.engine = self._load_or_create_run()

    # ------------------------------------------------------------------ run lifecycle
    def _load_or_create_run(self):
        row = self.conn.execute("SELECT * FROM runs WHERE kind='live' AND status<>'stopped' ORDER BY id DESC LIMIT 1").fetchone()
        if row and row["state"]:
            eng = build_engine(row["config"])
            eng.load_state(row["state"])
            self.cfg_hash = cfg_hash(row["config"])
            if cfg_hash(self.cfg) != self.cfg_hash:
                eng = self._apply_config(eng, row["id"])
            db.event(self.conn, "info", SRC, f"resumed live run #{row['id']} at {eng.last_ts}, equity {eng.equity():.2f}")
            return row["id"], eng
        return self._new_run()

    def _new_run(self):
        cfg = self.cfg
        rid = self.conn.execute("INSERT INTO runs(kind,name,status,config) VALUES ('live',%s,'running',%s) RETURNING id",
                                (f"live {cfg['strategy']['name']} {cfg['model'].split('/')[-1]}", Jsonb(cfg))).fetchone()["id"]
        self.cfg_hash = cfg_hash(cfg)
        eng = build_engine(cfg)
        self._warmup(eng)
        self._persist(rid, eng)
        db.event(self.conn, "info", SRC, f"started new live paper run #{rid} with {cfg['initial_cash']:.0f} USDT, "
                                         f"strategy {cfg['strategy']['name']} {cfg['strategy'].get('params')}")
        return rid, eng

    def _apply_config(self, eng: Engine, rid: int) -> Engine:
        """Config edited from the UI: keep cash/positions, rebuild strategy and risk settings."""
        state = eng.dump_state()
        old_strategy = eng.strategy.name
        new = build_engine(self.cfg)
        keep_strategy_state = new.strategy.name == old_strategy
        if not keep_strategy_state:
            state["strategy"] = None
        new.load_state(state)
        if not keep_strategy_state:
            self._warmup(new, until=new.last_ts)
        self.conn.execute("UPDATE runs SET config=%s, name=%s WHERE id=%s",
                          (Jsonb(self.cfg), f"live {self.cfg['strategy']['name']} {self.cfg['model'].split('/')[-1]}", rid))
        self.cfg_hash = cfg_hash(self.cfg)
        db.event(self.conn, "info", SRC, f"configuration updated: strategy {self.cfg['strategy']}")
        return new

    def _warmup(self, eng: Engine, until=None):
        """Feed the strategy the last WARMUP_BARS bars (forecasts generated/cached as needed) without trading."""
        if not eng.strategy.uses_forecast and eng.strategy.name != "sma_cross":
            return
        latest = until or self._latest_common_ts()
        if latest is None:
            return
        start = latest - WARMUP_BARS * self.step_td
        bars = self._bars(start, latest + self.step_td)
        for t in sorted(bars):
            for b in bars[t].values():
                eng.strategy.observe(b)
        if until is None:
            # first step will be the bar after `latest`; remember where we are
            eng.last_ts = latest
            for sym, b in bars.get(latest, {}).items():
                eng.last_close[sym] = b.close

    # ------------------------------------------------------------------ data
    def _latest_common_ts(self):
        rows = self.conn.execute("SELECT symbol, max(ts) m FROM candles WHERE timeframe=%s AND symbol = ANY(%s) GROUP BY symbol",
                                 (self.tf, self.cfg["symbols"])).fetchall()
        if len(rows) < len(self.cfg["symbols"]):
            return None
        return pd.Timestamp(min(r["m"] for r in rows)).tz_convert("UTC")

    def _bars(self, start, end) -> dict:
        """{ts: {sym: Bar}} for ts in [start, end) with forecasts (generated on demand)."""
        cfg = self.cfg
        model, fs, ctx, hz = cfg["model"], cfg["feature_set"], cfg["context_len"], cfg["horizon"]
        if self.engine_uses_forecast(cfg):
            generate(self.conn, model, cfg["symbols"], start, end, ctx, hz, fs, batch=64, timeframe=self.tf,
                     client=self.client)
        out: dict = {}
        for sym in cfg["symbols"]:
            df = db.load_candles(self.conn, sym, self.tf, start, end)
            if self.engine_uses_forecast(cfg):
                idx, Q, _ = db.load_forecasts(self.conn, model, sym, self.tf, fs, ctx, start, end)
                fmap = {t: Q[i] for i, t in enumerate(idx)}
            else:
                fmap = {}
            for t, r in df.iterrows():
                out.setdefault(t, {})[sym] = Bar(sym, t, r["open"], r["high"], r["low"], r["close"], fmap.get(t))
        return out

    @staticmethod
    def engine_uses_forecast(cfg) -> bool:
        return cfg["strategy"]["name"] not in ("buy_hold", "sma_cross")

    # ------------------------------------------------------------------ persistence
    def _persist(self, rid: int, eng: Engine, rec: Records | None = None):
        c = self.conn
        rec = rec or eng.rec.drain()
        for o in rec.orders:
            if o["status"] == "pending":
                c.execute("INSERT INTO orders(run_id,local_id,symbol,created_ts,side,qty,reason,status) "
                          "VALUES (%s,%s,%s,%s,%s,%s,%s,'pending') ON CONFLICT (run_id,local_id) DO NOTHING",
                          (rid, o["id"], o["symbol"], o["created_ts"], "buy" if o["qty"] > 0 else "sell", abs(o["qty"]), o["reason"]))
            else:
                c.execute("INSERT INTO orders(run_id,local_id,symbol,created_ts,side,qty,reason,status,fill_ts,fill_price,fee,slippage) "
                          "VALUES (%s,%s,%s,%s,%s,%s,%s,'filled',%s,%s,%s,%s) ON CONFLICT (run_id,local_id) DO UPDATE SET "
                          "status='filled', fill_ts=EXCLUDED.fill_ts, fill_price=EXCLUDED.fill_price, fee=EXCLUDED.fee, slippage=EXCLUDED.slippage",
                          (rid, o["id"], o["symbol"], o["created_ts"], "buy" if o["qty"] > 0 else "sell", abs(o["qty"]),
                           o["reason"], o["fill_ts"], o["fill_price"], o["fee"], o["slippage"]))
        for t in rec.trades:
            c.execute("INSERT INTO trades(run_id,symbol,side,qty,entry_ts,entry_price,exit_ts,exit_price,pnl,fees,bars,exit_reason) "
                      "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                      (rid, t["symbol"], t["side"], t["qty"], t["entry_ts"], t["entry_price"], t["exit_ts"], t["exit_price"],
                       t["pnl"], t["fees"], t["bars"], t["exit_reason"]))
        for e in rec.equity:
            c.execute("INSERT INTO equity(run_id,ts,cash,positions_value,equity,drawdown,exposure) VALUES (%s,%s,%s,%s,%s,%s,%s) "
                      "ON CONFLICT (run_id,ts) DO UPDATE SET equity=EXCLUDED.equity, cash=EXCLUDED.cash, "
                      "positions_value=EXCLUDED.positions_value, drawdown=EXCLUDED.drawdown, exposure=EXCLUDED.exposure",
                      (rid, e["ts"], e["cash"], e["positions_value"], e["equity"], e["drawdown"], e["exposure"]))
        for s in rec.signals:
            c.execute("INSERT INTO signals(run_id,symbol,ts,p_up,p_down,exp_ret,width,target,action,reason) "
                      "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                      (rid, s["symbol"], s["ts"], s["p_up"], s["p_down"], s["exp_ret"], s["width"], s["target"],
                       s["action"], s["reason"]))
        for level, msg in rec.events:
            db.event(c, level, SRC, msg)
        status = "halted" if eng.halted else "running"
        c.execute("UPDATE runs SET state=%s, status=%s WHERE id=%s", (Jsonb(db.jsonable(eng.dump_state())), status, rid))

    # ------------------------------------------------------------------ control
    def _handle_control(self):
        ctl = db.get_setting(self.conn, "control", {}) or {}
        changed = False
        if ctl.get("reset_run"):
            self.conn.execute("UPDATE runs SET status='stopped', finished_at=now() WHERE id=%s", (self.run_id,))
            db.event(self.conn, "warning", SRC, f"live run #{self.run_id} reset from the UI")
            ctl["reset_run"] = False
            db.set_setting(self.conn, "control", ctl)
            self.cfg = load_live_config(self.conn)
            self.run_id, self.engine = self._new_run()
            return ctl
        if ctl.get("flatten"):
            ts = self.engine.last_ts or db.utcnow()
            self.engine.flatten(ts, "manual flatten from UI")
            ctl["flatten"] = False
            changed = True
            db.event(self.conn, "warning", SRC, "flatten requested: all positions will be closed at the next open")
        if ctl.get("reset_halt"):
            self.engine.reset_halt()
            ctl["reset_halt"] = False
            changed = True
            db.event(self.conn, "info", SRC, "kill-switch reset from the UI")
        if changed:
            db.set_setting(self.conn, "control", ctl)
            self._persist(self.run_id, self.engine)
        return ctl

    # ------------------------------------------------------------------ main loop
    def tick(self) -> int:
        """Process every fully available bar after engine.last_ts. Returns number of bars processed."""
        new_cfg = load_live_config(self.conn)
        if cfg_hash(new_cfg) != self.cfg_hash:
            self.cfg = new_cfg
            self.engine = self._apply_config(self.engine, self.run_id)
        ctl = self._handle_control()
        latest = self._latest_common_ts()
        if latest is None:
            return 0
        last = self.engine.last_ts
        start = (last + self.step_td) if last is not None else latest
        if start > latest:
            return 0
        bars = self._bars(start, latest + self.step_td)
        n = 0
        for t in sorted(bars):
            self.engine.step(t, bars[t], allow_orders=not ctl.get("paused", False))
            self._persist(self.run_id, self.engine)
            n += 1
        if n:
            eq = self.engine.equity()
            log.info("processed %d bar(s) up to %s, equity %.2f, positions %s", n, latest, eq,
                     {s: round(p["qty"], 6) for s, p in self.engine.positions.items()})
        return n

    def loop(self):
        db.set_setting(self.conn, "trader_heartbeat", {"ts": db.utcnow().isoformat(), "run_id": self.run_id})
        while True:
            try:
                self.tick()
                db.set_setting(self.conn, "trader_heartbeat", {"ts": db.utcnow().isoformat(), "run_id": self.run_id})
            except Exception as e:
                log.exception("trader tick failed")
                try:
                    db.event(self.conn, "error", SRC, f"tick failed: {e}")
                except Exception:
                    self.conn = db.connect()
            time.sleep(15)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    LiveTrader().loop()


if __name__ == "__main__":
    main()
