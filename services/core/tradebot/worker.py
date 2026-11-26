"""Background worker: executes jobs queued from the web UI (forecast generation, evaluation, backtests)."""

from __future__ import annotations

import logging
import time
import traceback

from psycopg.types.json import Jsonb

from . import config, db
from .backtest import evaluate, generate, run

log = logging.getLogger("worker")


def execute(conn, job: dict) -> dict:
    p = job["params"]
    jid = job["id"]
    last = [0.0]

    def progress(frac: float, msg: str = ""):
        now = time.time()
        if now - last[0] > 2 or frac >= 1:
            last[0] = now
            conn.execute("UPDATE jobs SET progress=%s, message=%s WHERE id=%s", (float(frac), msg[:500], jid))

    symbols = p.get("symbols") or config.DEFAULT_SYMBOLS
    kind = job["kind"]
    if kind == "generate":
        res = {}
        for m in p["models"]:
            res[m] = generate.generate(conn, m, symbols, p["start"], p.get("end"), p.get("context_len", 1024),
                                       p.get("horizon", 24), p.get("feature_set", "close"), progress=progress)
        return res
    if kind == "evaluate":
        out = {}
        for m in p["models"]:
            r = evaluate.evaluate(conn, m, symbols, p["start"], p.get("end"), p.get("feature_set", "close"),
                                  p.get("context_len", 1024))
            out[m] = r["summary"]
        return out
    if kind == "backtest":
        spec = {**p, "symbols": symbols}
        cache = run.DataCache(conn, symbols, spec["start"], spec.get("end"))
        progress(0.3, "running")
        r = run.run_one(cache, spec, mc=int(p.get("mc", 500)), keep_series=True)
        rid = run.save_run(conn, r, p.get("name") or f"{spec['strategy']} {str(spec.get('model', '-')).split('/')[-1]}")
        return {"run_id": rid, "metrics": db.jsonable(r["metrics"])}
    if kind == "suite":
        r = run.suite(conn, p["models"], symbols, tuple(p["tune"]), tuple(p["validation"]), tuple(p["holdout"]),
                      p.get("feature_set", "close"), p.get("context_len", 1024), int(p.get("mc", 1000)),
                      progress=progress, quick=bool(p.get("quick")))
        return {"verdict": r["verdict"], "group": r["group"]}
    raise ValueError(f"unknown job kind {kind}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config.assert_paper_mode()
    conn = db.connect()
    db.init_schema(conn)
    # jobs left running by a previous crash are marked as failed
    conn.execute("UPDATE jobs SET status='error', message='worker restarted' WHERE status='running'")
    while True:
        job = conn.execute(
            "UPDATE jobs SET status='running', started_at=now() WHERE id = (SELECT id FROM jobs WHERE status='queued' "
            "ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED) RETURNING *").fetchone()
        if not job:
            time.sleep(3)
            continue
        log.info("job #%d %s %s", job["id"], job["kind"], job["params"])
        try:
            res = execute(conn, job)
            conn.execute("UPDATE jobs SET status='done', progress=1, finished_at=now(), result=%s WHERE id=%s",
                         (Jsonb(db.jsonable(res)), job["id"]))
        except Exception as e:
            log.exception("job failed")
            conn.execute("UPDATE jobs SET status='error', finished_at=now(), message=%s WHERE id=%s",
                         (f"{e}\n{traceback.format_exc()[-1500:]}", job["id"]))


if __name__ == "__main__":
    main()
