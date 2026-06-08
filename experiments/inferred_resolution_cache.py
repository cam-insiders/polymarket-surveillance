from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _norm_min_usd(value: Optional[float]) -> Optional[float]:
    return float(value) if value is not None else None


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS inference_jobs (
            job_id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at_utc TEXT NOT NULL,
            data_dir TEXT NOT NULL,
            resolution_threshold REAL NOT NULL,
            min_trades INTEGER NOT NULL,
            min_usd_amount REAL,
            total_markets INTEGER NOT NULL,
            resolved_markets INTEGER NOT NULL,
            unresolved_markets INTEGER NOT NULL,
            too_few_trades_markets INTEGER NOT NULL,
            no_trades_markets INTEGER NOT NULL,
            error_markets INTEGER NOT NULL,
            min_closed_time_utc TEXT,
            max_closed_time_utc TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS inferred_market_resolutions (
            job_id INTEGER NOT NULL,
            market_id INTEGER NOT NULL,
            market_slug TEXT,
            closed_time_utc TEXT,
            volume REAL,
            n_trades INTEGER NOT NULL,
            inferred_winning_outcome INTEGER,
            inference_status TEXT NOT NULL,
            latest_trade_ts_ms INTEGER,
            updated_at_utc TEXT NOT NULL,
            PRIMARY KEY (job_id, market_id),
            FOREIGN KEY (job_id) REFERENCES inference_jobs(job_id)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_imr_market_id ON inferred_market_resolutions(market_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_imr_closed_time ON inferred_market_resolutions(closed_time_utc)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_imr_status ON inferred_market_resolutions(inference_status)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS market_resolution_cache (
            market_id INTEGER PRIMARY KEY,
            market_slug TEXT,
            closed_time_utc TEXT,
            volume REAL,
            n_trades INTEGER NOT NULL,
            inferred_winning_outcome INTEGER,
            inference_status TEXT NOT NULL,
            latest_trade_ts_ms INTEGER,
            resolution_threshold REAL,
            updated_at_utc TEXT NOT NULL,
            source TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_mrc_status ON market_resolution_cache(inference_status)"
    )
    conn.commit()


def upsert_market_resolution_cache(
    *,
    db_path: str,
    rows: List[Dict],
    resolution_threshold: Optional[float] = None,
    source: str = "experiment",
) -> int:
    """
    Upsert per-market inferred resolutions into the canonical cache table.
    """
    if not rows:
        return 0

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(db_path)
    except Exception:
        return 0

    try:
        _init_db(conn)
        updated_at_utc = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            """
            INSERT OR REPLACE INTO market_resolution_cache (
                market_id,
                market_slug,
                closed_time_utc,
                volume,
                n_trades,
                inferred_winning_outcome,
                inference_status,
                latest_trade_ts_ms,
                resolution_threshold,
                updated_at_utc,
                source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    int(row["market_id"]),
                    row.get("market_slug"),
                    row.get("closed_time_utc"),
                    row.get("volume"),
                    int(row.get("n_trades", 0) or 0),
                    (
                        int(row["inferred_winning_outcome"])
                        if row.get("inferred_winning_outcome") is not None
                        else None
                    ),
                    str(row.get("inference_status", "unresolved")),
                    (
                        int(row["latest_trade_ts_ms"])
                        if row.get("latest_trade_ts_ms") is not None
                        else None
                    ),
                    float(resolution_threshold) if resolution_threshold is not None else None,
                    updated_at_utc,
                    str(source),
                )
                for row in rows
            ],
        )
        conn.commit()
        return len(rows)
    except Exception:
        return 0
    finally:
        conn.close()


def _row_to_resolution_dict(row: sqlite3.Row) -> Tuple[int, Dict]:
    mid = int(row["market_id"])
    return mid, {
        "market_id": mid,
        "market_slug": row["market_slug"],
        "closed_time_utc": row["closed_time_utc"],
        "volume": row["volume"],
        "n_trades": int(row["n_trades"] or 0),
        "inferred_winning_outcome": (
            int(row["inferred_winning_outcome"])
            if row["inferred_winning_outcome"] is not None
            else None
        ),
        "inference_status": str(row["inference_status"]),
        "latest_trade_ts_ms": (
            int(row["latest_trade_ts_ms"])
            if row["latest_trade_ts_ms"] is not None
            else None
        ),
        "resolution_threshold": row["resolution_threshold"],
        "updated_at_utc": row["updated_at_utc"],
        "source": row["source"],
    }


def load_cached_resolution_rows(
    *,
    db_path: str,
    market_ids: List[int],
) -> Dict[int, Dict]:
    """
    Load cached per-market resolution rows for the requested markets.
    """
    if not market_ids:
        return {}

    resolved = Path(db_path).expanduser().resolve()
    if not resolved.is_file():
        logging.warning(
            "Resolution cache DB not found (will infer from trades): %s | cwd=%s | "
            "use an absolute --inferred-resolutions-db if batch jobs start outside the repo.",
            resolved,
            Path.cwd(),
        )
        return {}

    try:
        # mode=ro: never create a new file; fail clearly if path is wrong.
        uri = f"{resolved.as_uri()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
    except Exception as exc:
        logging.warning("Could not open resolution cache %s: %s", resolved, exc)
        return {}

    # Stay under SQLITE_MAX_VARIABLE_NUMBER on typical builds.
    _batch_size = 500
    query = (
        "SELECT market_id, market_slug, closed_time_utc, volume, n_trades, "
        "inferred_winning_outcome, inference_status, latest_trade_ts_ms, "
        "resolution_threshold, updated_at_utc, source "
        "FROM market_resolution_cache "
        "WHERE market_id IN ({})"
    )

    out: Dict[int, Dict] = {}
    try:
        # Read-only: do not run _init_db (CREATE) here.
        ids = [int(mid) for mid in market_ids]
        for start in range(0, len(ids), _batch_size):
            chunk = ids[start : start + _batch_size]
            placeholders = ",".join("?" for _ in chunk)
            rows = conn.execute(query.format(placeholders), chunk).fetchall()
            for row in rows:
                mid, d = _row_to_resolution_dict(row)
                out[mid] = d
        return out
    except Exception as exc:
        logging.warning(
            "Failed to load resolution cache from %s (%s markets): %s",
            resolved,
            len(market_ids),
            exc,
        )
        return {}
    finally:
        conn.close()


def save_inferred_resolutions_job(
    *,
    db_path: str,
    data_dir: str,
    resolution_threshold: float,
    min_trades: int,
    min_usd_amount: Optional[float],
    rows: List[Dict],
) -> Optional[int]:
    """
    Persist a full inferred-resolution run to SQLite for reuse by later experiments.
    """
    if not rows:
        return None

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        conn = sqlite3.connect(db_path)
    except Exception:
        return None

    try:
        _init_db(conn)
        upsert_market_resolution_cache(
            db_path=db_path,
            rows=rows,
            resolution_threshold=resolution_threshold,
            source="job",
        )

        created_at_utc = datetime.now(timezone.utc).isoformat()
        resolved = sum(1 for row in rows if row.get("inference_status") == "resolved")
        unresolved = sum(1 for row in rows if row.get("inference_status") == "unresolved")
        too_few = sum(1 for row in rows if row.get("inference_status") == "too_few_trades")
        no_trades = sum(1 for row in rows if row.get("inference_status") == "no_trades")
        errors = sum(1 for row in rows if row.get("inference_status") == "error")

        closed_times = [str(row["closed_time_utc"]) for row in rows if row.get("closed_time_utc")]
        min_closed_time = min(closed_times) if closed_times else None
        max_closed_time = max(closed_times) if closed_times else None

        conn.execute(
            """
            INSERT INTO inference_jobs (
                created_at_utc,
                data_dir,
                resolution_threshold,
                min_trades,
                min_usd_amount,
                total_markets,
                resolved_markets,
                unresolved_markets,
                too_few_trades_markets,
                no_trades_markets,
                error_markets,
                min_closed_time_utc,
                max_closed_time_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at_utc,
                str(data_dir),
                float(resolution_threshold),
                int(min_trades),
                _norm_min_usd(min_usd_amount),
                len(rows),
                resolved,
                unresolved,
                too_few,
                no_trades,
                errors,
                min_closed_time,
                max_closed_time,
            ),
        )
        job_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])

        conn.executemany(
            """
            INSERT OR REPLACE INTO inferred_market_resolutions (
                job_id,
                market_id,
                market_slug,
                closed_time_utc,
                volume,
                n_trades,
                inferred_winning_outcome,
                inference_status,
                latest_trade_ts_ms,
                updated_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    job_id,
                    int(row["market_id"]),
                    row.get("market_slug"),
                    row.get("closed_time_utc"),
                    row.get("volume"),
                    int(row.get("n_trades", 0) or 0),
                    (
                        int(row["inferred_winning_outcome"])
                        if row.get("inferred_winning_outcome") is not None
                        else None
                    ),
                    str(row.get("inference_status", "error")),
                    (
                        int(row["latest_trade_ts_ms"])
                        if row.get("latest_trade_ts_ms") is not None
                        else None
                    ),
                    created_at_utc,
                )
                for row in rows
            ],
        )
        conn.commit()
        return job_id
    except Exception:
        return None
    finally:
        conn.close()


def load_cached_inferred_resolutions(
    db_path: str,
    market_ids: List[int],
    resolution_threshold: float,
    min_trades: int,
    min_usd_amount: Optional[float],
) -> Tuple[Optional[int], Dict[int, int], Dict[str, int]]:
    """
    Load inferred winners from SQLite cache.
    """
    stats = {
        "total_markets": len(market_ids),
        "cached_rows": 0,
        "cached_resolved": 0,
        "cached_known_unresolved": 0,
        "skipped_missing": 0,
    }
    cached_rows = load_cached_resolution_rows(db_path=db_path, market_ids=market_ids)
    if not cached_rows:
        return None, {}, stats

    overrides: Dict[int, int] = {}
    for market_id, row in cached_rows.items():
        if row.get("inference_status") == "resolved" and row.get("inferred_winning_outcome") is not None:
            overrides[int(market_id)] = int(row["inferred_winning_outcome"])

    stats["cached_rows"] = len(cached_rows)
    stats["cached_resolved"] = len(overrides)
    stats["cached_known_unresolved"] = len(cached_rows) - len(overrides)
    stats["skipped_missing"] = max(0, len(market_ids) - len(cached_rows))
    return 1, overrides, stats
