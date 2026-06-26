"""
sql/queries.py — SemiSight SQL Analytics
8 SQL queries analyzing semiconductor yield patterns.
"""

import sqlite3
import pandas as pd
import numpy as np


def build_db(df: pd.DataFrame, db_path: str = ":memory:") -> sqlite3.Connection:
    """Build SQLite database from SECOM dataframe."""
    conn = sqlite3.connect(db_path)

    # Main runs table (metadata + label only — parameters too wide for SQL demo)
    meta_cols = ["timestamp", "process_step", "chamber_id", "lot_id", "yield_pass"] \
                if "process_step" in df.columns else ["yield_pass"]

    # Add synthetic metadata if not present
    n = len(df)
    meta = pd.DataFrame({
        "run_id":       range(n),
        "timestamp":    pd.date_range("2023-01-01", periods=n, freq="30min"),
        "process_step": np.random.choice(["DEP", "ETCH", "LITHO", "CMP", "INSP"], n,
                                          p=[0.25, 0.25, 0.20, 0.15, 0.15]),
        "chamber_id":   np.random.choice(["A", "B", "C", "D"], n),
        "lot_id":       [f"LOT{i//25:04d}" for i in range(n)],
        "yield_pass":   df["yield_pass"].values,
        "hour":         pd.date_range("2023-01-01", periods=n, freq="30min").hour,
        "day_of_week":  pd.date_range("2023-01-01", periods=n, freq="30min").dayofweek,
        "week":         pd.date_range("2023-01-01", periods=n, freq="30min").isocalendar().week.values,
    })
    meta.to_sql("runs", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_pass ON runs(yield_pass)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_step ON runs(process_step)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chamber ON runs(chamber_id)")
    conn.commit()
    return conn


def run_all_queries(conn: sqlite3.Connection) -> dict:
    """Run all 8 SQL analytics queries."""
    results = {}

    # Q1: Yield rate by process step
    try:
        results["yield_by_step"] = pd.read_sql_query("""
            SELECT
                process_step,
                COUNT(*) AS total_runs,
                SUM(CASE WHEN yield_pass = 1 THEN 1 ELSE 0 END) AS passes,
                SUM(CASE WHEN yield_pass = 0 THEN 1 ELSE 0 END) AS failures,
                ROUND(100.0 * SUM(CASE WHEN yield_pass = 1 THEN 1 ELSE 0 END) / COUNT(*), 2) AS yield_pct,
                ROUND(100.0 * SUM(CASE WHEN yield_pass = 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS failure_pct
            FROM runs
            GROUP BY process_step
            ORDER BY failure_pct DESC
        """, conn)
    except Exception as e:
        results["yield_by_step"] = pd.DataFrame()

    # Q2: Chamber performance comparison
    try:
        results["chamber_yield"] = pd.read_sql_query("""
            SELECT
                chamber_id,
                COUNT(*) AS total_runs,
                ROUND(100.0 * SUM(yield_pass) / COUNT(*), 2) AS yield_pct,
                ROUND(100.0 * (1 - SUM(yield_pass) * 1.0 / COUNT(*)), 2) AS failure_pct
            FROM runs
            GROUP BY chamber_id
            ORDER BY failure_pct DESC
        """, conn)
    except Exception as e:
        results["chamber_yield"] = pd.DataFrame()

    # Q3: Weekly yield trend
    try:
        results["weekly_trend"] = pd.read_sql_query("""
            SELECT
                week,
                COUNT(*) AS runs,
                ROUND(100.0 * SUM(yield_pass) / COUNT(*), 2) AS yield_pct,
                SUM(CASE WHEN yield_pass = 0 THEN 1 ELSE 0 END) AS failures
            FROM runs
            GROUP BY week
            ORDER BY week
        """, conn)
    except Exception as e:
        results["weekly_trend"] = pd.DataFrame()

    # Q4: Lot failure analysis — lots with highest failure rates
    try:
        results["lot_failures"] = pd.read_sql_query("""
            SELECT
                lot_id,
                COUNT(*) AS runs_in_lot,
                SUM(CASE WHEN yield_pass = 0 THEN 1 ELSE 0 END) AS failures,
                ROUND(100.0 * SUM(CASE WHEN yield_pass = 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS failure_pct
            FROM runs
            GROUP BY lot_id
            HAVING COUNT(*) >= 10
            ORDER BY failure_pct DESC
            LIMIT 10
        """, conn)
    except Exception as e:
        results["lot_failures"] = pd.DataFrame()

    # Q5: Time-of-day failure pattern
    try:
        results["hourly_pattern"] = pd.read_sql_query("""
            SELECT
                hour,
                COUNT(*) AS runs,
                ROUND(100.0 * SUM(CASE WHEN yield_pass = 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS failure_pct
            FROM runs
            GROUP BY hour
            ORDER BY hour
        """, conn)
    except Exception as e:
        results["hourly_pattern"] = pd.DataFrame()

    # Q6: Cross-tabulation: chamber × process step failure rates
    try:
        results["chamber_step_cross"] = pd.read_sql_query("""
            SELECT
                chamber_id,
                process_step,
                COUNT(*) AS runs,
                ROUND(100.0 * SUM(CASE WHEN yield_pass = 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS failure_pct
            FROM runs
            GROUP BY chamber_id, process_step
            ORDER BY failure_pct DESC
        """, conn)
    except Exception as e:
        results["chamber_step_cross"] = pd.DataFrame()

    # Q7: Rolling 7-day yield using window function
    try:
        results["rolling_yield"] = pd.read_sql_query("""
            SELECT
                week,
                day_of_week,
                COUNT(*) AS daily_runs,
                ROUND(100.0 * AVG(yield_pass), 2) AS daily_yield_pct,
                ROUND(100.0 * AVG(AVG(yield_pass)) OVER (
                    ORDER BY week, day_of_week
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ), 2) AS rolling_7day_yield
            FROM runs
            GROUP BY week, day_of_week
            ORDER BY week, day_of_week
        """, conn)
    except:
        results["rolling_yield"] = pd.DataFrame()

    # Q8: Overall summary statistics
    try:
        results["summary"] = pd.read_sql_query("""
            SELECT
                COUNT(*) AS total_runs,
                SUM(yield_pass) AS total_passes,
                SUM(CASE WHEN yield_pass = 0 THEN 1 ELSE 0 END) AS total_failures,
                ROUND(100.0 * AVG(yield_pass), 2) AS overall_yield_pct,
                COUNT(DISTINCT lot_id) AS unique_lots,
                COUNT(DISTINCT chamber_id) AS chambers,
                COUNT(DISTINCT process_step) AS process_steps
            FROM runs
        """, conn)
    except Exception as e:
        results["summary"] = pd.DataFrame()

    return results
