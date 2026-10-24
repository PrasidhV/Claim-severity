"""Load raw CSVs into SQLite, run the SQL cleaning pipeline, and write a data-quality report."""
import sqlite3

import pandas as pd

from common import DB, RAW, OUT, ROOT, is_synthetic

SQL = ROOT / "sql"


def run_script(con, path):
    con.executescript(path.read_text())


def run_checks(con, path):
    """03_checks.sql holds named queries ('-- name: x' headers); run each and format the result."""
    blocks = [b for b in path.read_text().split("-- name:") if b.strip()]
    lines = [f"DATA SOURCE: {'SYNTHETIC (pipeline test)' if is_synthetic() else 'real freMTPL2'}\n"]
    for b in blocks:
        name, _, query = b.partition("\n")
        df = pd.read_sql(query.strip().rstrip(";"), con)
        lines += [f"== {name.strip()} ==", df.to_string(index=False), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    DB.parent.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    if DB.exists():
        DB.unlink()
    with sqlite3.connect(DB) as con:
        run_script(con, SQL / "01_schema.sql")
        pd.read_csv(RAW / "freMTPL2freq.csv").to_sql("raw_freq", con, if_exists="append", index=False)
        pd.read_csv(RAW / "freMTPL2sev.csv").to_sql("raw_sev", con, if_exists="append", index=False)
        run_script(con, SQL / "02_clean.sql")
        run_script(con, SQL / "04_frequency.sql")
        report = run_checks(con, SQL / "03_checks.sql")
    (OUT / "dq_report.txt").write_text(report)
    print(report)
