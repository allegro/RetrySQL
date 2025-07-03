import sqlite3
from typing import Any
from func_timeout import FunctionTimedOut

from upath import UPath

from text2sql.commons.logging_utils import get_logger

logger = get_logger(__name__)


def execute_sql(db_path: UPath, sql: str, fetch: str = "all") -> Any:
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)

        match fetch:
            case "all":
                return cursor.fetchall()
            case "one":
                return cursor.fetchone()
            case _:
                raise ValueError("Invalid fetch type. Supported are 'all' or 'one'.")

def is_the_generated_sql_executable(
        generated_sql: str,
        database_path: UPath,
) -> bool:
    if generated_sql.strip() == "":
        return False
    try:
        # use `EXPLAIN QUERY PLAN` to avoid actually executing the query
        # https://www.sqlite.org/eqp.html
        execute_sql(
            db_path=database_path,
            sql="EXPLAIN QUERY PLAN " + generated_sql
        )
        return True
    except FunctionTimedOut as fto:
        logger.info("SQL execution time out error: {}.".format(fto))
        return False
    except Exception as e:
        logger.info("SQL execution runtime error: {}.".format(e))
        return False