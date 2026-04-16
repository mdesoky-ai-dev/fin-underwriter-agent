"""
db/checkpointer.py
"""

import os
import contextlib
import structlog
import psycopg
from dotenv import load_dotenv
from langgraph.checkpoint.postgres import PostgresSaver

load_dotenv()

log = structlog.get_logger()


def get_connection_string() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL not set in .env")
    return url


@contextlib.contextmanager
def get_checkpointer():

    conn_string = get_connection_string()
    log.info("checkpointer.connecting")

    with PostgresSaver.from_conn_string(conn_string) as checkpointer:
        checkpointer.setup()
        log.info("checkpointer.ready")
        yield checkpointer


def test_connection() -> bool:
    try:
        with get_checkpointer() as checkpointer:
            log.info("checkpointer.test_passed")
            return True
    except Exception as e:
        log.error("checkpointer.test_failed", error=str(e))
        return False

#KEY CONNECTION
def get_prod_checkpointer():
    conn = psycopg.connect(os.getenv("DATABASE_URL"))
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    return checkpointer

if __name__ == "__main__":
    print("Testing PostgreSQL connection...")
    success = test_connection()
    if success:
        print("✓ PostgreSQL checkpointer connected and tables created")
    else:
        print("✗ Connection failed — check your DATABASE_URL in .env")