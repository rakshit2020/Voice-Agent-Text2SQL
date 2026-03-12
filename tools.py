"""
tools.py — All tool definitions, handlers, and the database executor.
             Add any new tools to TOOL_REGISTRY at the bottom.
"""

import os
from typing import Any, Dict

from dotenv import load_dotenv
from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

load_dotenv(override=True)

# ╔══════════════════════════════════════════════════════════════╗
# ║                   DATABASE EXECUTOR                          ║
# ╚══════════════════════════════════════════════════════════════╝

_engine: AsyncEngine | None = None


def _get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(
            os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/mydb"),
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            echo=False,
        )
        logger.info("Database engine created")
    return _engine


async def execute_readonly_query(sql: str) -> Dict[str, Any]:
    """Execute a read-only SQL query. Returns dict with success, columns, rows, error."""

    stripped = sql.strip().lower()

    # ── Safety: only SELECT allowed ──
    if not stripped.startswith("select"):
        return {
            "success": False, "columns": [], "rows": [],
            "row_count": 0, "error": "Only SELECT queries are allowed.",
        }

    forbidden = ["insert", "update", "delete", "drop", "alter", "truncate",
                  "create", "grant", "revoke", "exec"]
    for kw in forbidden:
        # Check outside of string literals (rough but effective)
        if kw in stripped.split("'")[0]:
            return {
                "success": False, "columns": [], "rows": [],
                "row_count": 0, "error": f"Forbidden keyword: {kw.upper()}",
            }

    engine = _get_engine()
    try:
        async with engine.connect() as conn:
            result = await conn.execute(text(sql))
            columns = list(result.keys())
            rows = [list(row) for row in result.fetchall()]
            logger.info(f"Query returned {len(rows)} rows")
            return {
                "success": True,
                "columns": columns,
                "rows": rows[:50],   # hard cap
                "row_count": len(rows),
                "error": None,
            }
    except Exception as e:
        logger.error(f"DB error: {e}")
        return {
            "success": False, "columns": [], "rows": [],
            "row_count": 0, "error": str(e),
        }


async def shutdown_db():
    """Gracefully dispose the DB engine."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        logger.info("Database engine disposed")


# ╔══════════════════════════════════════════════════════════════╗
# ║               TOOL 1: query_database                         ║
# ╚══════════════════════════════════════════════════════════════╝

QUERY_DATABASE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "query_database",
        "description": (
            "Convert the user's natural-language data question into a valid "
            "PostgreSQL SELECT query, execute it, and return results. "
            "Use the database schema in the system prompt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql_query": {
                    "type": "string",
                    "description": (
                        "A valid PostgreSQL SELECT query. Must be read-only. "
                        "Use table aliases. Limit to 20 rows unless user asks for more."
                    ),
                },
                "explanation": {
                    "type": "string",
                    "description": "Brief one-line explanation of what this query does.",
                },
            },
            "required": ["sql_query"],
        },
    },
}


async def handle_query_database(params):
    """
    Pipecat calls this when the LLM emits a `query_database` tool call.

    params.arguments  → dict with sql_query, explanation
    params.result_callback → async callable to return result to LLM
    """
    args = params.arguments
    sql = args.get("sql_query", "")
    explanation = args.get("explanation", "")

    logger.info(f"[query_database] SQL: {sql}")
    if explanation:
        logger.info(f"[query_database] Why: {explanation}")

    result = await execute_readonly_query(sql)

    if result["success"]:
        if result["row_count"] == 0:
            result_text = "The query ran successfully but returned no results."
        else:
            columns = result["columns"]
            rows = result["rows"]
            header = " | ".join(str(c) for c in columns)
            divider = "-" * len(header)
            row_lines = [" | ".join(str(v) for v in row) for row in rows[:20]]

            result_text = (
                f"Query returned {result['row_count']} row(s).\n"
                f"{header}\n{divider}\n" + "\n".join(row_lines)
            )
            if result["row_count"] > 20:
                result_text += f"\n... and {result['row_count'] - 20} more rows."
    else:
        result_text = f"Query failed: {result['error']}"

    logger.info(f"[query_database] Result length: {len(result_text)} chars")
    await params.result_callback(result_text)


# ╔══════════════════════════════════════════════════════════════╗
# ║    TOOL REGISTRY — add new tools here as (schema, name, fn) ║
# ╚══════════════════════════════════════════════════════════════╝

TOOL_REGISTRY = [
    (QUERY_DATABASE_SCHEMA, "query_database", handle_query_database),

    # ── Example: add another tool ──
    # (WEATHER_SCHEMA, "get_weather", handle_get_weather),
]