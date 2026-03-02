#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import psycopg
from psycopg import sql


@dataclass(frozen=True)
class TableStats:
    row_count: int
    xor_hash: int | None
    sum_hash: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare table-level parity between source and target Postgres databases."
    )
    parser.add_argument("--source-url", required=True, help="Source DATABASE_URL")
    parser.add_argument("--target-url", required=True, help="Target DATABASE_URL")
    parser.add_argument("--schema", default="public", help="Schema to compare (default: public)")
    parser.add_argument(
        "--exclude-table",
        action="append",
        default=["alembic_version"],
        help="Table name to exclude (repeatable). Default includes alembic_version.",
    )
    parser.add_argument(
        "--counts-only",
        action="store_true",
        help="Only compare row counts (skip checksums).",
    )
    return parser.parse_args()


def fetch_tables(conn: psycopg.Connection, schema: str, excluded: set[str]) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """,
            (schema,),
        )
        all_tables = [row[0] for row in cur.fetchall()]
    return [table for table in all_tables if table not in excluded]


def fetch_stats(
    conn: psycopg.Connection,
    schema: str,
    table: str,
    counts_only: bool,
) -> TableStats:
    with conn.cursor() as cur:
        if counts_only:
            query = sql.SQL("SELECT count(*)::bigint FROM {}.{}").format(
                sql.Identifier(schema),
                sql.Identifier(table),
            )
            cur.execute(query)
            row_count = int(cur.fetchone()[0])
            return TableStats(row_count=row_count, xor_hash=None, sum_hash=None)

        query = sql.SQL(
            """
            SELECT
              count(*)::bigint AS row_count,
              COALESCE(bit_xor(hashtextextended(to_jsonb(t)::text, 0)), 0)::bigint AS xor_hash,
              COALESCE(sum(hashtextextended(to_jsonb(t)::text, 0)::numeric), 0)::text AS sum_hash
            FROM {}.{} AS t
            """
        ).format(sql.Identifier(schema), sql.Identifier(table))
        cur.execute(query)
        row_count, xor_hash, sum_hash = cur.fetchone()
        return TableStats(
            row_count=int(row_count),
            xor_hash=int(xor_hash),
            sum_hash=str(sum_hash),
        )


def main() -> int:
    args = parse_args()
    excluded = {name.strip() for name in args.exclude_table if name.strip()}

    with psycopg.connect(args.source_url) as source_conn, psycopg.connect(args.target_url) as target_conn:
        source_tables = fetch_tables(source_conn, args.schema, excluded)
        target_tables = fetch_tables(target_conn, args.schema, excluded)

        source_set = set(source_tables)
        target_set = set(target_tables)
        only_in_source = sorted(source_set - target_set)
        only_in_target = sorted(target_set - source_set)
        common_tables = sorted(source_set & target_set)

        has_mismatch = False

        if only_in_source:
            has_mismatch = True
            print("Tables missing in target:")
            for table in only_in_source:
                print(f"  - {table}")

        if only_in_target:
            has_mismatch = True
            print("Tables missing in source:")
            for table in only_in_target:
                print(f"  - {table}")

        if common_tables:
            if args.counts_only:
                print("\nTable parity (counts-only):")
                print("table,row_count_source,row_count_target,status")
            else:
                print("\nTable parity (count + checksums):")
                print("table,row_count_source,row_count_target,xor_source,xor_target,sum_source,sum_target,status")

        for table in common_tables:
            source_stats = fetch_stats(source_conn, args.schema, table, args.counts_only)
            target_stats = fetch_stats(target_conn, args.schema, table, args.counts_only)

            same_count = source_stats.row_count == target_stats.row_count
            same_hashes = (
                args.counts_only
                or (
                    source_stats.xor_hash == target_stats.xor_hash
                    and source_stats.sum_hash == target_stats.sum_hash
                )
            )
            status = "OK" if (same_count and same_hashes) else "MISMATCH"

            if status != "OK":
                has_mismatch = True

            if args.counts_only:
                print(
                    f"{table},{source_stats.row_count},{target_stats.row_count},{status}"
                )
            else:
                print(
                    f"{table},"
                    f"{source_stats.row_count},{target_stats.row_count},"
                    f"{source_stats.xor_hash},{target_stats.xor_hash},"
                    f"{source_stats.sum_hash},{target_stats.sum_hash},"
                    f"{status}"
                )

        if has_mismatch:
            print("\nParity check failed.")
            return 1

        print("\nParity check passed.")
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        raise SystemExit(130)
