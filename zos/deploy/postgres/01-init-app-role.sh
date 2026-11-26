#!/usr/bin/env bash
# Postgres init script: create a restricted non-superuser core role (CREATEROLE + DB/schema owner)
# and pre-install pg_trgm so ZOS runs cleanly without Postgres superuser privileges.
set -euo pipefail

APP_USER="${ZOS_DB_USER:-${POSTGRES_USER}}"
APP_PASSWORD="${ZOS_DB_PASSWORD:-${POSTGRES_PASSWORD}}"
DB_NAME="${POSTGRES_DB}"

if [ "${APP_USER}" = "${POSTGRES_USER}" ]; then
  psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${DB_NAME}" <<-EOSQL
    CREATE SCHEMA IF NOT EXISTS extensions;
    REVOKE ALL ON SCHEMA extensions FROM PUBLIC;
    CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA extensions;
    ALTER ROLE "${POSTGRES_USER}" SET search_path TO public, extensions;
EOSQL
  exit 0
fi

psql -v ON_ERROR_STOP=1 --username "${POSTGRES_USER}" --dbname "${DB_NAME}" \
  -v app_user="${APP_USER}" \
  -v app_password="${APP_PASSWORD}" \
  -v db_name="${DB_NAME}" <<-'EOSQL'
  SELECT format(
    'CREATE ROLE %I WITH LOGIN PASSWORD %L CREATEROLE NOSUPERUSER NOCREATEDB NOREPLICATION',
    :'app_user',
    :'app_password'
  )
  WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'app_user')
  \gexec

  SELECT format('ALTER DATABASE %I OWNER TO %I', :'db_name', :'app_user') \gexec
  SELECT format('ALTER SCHEMA public OWNER TO %I', :'app_user') \gexec
  REVOKE CREATE ON SCHEMA public FROM PUBLIC;
  SELECT format('SET ROLE %I', :'app_user') \gexec
  CREATE SCHEMA IF NOT EXISTS extensions;
  REVOKE ALL ON SCHEMA extensions FROM PUBLIC;
  CREATE EXTENSION IF NOT EXISTS pg_trgm SCHEMA extensions;
  RESET ROLE;
  SELECT format('ALTER ROLE %I SET search_path TO public, extensions', :'app_user') \gexec
EOSQL
