"""Starlette routes for the owner admin dashboard (/admin)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any
from uuid import UUID

from psycopg import AsyncConnection
from psycopg.rows import DictRow
from psycopg_pool import AsyncConnectionPool
from starlette.requests import Request
from starlette.responses import HTMLResponse, RedirectResponse, Response
from starlette.routing import Route

from zos.admin_service import AdminService
from zos.auth.oauth import verify_owner_passphrase
from zos.auth.rate_limit import GlobalPassphraseRateLimiter, record_auth_failure
from zos.config import Settings
from zos.providers.upstream_manager import UpstreamManager
from zos.router import ProviderRegistry, SessionTracker
from zos.web.admin_auth import (
    ADMIN_SECURITY_HEADERS,
    AdminSession,
    AdminSessionManager,
    derive_session_signing_key,
)
from zos.web.admin_views import (
    render_activity_page,
    render_admin_login_page,
    render_clients_page,
    render_overview_page,
    render_policy_page,
    render_upstreams_page,
)


def _html(content: str, *, status_code: int = 200, extra_headers: dict[str, str] | None = None) -> HTMLResponse:
    headers = dict(ADMIN_SECURITY_HEADERS)
    if extra_headers:
        headers.update(extra_headers)
    return HTMLResponse(content, status_code=status_code, headers=headers)


def _redirect(location: str) -> RedirectResponse:
    return RedirectResponse(location, status_code=303, headers=dict(ADMIN_SECURITY_HEADERS))


def _parse_uuid_or_none(value: str | None) -> UUID | None:
    if not value or not value.strip():
        return None
    try:
        return UUID(value.strip())
    except ValueError:
        return None


def create_admin_routes(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    settings: Settings,
    registry: ProviderRegistry,
    sessions: SessionTracker,
    *,
    session_manager: AdminSessionManager | None = None,
    passphrase_limiter: GlobalPassphraseRateLimiter | None = None,
    switch_provider_backend: Callable[[str, str], Awaitable[Any]] | None = None,
    upstream_manager: UpstreamManager | None = None,
) -> list[Route]:
    """Create passphrase-gated, CSRF-protected /admin dashboard routes."""
    service = AdminService(pool)
    auth = session_manager or AdminSessionManager(
        signing_key=derive_session_signing_key(
            secret_key=settings.zos_secret_key,
            owner_password_hash=settings.zos_owner_password_hash,
        ),
        secure_cookie=settings.zos_base_url.lower().startswith("https://"),
    )
    limiter = passphrase_limiter or GlobalPassphraseRateLimiter()

    def _require_session(request: Request) -> AdminSession | None:
        return auth.read_session(request)

    async def _require_csrf(request: Request, session: AdminSession) -> dict[str, str] | None:
        form = await request.form()
        data = {k: str(v) for k, v in form.items()}
        if not auth.verify_csrf(session, data.get("csrf_token")):
            return None
        return data

    async def login(request: Request) -> Response:
        if request.method == "GET":
            if _require_session(request) is not None:
                return _redirect("/admin")
            return _html(render_admin_login_page())

        retry_after = limiter.retry_after_seconds()
        if retry_after is not None:
            return _html(
                render_admin_login_page(
                    error=f"Too many failed login attempts. Try again in {retry_after}s."
                ),
                status_code=429,
                extra_headers={"Retry-After": str(retry_after)},
            )

        if not settings.zos_owner_password_hash:
            return _html(
                render_admin_login_page(error="ZOS_OWNER_PASSWORD_HASH is not configured."),
                status_code=503,
            )

        form = await request.form()
        passphrase = str(form.get("passphrase", ""))
        if not verify_owner_passphrase(passphrase, settings.zos_owner_password_hash):
            limiter.record_failure()
            await record_auth_failure(pool, route="/admin/login", reason="invalid_passphrase")
            return _html(
                render_admin_login_page(error="Invalid owner passphrase."),
                status_code=401,
            )

        limiter.record_success()
        _sess, cookie_token = auth.create_session()
        resp = _redirect("/admin")
        auth.set_cookie(resp, cookie_token)
        return resp

    async def logout(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)
        resp = _redirect("/admin/login")
        auth.clear_cookie(resp)
        return resp

    async def overview_page(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        overview = await service.get_overview(registry, sessions)
        return _html(render_overview_page(overview=overview, csrf_token=session.csrf_token))

    async def provider_set_backend(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        slug = data.get("slug", "").strip()
        backend = data.get("backend", "").strip()
        notice = None
        error = None
        try:
            res = await service.set_provider_backend(
                slug=slug,
                backend=backend,
                on_switch=switch_provider_backend,
            )
            notice = f"Switched provider '{res['slug']}' backend to '{res['backend']}'."
        except Exception as exc:
            error = str(exc)

        overview = await service.get_overview(registry, sessions)
        return _html(
            render_overview_page(
                overview=overview,
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
            )
        )

    async def clients_page(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        clients = await service.list_clients()
        return _html(
            render_clients_page(
                clients=clients,
                csrf_token=session.csrf_token,
                max_oauth_clients=settings.max_oauth_clients,
            )
        )

    async def clients_create(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        name = data.get("name", "").strip()
        trust_ceiling = data.get("trust_ceiling", "owner").strip()
        error = None
        token_info = None
        notice = None
        try:
            res = await service.register_bearer_client(name=name, trust_ceiling=trust_ceiling)
            token_info = {"client_name": res["client"]["name"], "token": res["token"]}
            notice = f"Created bearer client '{res['client']['name']}'."
        except Exception as exc:
            error = str(exc)

        clients = await service.list_clients()
        return _html(
            render_clients_page(
                clients=clients,
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
                new_token_info=token_info,
                max_oauth_clients=settings.max_oauth_clients,
            )
        )

    async def clients_new_token(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        client_name = data.get("client_name", "").strip()
        error = None
        token_info = None
        notice = None
        try:
            res = await service.create_bearer_token(client_name=client_name)
            token_info = {"client_name": res["client"]["name"], "token": res["token"]}
            notice = f"Minted new bearer token for '{res['client']['name']}'."
        except Exception as exc:
            error = str(exc)

        clients = await service.list_clients()
        return _html(
            render_clients_page(
                clients=clients,
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
                new_token_info=token_info,
                max_oauth_clients=settings.max_oauth_clients,
            )
        )

    async def clients_revoke(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        client_id = data.get("client_id", "").strip()
        error = None
        notice = None
        try:
            count = await service.revoke_tokens(client_identifier=client_id)
            notice = f"Revoked {count} active token(s)."
        except Exception as exc:
            error = str(exc)

        clients = await service.list_clients()
        return _html(
            render_clients_page(
                clients=clients,
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
                max_oauth_clients=settings.max_oauth_clients,
            )
        )

    async def clients_delete(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        client_id = data.get("client_id", "").strip()
        error = None
        notice = None
        try:
            deleted = await service.delete_oauth_client(client_identifier=client_id)
            notice = f"Pruned OAuth client '{deleted['name']}'."
        except Exception as exc:
            error = str(exc)

        clients = await service.list_clients()
        return _html(
            render_clients_page(
                clients=clients,
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
                max_oauth_clients=settings.max_oauth_clients,
            )
        )

    async def policy_page(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        matrix = await service.get_policy_matrix(registry)
        return _html(render_policy_page(matrix=matrix, csrf_token=session.csrf_token))

    async def policy_set(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        pattern = data.get("pattern", "").strip()
        decision = data.get("decision", "").strip()
        client_id = data.get("client_id", "").strip() or None
        error = None
        notice = None
        try:
            row = await service.set_policy(
                pattern=pattern,
                decision=decision,
                client_identifier=client_id,
            )
            notice = f"Policy set: {row['tool_pattern']} -> {row['decision']}."
        except Exception as exc:
            error = str(exc)

        matrix = await service.get_policy_matrix(registry)
        return _html(
            render_policy_page(
                matrix=matrix,
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
            )
        )

    async def policy_clear(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        pattern = data.get("pattern", "").strip()
        client_id = data.get("client_id", "").strip() or None
        error = None
        notice = None
        try:
            deleted = await service.clear_policy(
                pattern=pattern,
                client_identifier=client_id,
            )
            notice = f"Cleared override for '{pattern}'." if deleted else f"No override found for '{pattern}'."
        except Exception as exc:
            error = str(exc)

        matrix = await service.get_policy_matrix(registry)
        return _html(
            render_policy_page(
                matrix=matrix,
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
            )
        )

    async def policy_set_tier(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        pattern = data.get("pattern", "").strip()
        tier = data.get("tier", "").strip()
        error = None
        notice = None
        try:
            row = await service.set_tool_tier(pattern=pattern, tier=tier)
            notice = f"Tier set: {row['tool_pattern']} -> {row['tier']}."
        except Exception as exc:
            error = str(exc)

        matrix = await service.get_policy_matrix(registry)
        return _html(
            render_policy_page(
                matrix=matrix,
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
            )
        )

    async def upstreams_page(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        upstreams = await service.list_remote_upstreams(upstream_manager=upstream_manager)
        return _html(render_upstreams_page(upstreams=upstreams, csrf_token=session.csrf_token))

    async def upstreams_add(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        slug = data.get("slug", "").strip()
        url = data.get("url", "").strip()
        auth_mode = data.get("auth_mode", "none").strip()
        default_tier = data.get("default_tier", "external").strip()
        notice = None
        error = None
        try:
            row = await service.add_remote_upstream(
                slug=slug,
                url=url,
                auth_mode=auth_mode,
                default_tier=default_tier,
                upstream_manager=upstream_manager,
            )
            notice = f"Added MCP server '{row['slug']}'."
        except Exception as exc:
            error = str(exc)

        upstreams = await service.list_remote_upstreams(upstream_manager=upstream_manager)
        return _html(
            render_upstreams_page(
                upstreams=upstreams,
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
            )
        )

    async def upstreams_set_token(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        slug = data.get("slug", "").strip()
        token = data.get("token", "")
        notice = None
        error = None
        try:
            await service.set_upstream_bearer_token(
                slug=slug,
                token=token,
                secret_key=settings.zos_secret_key,
                upstream_manager=upstream_manager,
            )
            notice = f"Updated bearer token for '{slug}'."
        except Exception as exc:
            error = str(exc)

        upstreams = await service.list_remote_upstreams(upstream_manager=upstream_manager)
        return _html(
            render_upstreams_page(
                upstreams=upstreams,
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
            )
        )

    async def upstreams_disconnect(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        slug = data.get("slug", "").strip()
        notice = None
        error = None
        try:
            await service.disconnect_remote_upstream(
                slug=slug,
                upstream_manager=upstream_manager,
            )
            notice = f"Disconnected '{slug}'."
        except Exception as exc:
            error = str(exc)

        upstreams = await service.list_remote_upstreams(upstream_manager=upstream_manager)
        return _html(
            render_upstreams_page(
                upstreams=upstreams,
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
            )
        )

    async def upstreams_remove(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        slug = data.get("slug", "").strip()
        notice = None
        error = None
        try:
            await service.remove_remote_upstream(
                slug=slug,
                upstream_manager=upstream_manager,
            )
            notice = f"Removed MCP server '{slug}'."
        except Exception as exc:
            error = str(exc)

        upstreams = await service.list_remote_upstreams(upstream_manager=upstream_manager)
        return _html(
            render_upstreams_page(
                upstreams=upstreams,
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
            )
        )

    async def activity_page(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")

        qp = request.query_params
        before_raw = qp.get("before_id", "").strip()
        before_id = int(before_raw) if before_raw.isdigit() else None
        limit_raw = qp.get("limit", "").strip()
        limit = int(limit_raw) if limit_raw.isdigit() else 50
        client_id_str = qp.get("client_id", "").strip()
        provider_id_str = qp.get("provider_id", "").strip()
        event_type = qp.get("type", "").strip()
        outcome = qp.get("outcome", "").strip()

        data = await service.list_events(
            before_id=before_id,
            client_id=_parse_uuid_or_none(client_id_str),
            provider_id=_parse_uuid_or_none(provider_id_str),
            event_type=event_type or None,
            outcome=outcome if outcome in ("ok", "error", "denied") else None,
            limit=limit,
        )
        filters = {
            "client_id": client_id_str,
            "provider_id": provider_id_str,
            "type": event_type,
            "outcome": outcome,
        }
        return _html(render_activity_page(data=data, filters=filters, csrf_token=session.csrf_token))

    return [
        Route("/admin", overview_page, methods=["GET"]),
        Route("/admin/providers/backend", provider_set_backend, methods=["POST"]),
        Route("/admin/login", login, methods=["GET", "POST"]),
        Route("/admin/logout", logout, methods=["POST"]),
        Route("/admin/clients", clients_page, methods=["GET"]),
        Route("/admin/clients/create", clients_create, methods=["POST"]),
        Route("/admin/clients/token", clients_new_token, methods=["POST"]),
        Route("/admin/clients/revoke", clients_revoke, methods=["POST"]),
        Route("/admin/clients/delete", clients_delete, methods=["POST"]),
        Route("/admin/policy", policy_page, methods=["GET"]),
        Route("/admin/policy/set", policy_set, methods=["POST"]),
        Route("/admin/policy/clear", policy_clear, methods=["POST"]),
        Route("/admin/policy/tier", policy_set_tier, methods=["POST"]),
        Route("/admin/upstreams", upstreams_page, methods=["GET"]),
        Route("/admin/upstreams/add", upstreams_add, methods=["POST"]),
        Route("/admin/upstreams/token", upstreams_set_token, methods=["POST"]),
        Route("/admin/upstreams/disconnect", upstreams_disconnect, methods=["POST"]),
        Route("/admin/upstreams/remove", upstreams_remove, methods=["POST"]),
        Route("/admin/activity", activity_page, methods=["GET"]),
    ]
