"""Starlette routes for the owner admin dashboard (/admin)."""

from __future__ import annotations

import html
from typing import Any
from urllib.parse import quote
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
from zos.catalog import load_catalog, load_module_catalog
from zos.config import Settings
from zos.modules import ModuleManager
from zos.policy import ApprovalError, ApprovalManager
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
    render_approval_detail_page,
    render_approvals_page,
    render_clients_page,
    render_modules_page,
    render_overview_page,
    render_policy_page,
    render_upstreams_page,
)
from zos.web.module_proxy import proxy_module_ui_request


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


def _safe_next_url(raw: str | None) -> str:
    if not raw:
        return "/admin"
    candidate = raw.strip()
    if (
        candidate.startswith("/admin")
        and not candidate.startswith("//")
        and "\r" not in candidate
        and "\n" not in candidate
    ):
        return candidate
    return "/admin"


def create_admin_routes(
    pool: AsyncConnectionPool[AsyncConnection[DictRow]],
    settings: Settings,
    registry: ProviderRegistry,
    sessions: SessionTracker,
    *,
    session_manager: AdminSessionManager | None = None,
    passphrase_limiter: GlobalPassphraseRateLimiter | None = None,
    upstream_manager: UpstreamManager | None = None,
    approval_manager: ApprovalManager | None = None,
    module_manager: ModuleManager | None = None,
) -> list[Route]:
    """Create passphrase-gated, CSRF-protected /admin dashboard routes."""
    service = AdminService(pool)
    modules = module_manager or ModuleManager(
        pool,
        settings,
        upstream_manager=upstream_manager,
    )
    approvals = approval_manager or ApprovalManager(
        pool,
        secret_key=settings.zos_secret_key,
        base_url=settings.zos_base_url,
        wait_seconds=settings.approval_wait_seconds,
        ttl_seconds=settings.approval_ttl_seconds,
    )
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
            next_url = _safe_next_url(request.query_params.get("next"))
            if _require_session(request) is not None:
                return _redirect(next_url)
            return _html(render_admin_login_page(next_url=next_url if next_url != "/admin" else None))

        form = await request.form()
        next_url = _safe_next_url(str(form.get("next", "")))
        retry_after = limiter.retry_after_seconds()
        if retry_after is not None:
            return _html(
                render_admin_login_page(
                    error=f"Too many failed login attempts. Try again in {retry_after}s.",
                    next_url=next_url if next_url != "/admin" else None,
                ),
                status_code=429,
                extra_headers={"Retry-After": str(retry_after)},
            )

        if not settings.zos_owner_password_hash:
            return _html(
                render_admin_login_page(
                    error="ZOS_OWNER_PASSWORD_HASH is not configured.",
                    next_url=next_url if next_url != "/admin" else None,
                ),
                status_code=503,
            )

        passphrase = str(form.get("passphrase", ""))
        if not verify_owner_passphrase(passphrase, settings.zos_owner_password_hash):
            limiter.record_failure()
            await record_auth_failure(pool, route="/admin/login", reason="invalid_passphrase")
            return _html(
                render_admin_login_page(
                    error="Invalid owner passphrase.",
                    next_url=next_url if next_url != "/admin" else None,
                ),
                status_code=401,
            )

        limiter.record_success()
        _sess, cookie_token = auth.create_session()
        resp = _redirect(next_url)
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

    async def _ui_modules() -> list[dict[str, Any]]:
        try:
            all_mods = await modules.list_modules()
        except Exception:
            return []
        return [
            {
                "slug": str(m["slug"]),
                "title": str(m.get("ui_title") or m["slug"]),
                "href": f"/admin/m/{m['slug']}/",
                "online": bool(m.get("online", False)),
            }
            for m in all_mods
            if m.get("ui_url") and m.get("disabled_at") is None and m.get("status") != "disabled"
        ]

    async def overview_page(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        await approvals.sweep_expired()
        overview = await service.get_overview(registry, sessions)
        ui_mods = await _ui_modules()
        return _html(
            render_overview_page(
                overview=overview,
                csrf_token=session.csrf_token,
                ui_modules=ui_mods,
            )
        )

    async def _render_clients(
        session: AdminSession,
        *,
        notice: str | None = None,
        error: str | None = None,
        new_token_info: dict[str, str] | None = None,
    ) -> Response:
        clients = await service.list_clients()
        allowed_hosts = await service.list_allowed_redirect_hosts()
        rejected_attempts = await service.list_dcr_rejected_attempts()
        return _html(
            render_clients_page(
                clients=clients,
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
                new_token_info=new_token_info,
                max_oauth_clients=settings.max_oauth_clients,
                allowed_redirect_hosts=allowed_hosts,
                rejected_attempts=rejected_attempts,
            )
        )

    async def clients_page(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        return await _render_clients(session)

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

        return await _render_clients(session, notice=notice, error=error, new_token_info=token_info)

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

        return await _render_clients(session, notice=notice, error=error, new_token_info=token_info)

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

        return await _render_clients(session, notice=notice, error=error)

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

        return await _render_clients(session, notice=notice, error=error)

    async def clients_disable(request: Request) -> Response:
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
            disabled = await service.disable_client(client_identifier=client_id)
            notice = f"Disabled client '{disabled['name']}' (audit & memory history preserved)."
        except Exception as exc:
            error = str(exc)

        return await _render_clients(session, notice=notice, error=error)

    async def clients_redirect_host_allow(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        host = data.get("host", "").strip()
        error = None
        notice = None
        try:
            clean = await service.allow_redirect_host(host=host)
            notice = f"Allowed redirect host '{clean}'."
        except Exception as exc:
            error = str(exc)
        return await _render_clients(session, notice=notice, error=error)

    async def clients_redirect_host_dismiss(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        error = None
        notice = None
        try:
            attempt_id = int(data.get("attempt_id", "").strip())
            dismissed = await service.dismiss_dcr_rejected_attempt(attempt_id=attempt_id)
            notice = f"Dismissed blocked attempt from '{dismissed['host']}'."
        except Exception as exc:
            error = str(exc)
        return await _render_clients(session, notice=notice, error=error)

    async def clients_redirect_host_remove(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        host = data.get("host", "").strip()
        error = None
        notice = None
        try:
            removed = await service.remove_redirect_host(host=host)
            notice = f"Removed redirect host '{host}'." if removed else f"Host '{host}' was not listed."
        except Exception as exc:
            error = str(exc)
        return await _render_clients(session, notice=notice, error=error)

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

    def _catalog_entries() -> list[dict[str, Any]]:
        return [
            {
                "slug": m.slug,
                "name": m.name,
                "description": m.description,
                "docs_url": m.docs_url,
                "url": m.runtime.url,
                "auth_mode": m.auth.mode,
                "requires_client_registration": m.auth.requires_client_registration,
                "default_tier": m.tiers.default,
            }
            for m in load_catalog(settings.catalog_dir)
        ]

    async def upstreams_page(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        upstreams = await service.list_remote_upstreams(upstream_manager=upstream_manager)
        return _html(
            render_upstreams_page(
                upstreams=upstreams,
                catalog=_catalog_entries(),
                csrf_token=session.csrf_token,
            )
        )

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
                catalog=_catalog_entries(),
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
            )
        )

    async def upstreams_install(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        catalog_slug = data.get("catalog_slug", "").strip()
        as_slug = data.get("as_slug", "").strip() or None
        notice = None
        error = None
        try:
            res = await service.install_catalog_connector(
                catalog_slug=catalog_slug,
                as_slug=as_slug,
                catalog_dir=settings.catalog_dir,
                upstream_manager=upstream_manager,
            )
            manifest = res["manifest"]
            target = res["provider"]["slug"]
            if manifest.auth.mode == "oauth" and not manifest.auth.requires_client_registration:
                return _redirect(f"/connect/{target}")
            if manifest.auth.mode == "bearer":
                notice = f"Installed '{manifest.name}' as '{target}'. Paste your bearer token below to connect."
            elif manifest.auth.requires_client_registration:
                notice = (
                    f"Installed '{manifest.name}' as '{target}'. "
                    "Enter your pre-registered OAuth Client ID & Secret below, then click Connect."
                )
            else:
                notice = f"Installed '{manifest.name}' as '{target}'."
        except Exception as exc:
            error = str(exc)

        upstreams = await service.list_remote_upstreams(upstream_manager=upstream_manager)
        return _html(
            render_upstreams_page(
                upstreams=upstreams,
                catalog=_catalog_entries(),
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
                catalog=_catalog_entries(),
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
            )
        )

    async def upstreams_set_oauth_client(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        slug = data.get("slug", "").strip()
        client_id = data.get("client_id", "").strip()
        client_secret = data.get("client_secret", "")
        notice = None
        error = None
        try:
            await service.set_upstream_oauth_client(
                slug=slug,
                client_id=client_id,
                client_secret=client_secret,
                secret_key=settings.zos_secret_key,
                base_url=settings.zos_base_url,
                upstream_manager=upstream_manager,
            )
            notice = f"Saved OAuth client credentials for '{slug}'. Click Connect to authorize."
        except Exception as exc:
            error = str(exc)

        upstreams = await service.list_remote_upstreams(upstream_manager=upstream_manager)
        return _html(
            render_upstreams_page(
                upstreams=upstreams,
                catalog=_catalog_entries(),
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
                catalog=_catalog_entries(),
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
                catalog=_catalog_entries(),
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

    async def approvals_page(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect(f"/admin/login?next={quote(request.url.path, safe='/')}")
        data = await approvals.list_approvals()
        return _html(render_approvals_page(approvals=data, csrf_token=session.csrf_token))

    async def approval_detail_page(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect(f"/admin/login?next={quote(request.url.path, safe='/')}")
        approval_id = str(request.path_params.get("approval_id", "")).strip()
        item = await approvals.get_approval(approval_id)
        pending_count = await approvals.count_pending()
        status_code = 200 if item is not None else 404
        return _html(
            render_approval_detail_page(
                approval=item,
                csrf_token=session.csrf_token,
                pending_approvals=pending_count,
            ),
            status_code=status_code,
        )

    async def _approval_decide(request: Request, *, approve: bool) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        approval_id = str(request.path_params.get("approval_id", "")).strip()
        notice = None
        error = None
        try:
            decided = await approvals.decide(approval_id, approve=approve)
            if approve:
                notice = f"Approved '{decided['tool']}'. The tool call will execute now (or on retry within the TTL)."
            else:
                notice = f"Denied '{decided['tool']}'."
        except ApprovalError as exc:
            error = str(exc)

        item = await approvals.get_approval(approval_id)
        pending_count = await approvals.count_pending()
        status_code = 200 if item is not None else 404
        return _html(
            render_approval_detail_page(
                approval=item,
                csrf_token=session.csrf_token,
                pending_approvals=pending_count,
                notice=notice,
                error=error,
            ),
            status_code=status_code,
        )

    async def approval_approve(request: Request) -> Response:
        return await _approval_decide(request, approve=True)

    async def approval_deny(request: Request) -> Response:
        return await _approval_decide(request, approve=False)

    def _module_catalog_entries():
        return load_module_catalog(settings.modules_catalog_dir)

    async def _render_modules(
        session: AdminSession,
        *,
        notice: str | None = None,
        error: str | None = None,
        instructions: str | None = None,
    ) -> Response:
        installed = await modules.list_modules()
        ui_mods = [
            {
                "slug": str(m["slug"]),
                "title": str(m.get("ui_title") or m["slug"]),
                "href": f"/admin/m/{m['slug']}/",
                "online": bool(m.get("online", False)),
            }
            for m in installed
            if m.get("ui_url") and m.get("disabled_at") is None and m.get("status") != "disabled"
        ]
        return _html(
            render_modules_page(
                installed=installed,
                catalog=_module_catalog_entries(),
                host_kind=modules.host.kind,
                csrf_token=session.csrf_token,
                notice=notice,
                error=error,
                instructions=instructions,
                ui_modules=ui_mods,
            )
        )

    async def modules_page(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect(f"/admin/login?next={quote(request.url.path, safe='/')}")
        return await _render_modules(session)

    async def modules_install(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)

        slug = data.get("slug", "").strip().lower()
        cfg_values: dict[str, str] = {}
        for k, v in data.items():
            if k.startswith("cfg_"):
                cfg_values[k[4:]] = v

        notice = None
        error = None
        instructions = None
        try:
            res = await modules.install(slug, config_values=cfg_values)
            mod = res["module"]
            instructions = res.get("instructions")
            if mod["status"] == "pending_attach":
                notice = f"Prepared manual module '{slug}'. Follow the instructions below to start and attach it."
            else:
                notice = (
                    f"Installed module '{slug}' (v{mod['version']}). "
                    "New tools appear in clients after they refresh/reconnect their ZOS connector."
                )
        except Exception as exc:
            error = str(exc)

        return await _render_modules(session, notice=notice, error=error, instructions=instructions)

    async def modules_upgrade(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)
        slug = data.get("slug", "").strip().lower()
        notice = None
        error = None
        try:
            mod = await modules.upgrade(slug)
            notice = (
                f"Upgraded module '{slug}' (v{mod['version']}). "
                "New tools appear in clients after they refresh/reconnect their ZOS connector."
            )
        except Exception as exc:
            error = str(exc)
        return await _render_modules(session, notice=notice, error=error)

    async def modules_disable(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)
        slug = data.get("slug", "").strip().lower()
        notice = None
        error = None
        try:
            await modules.disable(slug)
            notice = f"Disabled module '{slug}'. Its tools are hidden and its data is preserved."
        except Exception as exc:
            error = str(exc)
        return await _render_modules(session, notice=notice, error=error)

    async def modules_enable(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)
        slug = data.get("slug", "").strip().lower()
        notice = None
        error = None
        try:
            await modules.enable(slug)
            notice = f"Enabled module '{slug}'."
        except Exception as exc:
            error = str(exc)
        return await _render_modules(session, notice=notice, error=error)

    async def modules_uninstall(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)
        slug = data.get("slug", "").strip().lower()
        purge = data.get("purge", "").strip().lower() in ("1", "true", "yes", "on")
        notice = None
        error = None
        try:
            await modules.uninstall(slug, purge=purge)
            if purge:
                notice = f"Uninstalled module '{slug}' and purged its database schema and secrets."
            else:
                notice = f"Uninstalled module '{slug}' (database schema kept)."
        except Exception as exc:
            error = str(exc)
        return await _render_modules(session, notice=notice, error=error)

    async def modules_attach(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            return _redirect("/admin/login")
        data = await _require_csrf(request, session)
        if data is None:
            return _html("<h1>Invalid CSRF token</h1>", status_code=403)
        slug = data.get("slug", "").strip().lower()
        mcp_url = data.get("mcp_url", "").strip()
        ui_url = data.get("ui_url", "").strip() or None
        notice = None
        error = None
        try:
            await modules.attach(slug, mcp_url=mcp_url, ui_url=ui_url)
            notice = f"Attached module '{slug}'."
        except Exception as exc:
            error = str(exc)
        return await _render_modules(session, notice=notice, error=error)

    async def module_ui_proxy(request: Request) -> Response:
        session = _require_session(request)
        if session is None:
            next_path = request.url.path
            if request.url.query:
                next_path = f"{next_path}?{request.url.query}"
            # SameSite=Strict admin cookies are omitted by browsers on cross-site top-level GET
            # redirects (e.g. OAuth callbacks to /admin/m/<slug>/callback). Serve a tiny same-origin
            # meta-refresh interstitial so the follow-up same-origin request carries the cookie
            # without weakening SameSite=Strict.
            sec_fetch_site = request.headers.get("sec-fetch-site", "").strip().lower()
            sec_fetch_mode = request.headers.get("sec-fetch-mode", "").strip().lower()
            if (
                request.method == "GET"
                and sec_fetch_site == "cross-site"
                and sec_fetch_mode in ("", "navigate")
                and next_path.startswith("/admin/m/")
            ):
                safe_target = html.escape(next_path, quote=True)
                return _html(
                    "<!doctype html>"
                    '<html lang="en"><head><meta charset="utf-8">'
                    '<meta name="viewport" content="width=device-width, initial-scale=1">'
                    f'<meta http-equiv="refresh" content="0;url={safe_target}">'
                    "<title>Continuing…</title></head>"
                    f'<body><p>Completing sign-in… <a href="{safe_target}">Continue</a></p></body>'
                    "</html>",
                    status_code=200,
                )
            return _redirect(f"/admin/login?next={quote(next_path, safe='/?=&')}")
        slug = str(request.path_params.get("slug", "")).strip()
        subpath = str(request.path_params.get("path", ""))
        return await proxy_module_ui_request(
            request,
            slug=slug,
            subpath=subpath,
            module_manager=modules,
            base_url=settings.zos_base_url,
        )

    return [
        Route("/admin", overview_page, methods=["GET"]),
        Route("/admin/login", login, methods=["GET", "POST"]),
        Route("/admin/logout", logout, methods=["POST"]),
        Route("/admin/approvals", approvals_page, methods=["GET"]),
        Route("/admin/approvals/{approval_id}", approval_detail_page, methods=["GET"]),
        Route("/admin/approvals/{approval_id}/approve", approval_approve, methods=["POST"]),
        Route("/admin/approvals/{approval_id}/deny", approval_deny, methods=["POST"]),
        Route("/admin/modules", modules_page, methods=["GET"]),
        Route("/admin/modules/install", modules_install, methods=["POST"]),
        Route("/admin/modules/upgrade", modules_upgrade, methods=["POST"]),
        Route("/admin/modules/disable", modules_disable, methods=["POST"]),
        Route("/admin/modules/enable", modules_enable, methods=["POST"]),
        Route("/admin/modules/uninstall", modules_uninstall, methods=["POST"]),
        Route("/admin/modules/attach", modules_attach, methods=["POST"]),
        Route(
            "/admin/m/{slug:str}",
            module_ui_proxy,
            methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        ),
        Route(
            "/admin/m/{slug:str}/{path:path}",
            module_ui_proxy,
            methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        ),
        Route("/admin/clients", clients_page, methods=["GET"]),
        Route("/admin/clients/create", clients_create, methods=["POST"]),
        Route("/admin/clients/token", clients_new_token, methods=["POST"]),
        Route("/admin/clients/revoke", clients_revoke, methods=["POST"]),
        Route("/admin/clients/disable", clients_disable, methods=["POST"]),
        Route("/admin/clients/delete", clients_delete, methods=["POST"]),
        Route("/admin/clients/redirect-hosts/allow", clients_redirect_host_allow, methods=["POST"]),
        Route("/admin/clients/redirect-hosts/dismiss", clients_redirect_host_dismiss, methods=["POST"]),
        Route("/admin/clients/redirect-hosts/remove", clients_redirect_host_remove, methods=["POST"]),
        Route("/admin/policy", policy_page, methods=["GET"]),
        Route("/admin/policy/set", policy_set, methods=["POST"]),
        Route("/admin/policy/clear", policy_clear, methods=["POST"]),
        Route("/admin/policy/tier", policy_set_tier, methods=["POST"]),
        Route("/admin/upstreams", upstreams_page, methods=["GET"]),
        Route("/admin/upstreams/add", upstreams_add, methods=["POST"]),
        Route("/admin/upstreams/install", upstreams_install, methods=["POST"]),
        Route("/admin/upstreams/token", upstreams_set_token, methods=["POST"]),
        Route("/admin/upstreams/oauth-client", upstreams_set_oauth_client, methods=["POST"]),
        Route("/admin/upstreams/disconnect", upstreams_disconnect, methods=["POST"]),
        Route("/admin/upstreams/remove", upstreams_remove, methods=["POST"]),
        Route("/admin/activity", activity_page, methods=["GET"]),
    ]
