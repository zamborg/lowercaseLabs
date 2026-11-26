"""Unit tests verifying edge path-filter allowlist parity, image pinning, and compose isolation."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

DEPLOY_DIR = Path(__file__).resolve().parents[2] / "deploy"
CADDY_DIR = DEPLOY_DIR / "caddy"

EXPECTED_ALLOWLIST_PATHS = [
    "/mcp",
    "/mcp/*",
    "/.well-known/*",
    "/oauth/*",
    "/admin",
    "/admin/*",
    "/connect/*",
    "/actuator",
]


def test_caddy_template_and_filter_share_single_allowlist_source() -> None:
    """Caddy TLS edge and plain-HTTP edge-filter import the exact same allowlist snippet."""
    allowlist_text = (CADDY_DIR / "Caddyfile.allowlist").read_text(encoding="utf-8")
    template_text = (CADDY_DIR / "Caddyfile.template").read_text(encoding="utf-8")
    filter_text = (CADDY_DIR / "Caddyfile.filter").read_text(encoding="utf-8")

    # Both Caddyfiles must import the shared allowlist (zero drift)
    assert "import /etc/caddy/Caddyfile.allowlist" in template_text
    assert "import /etc/caddy/Caddyfile.allowlist" in filter_text

    # Plain HTTP filter disables auto_https & admin and listens on :8000
    assert "auto_https off" in filter_text
    assert "admin off" in filter_text
    assert ":8000" in filter_text

    # Parse path directive from shared allowlist
    match = re.search(r"^\s*path\s+(.+)$", allowlist_text, flags=re.MULTILINE)
    assert match is not None
    paths = match.group(1).split()
    assert paths == EXPECTED_ALLOWLIST_PATHS
    assert "/healthz" not in paths
    assert "reverse_proxy zos:8080" in allowlist_text
    assert "flush_interval -1" in allowlist_text
    assert "respond 404" in allowlist_text


def test_cloudflared_local_config_example_matches_allowlist() -> None:
    """Locally-managed cloudflared ingress example mirrors the Caddy allowlist including /actuator."""
    cfg_text = (DEPLOY_DIR / "cloudflared" / "config.yml.example").read_text(encoding="utf-8")
    patterns = re.findall(r"^\s*path:\s*(\S+)\s*$", cfg_text, flags=re.MULTILINE)
    assert patterns, "Expected ingress path rules in config.yml.example"

    def _matches(req_path: str) -> bool:
        return any(re.search(pat, req_path) for pat in patterns)

    # Allowed paths (must match Caddyfile.allowlist)
    for ok_path in (
        "/mcp",
        "/mcp/sub",
        "/.well-known/oauth-protected-resource",
        "/oauth/register",
        "/admin",
        "/admin/m/memory/",
        "/connect/linear",
        "/actuator",
    ):
        assert _matches(ok_path), f"Expected cloudflared config.yml.example to allow {ok_path}"

    # Blocked paths (must hit catch-all 404)
    for bad_path in ("/healthz", "/nope", "/", "/actuator/extra", "/mcp2"):
        assert not _matches(bad_path), f"Expected cloudflared config.yml.example to block {bad_path}"

    assert "service: http://127.0.0.1:8787" in cfg_text
    assert "service: http_status:404" in cfg_text


def test_compose_edge_files_pin_images_and_isolate_cloudflared() -> None:
    """Caddy and cloudflared use pinned version tags; exe exposes 8000; cloudflared has no filter host port."""
    caddy_yml = (DEPLOY_DIR / "docker-compose.caddy.yml").read_text(encoding="utf-8")
    filter_yml = (DEPLOY_DIR / "docker-compose.filter.yml").read_text(encoding="utf-8")
    exe_yml = (DEPLOY_DIR / "docker-compose.exe.yml").read_text(encoding="utf-8")
    cf_yml = (DEPLOY_DIR / "docker-compose.cloudflared.yml").read_text(encoding="utf-8")

    caddy_pin_re = re.compile(r"image:\s*caddy:2\.\d+\.\d+-alpine")
    assert caddy_pin_re.search(caddy_yml), "docker-compose.caddy.yml must pin caddy:2.x.y-alpine"
    assert caddy_pin_re.search(filter_yml), "docker-compose.filter.yml must pin caddy:2.x.y-alpine"
    assert "./caddy/Caddyfile.allowlist:/etc/caddy/Caddyfile.allowlist:ro" in caddy_yml
    assert "./caddy/Caddyfile.allowlist:/etc/caddy/Caddyfile.allowlist:ro" in filter_yml

    # Base filter publishes no host ports; exe override publishes 8000:8000
    assert "ports:" not in filter_yml
    assert '"8000:8000"' in exe_yml

    # Cloudflared pins a specific version (never :latest), forwards to edge-filter:8000, no host ports
    assert ":latest" not in cf_yml
    assert re.search(r"image:\s*cloudflare/cloudflared:20\d{2}\.\d+\.\d+", cf_yml)
    assert "tunnel --no-autoupdate run --url http://edge-filter:8000" in cf_yml
    assert "ports:" not in cf_yml
    assert "zos_edge" in cf_yml


def test_install_script_syntax_and_exe_edge_persistence(tmp_path: Path) -> None:
    """install.sh passes bash -n and persists/reuses --edge exe, domain, and ref cleanly."""
    script_path = DEPLOY_DIR / "install.sh"
    res = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True, check=False)
    assert res.returncode == 0, res.stderr

    lines = script_path.read_text(encoding="utf-8").splitlines()
    end_idx = next(i for i, line in enumerate(lines) if "validate_digest_image()" in line)
    snippet = "\n".join(lines[:end_idx]).replace('if [ "$(id -u)" -ne 0 ]; then', "if false; then")
    harness = snippet + '\necho "EDGE=$EDGE|DOMAIN=$DOMAIN|REF=$GIT_REF|SOURCE=$LOCAL_SOURCE"\n'

    def _resolve(env_text: str | None, args: list[str]) -> dict[str, str]:
        d = tmp_path / f"case_{abs(hash((env_text, tuple(args))))}"
        zos_dir = d / "zos"
        zos_dir.mkdir(parents=True, exist_ok=True)
        if env_text is not None:
            (zos_dir / ".env").write_text(env_text, encoding="utf-8")
        out = subprocess.check_output(
            ["bash", "-c", harness, "--", "--dir", str(d), *args],
            text=True,
        ).strip()
        return dict(part.split("=", 1) for part in out.split("|"))

    # Fresh --edge exe with tag
    r1 = _resolve(None, ["--edge", "exe", "--ref", "zos-v3.0.1"])
    assert r1 == {"EDGE": "exe", "DOMAIN": "", "REF": "zos-v3.0.1", "SOURCE": ""}

    # Re-run Step 3 without flags preserves exe edge and tag (never moves to main)
    r2 = _resolve("ZOS_INSTALL_EDGE=exe\nZOS_INSTALL_DOMAIN=\nZOS_INSTALL_REF=zos-v3.0.1\n", [])
    assert r2 == {"EDGE": "exe", "DOMAIN": "", "REF": "zos-v3.0.1", "SOURCE": ""}

    # Custom domain with --edge exe persists across re-run
    r3 = _resolve("ZOS_INSTALL_EDGE=exe\nZOS_INSTALL_DOMAIN=zos.custom.dev\nZOS_INSTALL_REF=zos-v3.0.1\n", [])
    assert r3 == {"EDGE": "exe", "DOMAIN": "zos.custom.dev", "REF": "zos-v3.0.1", "SOURCE": ""}

    # Switching caddy -> exe without --domain clears prior caddy domain
    r4 = _resolve(
        "ZOS_INSTALL_EDGE=caddy\nZOS_INSTALL_DOMAIN=old.example.com\nZOS_INSTALL_REF=zos-v3.0.1\n",
        ["--edge", "exe"],
    )
    assert r4 == {"EDGE": "exe", "DOMAIN": "", "REF": "zos-v3.0.1", "SOURCE": ""}
