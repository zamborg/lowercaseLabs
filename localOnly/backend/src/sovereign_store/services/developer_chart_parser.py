from __future__ import annotations

import shutil
import subprocess
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from sovereign_store.core.config import Settings, get_settings
from sovereign_store.models.app import NetworkingMode


@dataclass(slots=True)
class DeveloperChartParseResult:
    parse_mode: str
    repository: str
    dockerfile_found: bool
    fly_toml_found: bool
    fly_app_name: str | None
    networking_mode: NetworkingMode | None
    internal_port: int | None
    requires_volume: bool
    warnings: list[str]


class DeveloperChartParser:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def parse(
        self,
        *,
        repo_path: str | None,
        repo_url: str | None,
        ref: str | None = None,
    ) -> DeveloperChartParseResult:
        if repo_path and repo_url:
            raise ValueError("Provide only one of repo_path or repo_url")
        if not repo_path and not repo_url:
            raise ValueError("Either repo_path or repo_url is required")

        if repo_path:
            candidate = Path(repo_path).expanduser().resolve()
            return self._parse_checked_out_repo(
                candidate=candidate,
                parse_mode="local",
                repository_label=str(candidate),
                enforce_local_guardrail=True,
            )

        remote_url = repo_url or ""
        return self._parse_remote_repo(repo_url=remote_url, ref=ref)

    def _parse_checked_out_repo(
        self,
        *,
        candidate: Path,
        parse_mode: str,
        repository_label: str,
        enforce_local_guardrail: bool,
    ) -> DeveloperChartParseResult:
        if enforce_local_guardrail and self.settings.developer_repo_root:
            allowed_root = Path(self.settings.developer_repo_root).expanduser().resolve()
            if not str(candidate).startswith(str(allowed_root)):
                raise ValueError(
                    f"repo_path must be within DEVELOPER_REPO_ROOT ({allowed_root})"
                )

        if not candidate.exists() or not candidate.is_dir():
            raise ValueError(f"Repo path does not exist: {candidate}")

        dockerfile = candidate / "Dockerfile"
        fly_toml = candidate / "fly.toml"

        warnings: list[str] = []
        fly_app_name: str | None = None
        networking_mode: NetworkingMode | None = None
        internal_port: int | None = None
        requires_volume = False

        if not dockerfile.exists():
            warnings.append("Dockerfile missing")

        if fly_toml.exists():
            (
                fly_app_name,
                networking_mode,
                internal_port,
                requires_volume,
                parse_warnings,
            ) = self._parse_fly_toml(fly_toml)
            warnings.extend(parse_warnings)
        else:
            warnings.append("fly.toml missing")

        return DeveloperChartParseResult(
            parse_mode=parse_mode,
            repository=repository_label,
            dockerfile_found=dockerfile.exists(),
            fly_toml_found=fly_toml.exists(),
            fly_app_name=fly_app_name,
            networking_mode=networking_mode,
            internal_port=internal_port,
            requires_volume=requires_volume,
            warnings=warnings,
        )

    def _parse_remote_repo(self, *, repo_url: str, ref: str | None) -> DeveloperChartParseResult:
        self._validate_remote_url(repo_url)

        clone_root = Path(self.settings.repo_clone_root).expanduser().resolve()
        clone_root.mkdir(parents=True, exist_ok=True)

        checkout_dir = clone_root / f"chart-{uuid.uuid4().hex}"
        timeout = self.settings.repo_clone_timeout_seconds

        try:
            self._run(
                ["git", "clone", "--depth", "1", repo_url, str(checkout_dir)],
                timeout=timeout,
                error_prefix="git clone",
            )

            if ref:
                self._checkout_ref(checkout_dir=checkout_dir, ref=ref, timeout=timeout)

            parsed = self._parse_checked_out_repo(
                candidate=checkout_dir,
                parse_mode="remote",
                repository_label=repo_url,
                enforce_local_guardrail=False,
            )
            parsed.warnings.append(
                "Remote repository parsed via temporary git clone; clone directory cleaned up after parse."
            )
            return parsed
        finally:
            shutil.rmtree(checkout_dir, ignore_errors=True)

    def _validate_remote_url(self, repo_url: str) -> None:
        parsed = urlparse(repo_url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("repo_url must be an http(s) URL")
        if not parsed.netloc:
            raise ValueError("repo_url must include a host")

        if self.settings.allowed_repo_hosts:
            allowed = self.settings.allowed_repo_hosts
            if parsed.netloc not in allowed:
                raise ValueError(
                    f"repo_url host '{parsed.netloc}' is not allowed; expected one of {sorted(allowed)}"
                )

    def _checkout_ref(self, *, checkout_dir: Path, ref: str, timeout: int) -> None:
        # First try local checkout (works for default branch names, tags that were cloned).
        try:
            self._run(
                ["git", "-C", str(checkout_dir), "checkout", ref],
                timeout=timeout,
                error_prefix="git checkout",
            )
            return
        except ValueError:
            pass

        # Fallback for branches/tags/SHAs not in shallow clone.
        self._run(
            ["git", "-C", str(checkout_dir), "fetch", "--depth", "1", "origin", ref],
            timeout=timeout,
            error_prefix="git fetch",
        )
        self._run(
            ["git", "-C", str(checkout_dir), "checkout", "FETCH_HEAD"],
            timeout=timeout,
            error_prefix="git checkout FETCH_HEAD",
        )

    @staticmethod
    def _run(command: list[str], *, timeout: int, error_prefix: str) -> None:
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ValueError("git executable was not found on PATH") from exc
        except subprocess.TimeoutExpired as exc:
            raise ValueError(f"{error_prefix} timed out after {timeout}s") from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            raise ValueError(f"{error_prefix} failed: {stderr or exc}") from exc

    def _parse_fly_toml(
        self, fly_toml: Path
    ) -> tuple[str | None, NetworkingMode | None, int | None, bool, list[str]]:
        warnings: list[str] = []

        with fly_toml.open("rb") as handle:
            payload = tomllib.load(handle)

        fly_app_name = payload.get("app")
        internal_port = self._extract_internal_port(payload)
        requires_volume = bool(payload.get("mounts"))

        networking_mode_raw = self._extract_networking_mode(payload)
        networking_mode: NetworkingMode | None = None
        if networking_mode_raw:
            try:
                networking_mode = NetworkingMode(networking_mode_raw)
            except ValueError:
                warnings.append(
                    f"Unsupported networking_mode '{networking_mode_raw}' (expected proxy/direct)"
                )
        else:
            warnings.append("No networking_mode found in fly.toml metadata")

        if internal_port is None:
            warnings.append("No internal_port found in fly.toml")

        return fly_app_name, networking_mode, internal_port, requires_volume, warnings

    @staticmethod
    def _extract_networking_mode(payload: dict) -> str | None:
        top_level = payload.get("networking_mode")
        if isinstance(top_level, str):
            return top_level.strip().lower()

        metadata = payload.get("metadata")
        if isinstance(metadata, dict):
            mode = metadata.get("networking_mode")
            if isinstance(mode, str):
                return mode.strip().lower()

        deploy = payload.get("deploy")
        if isinstance(deploy, dict):
            deploy_meta = deploy.get("metadata")
            if isinstance(deploy_meta, dict):
                mode = deploy_meta.get("networking_mode")
                if isinstance(mode, str):
                    return mode.strip().lower()

        return None

    @staticmethod
    def _extract_internal_port(payload: dict) -> int | None:
        http_service = payload.get("http_service")
        if isinstance(http_service, dict):
            port = http_service.get("internal_port")
            if isinstance(port, int):
                return port

        services = payload.get("services")
        if isinstance(services, list) and services:
            first_service = services[0]
            if isinstance(first_service, dict):
                port = first_service.get("internal_port")
                if isinstance(port, int):
                    return port

        return None
