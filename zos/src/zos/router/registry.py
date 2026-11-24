"""Provider registry holding hosted, upstream, and actuator ToolProviders."""

from __future__ import annotations

from zos.providers.base import ToolProvider
from zos.router.naming import validate_prefix


class ProviderRegistry:
    """In-process registry of ToolProvider instances keyed by namespace."""

    def __init__(self) -> None:
        self._providers: dict[str, ToolProvider] = {}

    def register(self, provider: ToolProvider) -> None:
        validate_prefix(provider.namespace, allow_reserved=False)
        self._providers[provider.namespace] = provider

    def unregister(self, namespace: str) -> ToolProvider | None:
        return self._providers.pop(namespace, None)

    def get(self, namespace: str) -> ToolProvider | None:
        return self._providers.get(namespace)

    def all_providers(self) -> list[ToolProvider]:
        return [self._providers[k] for k in sorted(self._providers)]

    def online_providers(self) -> list[ToolProvider]:
        return [p for p in self.all_providers() if p.is_online()]
