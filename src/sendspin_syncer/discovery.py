"""Find Sendspin players on the network through mDNS.

Discovery only *listens* for ``_sendspin._tcp`` adverts; it never connects to
a player, so it can't disturb Music Assistant.
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from zeroconf import IPVersion, ServiceStateChange, Zeroconf
from zeroconf.asyncio import AsyncServiceBrowser, AsyncServiceInfo, AsyncZeroconf

SERVICE_TYPE = "_sendspin._tcp.local."
# Spec-1.0 client ids are 32-byte Curve25519 keys in unpadded base64url.
_KEY_ID = re.compile(r"^[A-Za-z0-9_-]{43}$")


@dataclass(slots=True)
class DiscoveredPlayer:
    instance: str
    """mDNS instance name, which players set to their client id."""
    name: str
    host: str
    port: int
    path: str
    properties: dict[str, str] = field(default_factory=dict)

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"ws://{host}:{self.port}{self.path}"

    @property
    def client_id_hint(self) -> str:
        return self.instance

    @property
    def looks_encrypted(self) -> bool:
        """Spec-1.0 (Noise) players use their public key as the client id."""
        return bool(_KEY_ID.match(self.instance))

    def matches(self, needle: str) -> bool:
        n = needle.lower()
        return n in self.name.lower() or n in self.instance.lower() or n == self.host

    @classmethod
    def from_url(cls, url: str) -> DiscoveredPlayer:
        parts = urlsplit(url)
        if parts.scheme not in ("ws", "wss") or not parts.hostname:
            raise ValueError(f"not a websocket URL: {url!r} (expected ws://host:port/path)")
        return cls(
            instance=url,
            name=parts.hostname,
            host=parts.hostname,
            port=parts.port or 8928,
            path=parts.path or "/sendspin",
        )


def _pick_address(addresses: list[str]) -> str | None:
    """Prefer a routable IPv4 address; skip link-local and unspecified ones."""
    usable = []
    for a in addresses:
        try:
            ip = ipaddress.ip_address(a)
        except ValueError:
            continue
        if not (ip.is_link_local or ip.is_unspecified):
            usable.append(ip)
    usable.sort(key=lambda ip: (ip.is_loopback, ip.version != 4))
    return str(usable[0]) if usable else None


async def discover_players(timeout_s: float = 4.0) -> list[DiscoveredPlayer]:
    """Browse mDNS for ``timeout_s`` seconds and return every player seen."""
    found: dict[str, DiscoveredPlayer] = {}
    pending: set[asyncio.Task[None]] = set()
    azc = AsyncZeroconf(ip_version=IPVersion.V4Only)

    async def resolve(zc: Zeroconf, service_type: str, name: str) -> None:
        info = AsyncServiceInfo(service_type, name)
        if not info.load_from_cache(zc):
            await info.async_request(zc, 3000)
        host = _pick_address(info.parsed_addresses())
        if host is None or info.port is None:
            return
        props = {
            (k.decode() if isinstance(k, bytes) else k): (
                v.decode(errors="replace") if isinstance(v, bytes) else (v or "")
            )
            for k, v in (info.properties or {}).items()
        }
        path = props.get("path") or "/sendspin"
        if not path.startswith("/"):
            path = "/" + path
        instance = name.removesuffix("." + SERVICE_TYPE)
        found[name] = DiscoveredPlayer(
            instance=instance,
            name=props.get("name") or instance,
            host=host,
            port=info.port,
            path=path,
            properties=props,
        )

    def on_change(
        zeroconf: Zeroconf, service_type: str, name: str, state_change: ServiceStateChange
    ) -> None:
        if state_change is ServiceStateChange.Removed:
            found.pop(name, None)
            return
        task = asyncio.ensure_future(resolve(zeroconf, service_type, name))
        pending.add(task)
        task.add_done_callback(pending.discard)

    browser = AsyncServiceBrowser(azc.zeroconf, SERVICE_TYPE, handlers=[on_change])
    try:
        await asyncio.sleep(timeout_s)
        if pending:
            await asyncio.wait(pending, timeout=3.5)
    finally:
        await browser.async_cancel()
        await azc.async_close()
    return sorted(found.values(), key=lambda p: p.name.lower())


def select_players(
    players: list[DiscoveredPlayer], filters: list[str] | None
) -> list[DiscoveredPlayer]:
    """Keep players matching any filter (name, id or host). No filters keeps all."""
    if not filters:
        return players
    chosen: list[DiscoveredPlayer] = []
    for f in filters:
        hits = [p for p in players if p.matches(f)]
        if not hits:
            raise LookupError(f"no discovered player matches {f!r}")
        chosen.extend(h for h in hits if h not in chosen)
    return chosen
