"""CPU-only checks for the Mooncake service configured for a KVCM instance group."""

import http.client
import json
import os
from urllib.parse import urlsplit


class ServicePreflightError(RuntimeError):
    def __init__(self, diagnostics):
        self.diagnostics = diagnostics
        super().__init__(
            "KVCM service preflight failed: " + json.dumps(diagnostics, sort_keys=True)
        )


def service_settings(
    mooncake_master=None, master_metrics_port=None, storage_config=None
):
    """Resolve the SDK's master host and metrics port without loading native ops."""
    master = mooncake_master or os.environ.get("KVCM_MOONCAKE_MASTER")
    config_path = storage_config or os.environ.get("KVCM_STORAGE_CONFIG")
    if config_path:
        with open(config_path) as source:
            storage = json.load(source)
        storage = storage.get("storage", storage)
        mooncake = storage.get("mooncake", {})
        configured_master = mooncake.get("master_service_entry") or mooncake.get(
            "master_server_entry"
        )
        if not isinstance(configured_master, str) or not configured_master:
            raise ValueError(
                "KVCM storage config requires mooncake.master_service_entry (or master_server_entry)"
            )
        if master and master != configured_master:
            raise ValueError("KVCM Mooncake master and storage config disagree")
        master = configured_master

    def port_number(value):
        if (
            isinstance(value, bool)
            or not str(value).isascii()
            or not str(value).isdigit()
        ):
            raise ValueError("MASTER_METRICS must be an integer port from 1 to 65535")
        port = int(value)
        if not 1 <= port <= 65535:
            raise ValueError("MASTER_METRICS must be an integer port from 1 to 65535")
        return port

    environment_port = os.environ.get("MASTER_METRICS")
    port = port_number(
        master_metrics_port
        if master_metrics_port is not None
        else environment_port or 9003
    )
    if (
        master_metrics_port is not None
        and environment_port
        and port != port_number(environment_port)
    ):
        raise ValueError(
            "Configured Mooncake metrics port disagrees with MASTER_METRICS in this process"
        )
    return master, port


def _capacity_metrics(payload):
    # client_c.cc consumes these two unlabelled uint64 gauges, not JSON or
    # per-segment labelled series. Reject ambiguous or unsupported responses.
    names = {"master_allocated_bytes", "master_total_capacity_bytes"}
    values = {}
    for line in payload.splitlines():
        fields = line.split()
        if not fields or fields[0] not in names:
            continue
        name = fields[0]
        if name in values or len(fields) not in (2, 3):
            raise ValueError("Duplicate or malformed capacity metric " + name)
        raw = fields[1]
        if not raw.isascii() or not raw.isdigit() or int(raw) >= 1 << 64:
            raise ValueError("Capacity metric must be uint64: " + name)
        values[name] = int(raw)
    missing = names - values.keys()
    if missing:
        raise ValueError(
            "Missing Mooncake capacity metrics: " + ", ".join(sorted(missing))
        )
    return values["master_allocated_bytes"], values["master_total_capacity_bytes"]


def service_preflight(
    address,
    instance_group,
    *,
    mooncake_master=None,
    master_metrics_port=None,
    storage_config=None,
    min_free_bytes=1,
    timeout_s=5,
    instance_id=None,
):
    """Check /health and /metrics using the vendor client_c.cc service contract.

    This proves HTTP health and available memory capacity at this instant. The
    native GPU probe is still required for KVCM registration and RDMA transfers.
    """
    master, port = service_settings(
        mooncake_master, master_metrics_port, storage_config
    )
    if not master or "://" in master:
        raise ValueError(
            "Mooncake service preflight requires a direct master host:port; HA resolution is unsupported"
        )
    endpoint = urlsplit("http://" + master)
    if (
        not endpoint.hostname
        or endpoint.username is not None
        or endpoint.password is not None
        or endpoint.path
        or endpoint.query
        or endpoint.fragment
        or any(c.isspace() for c in master)
    ):
        raise ValueError(
            "Mooncake service preflight requires a direct master host:port"
        )
    if endpoint.port is not None and not 1 <= endpoint.port <= 65535:
        raise ValueError("Invalid Mooncake master RPC port")
    if min_free_bytes <= 0 or timeout_s <= 0:
        raise ValueError(
            "Service preflight requires positive min_free_bytes and timeout_s"
        )
    host = endpoint.hostname
    http_host = "[" + host + "]" if ":" in host else host
    base_url = f"http://{http_host}:{port}"
    result = {
        "status": "checking",
        "operation": "service_preflight",
        "address": address,
        "instance_group": instance_group,
        "instance_id": instance_id,
        "mooncake_master": master,
        "master_metrics_port": port,
        "metrics_base_url": base_url,
        "min_free_bytes": min_free_bytes,
    }

    def fetch(path):
        connection = http.client.HTTPConnection(host, port, timeout=timeout_s)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            payload = response.read((4 << 20) + 1)
            if len(payload) > 4 << 20:
                raise ValueError("Mooncake HTTP response exceeds 4 MiB")
            return response.status, payload.decode("utf-8")
        finally:
            connection.close()

    try:
        result["endpoint"] = base_url + "/health"
        result["health_status_code"], _ = fetch("/health")
        result["endpoint"] = base_url + "/metrics"
        result["metrics_status_code"], payload = fetch("/metrics")
        if result["metrics_status_code"] != 200:
            raise ValueError("Mooncake /metrics did not return HTTP 200")
        allocated, total = _capacity_metrics(payload)
        result.update(
            allocated_bytes=allocated,
            total_capacity_bytes=total,
            free_bytes=total - allocated,
        )
        if result["health_status_code"] != 200:
            raise ValueError("Mooncake /health did not return HTTP 200")
        if total == 0 or allocated > total or total - allocated < min_free_bytes:
            raise ValueError(
                "Mooncake has insufficient free capacity; check mounted storage segments"
            )
    except (OSError, http.client.HTTPException, ValueError) as error:
        result.update(status="failed", error=str(error))
        raise ServicePreflightError(result) from error
    result["status"] = "healthy"
    return result
