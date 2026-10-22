import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch


@contextlib.contextmanager
def mooncake_server(health_status=200, metrics_status=200, metrics=None):
    if metrics is None:
        metrics = (
            "# TYPE master_allocated_bytes gauge\n"
            "master_allocated_bytes 1024\n"
            "master_total_capacity_bytes 8192\n"
        )
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            status, body = {
                "/health": (health_status, "OK"),
                "/metrics": (metrics_status, metrics),
            }.get(self.path, (404, ""))
            self.send_response(status)
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port, requests
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


class KVCMServicePreflightTest(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def preflight(self, port, **kwargs):
        from rtp_llm.utils.kvcm_service_preflight import service_preflight

        return service_preflight(
            "kvcm:6381",
            "rollout-group",
            mooncake_master="127.0.0.1:50151",
            master_metrics_port=port,
            **kwargs,
        )

    def test_checks_http_health_and_actual_free_capacity(self):
        with mooncake_server() as (port, requests):
            result = self.preflight(port, min_free_bytes=4096)
        self.assertEqual(requests, ["/health", "/metrics"])
        self.assertEqual(result["status"], "healthy")
        self.assertEqual(result["free_bytes"], 7168)
        self.assertEqual(result["total_capacity_bytes"], 8192)
        self.assertEqual(result["allocated_bytes"], 1024)
        self.assertEqual(result["master_metrics_port"], port)
        self.assertEqual(result["address"], "kvcm:6381")
        self.assertEqual(result["instance_group"], "rollout-group")

    def test_rejects_unhealthy_http_even_with_available_capacity(self):
        with (
            mooncake_server(health_status=503) as (port, _),
            self.assertRaises(RuntimeError) as error,
        ):
            self.preflight(port)
        diagnostic = error.exception.diagnostics
        self.assertEqual(diagnostic["health_status_code"], 503)
        self.assertEqual(diagnostic["free_bytes"], 7168)
        self.assertEqual(diagnostic["status"], "failed")

    def test_rejects_missing_invalid_and_exhausted_storage_capacity(self):
        for metrics in (
            "master_allocated_bytes 0\nmaster_total_capacity_bytes 0\n",
            "master_allocated_bytes 8192\nmaster_total_capacity_bytes 8192\n",
            "master_allocated_bytes 8193\nmaster_total_capacity_bytes 8192\n",
            "master_allocated_bytes 0\n",
            "master_allocated_bytes -1\nmaster_total_capacity_bytes 8192\n",
            "master_allocated_bytes NaN\nmaster_total_capacity_bytes 8192\n",
            'master_allocated_bytes{node="a"} 0\nmaster_total_capacity_bytes 8192\n',
            "master_allocated_bytes 0\nmaster_allocated_bytes 1\nmaster_total_capacity_bytes 8192\n",
        ):
            with (
                self.subTest(metrics=metrics),
                mooncake_server(metrics=metrics) as (port, _),
                self.assertRaises(RuntimeError),
            ):
                self.preflight(port, min_free_bytes=1)

    def test_requires_requested_free_capacity(self):
        with (
            mooncake_server() as (port, _),
            self.assertRaisesRegex(RuntimeError, "free capacity"),
        ):
            self.preflight(port, min_free_bytes=7169)

    def test_reports_metrics_http_failure_with_endpoint_context(self):
        with (
            mooncake_server(metrics_status=503) as (port, _),
            self.assertRaises(RuntimeError) as error,
        ):
            self.preflight(port)
        text = str(error.exception)
        for expected in ("kvcm:6381", "rollout-group", "/metrics", "503", str(port)):
            self.assertIn(expected, text)

    def test_reports_connection_failure_as_service_health_failure(self):
        with mooncake_server() as (port, _):
            pass
        with self.assertRaises(RuntimeError) as error:
            self.preflight(port, timeout_s=0.1)
        self.assertEqual(error.exception.diagnostics["status"], "failed")
        self.assertTrue(error.exception.diagnostics["endpoint"].endswith("/health"))

    def test_reads_storage_config_and_metrics_environment(self):
        from rtp_llm.utils.kvcm_service_preflight import service_preflight

        with mooncake_server() as (port, _), tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "storage.json"
            config.write_text(
                json.dumps(
                    {
                        "storage": {
                            "mooncake": {
                                "master_service_entry": "127.0.0.1:50151",
                                "metadata_connstring": "http://127.0.0.1:8082/metadata",
                            }
                        }
                    }
                )
            )
            with patch.dict(
                os.environ,
                {"KVCM_STORAGE_CONFIG": str(config), "MASTER_METRICS": str(port)},
            ):
                result = service_preflight("kvcm:6381", "rollout-group")
        self.assertEqual(result["mooncake_master"], "127.0.0.1:50151")
        self.assertEqual(result["master_metrics_port"], port)

    def test_rejects_disagreement_with_sdk_metrics_environment(self):
        with (
            patch.dict(os.environ, {"MASTER_METRICS": "9003"}),
            self.assertRaisesRegex(ValueError, "MASTER_METRICS"),
        ):
            self.preflight(9004)

    def test_rejects_conflicting_master_and_storage_config(self):
        from rtp_llm.utils.kvcm_service_preflight import service_preflight

        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / "storage.json"
            config.write_text(
                json.dumps({"mooncake": {"master_server_entry": "actual:50151"}})
            )
            with self.assertRaisesRegex(ValueError, "disagree"):
                service_preflight(
                    "kvcm:6381",
                    "rollout-group",
                    mooncake_master="other:50151",
                    storage_config=str(config),
                )

    def test_rejects_unresolved_ha_master_without_assuming_a_host(self):
        from rtp_llm.utils.kvcm_service_preflight import service_preflight

        with self.assertRaisesRegex(ValueError, "direct"):
            service_preflight(
                "kvcm:6381", "rollout-group", mooncake_master="etcd://etcd:2379"
            )

    def test_service_only_cli_needs_no_cuda_and_checks_real_endpoint(self):
        with mooncake_server() as (port, _):
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rtp_llm.utils.kvcm_probe",
                    "--service-only",
                    "--address",
                    "kvcm:6381",
                    "--instance-group",
                    "rollout-group",
                    "--mooncake-master",
                    "127.0.0.1:50151",
                    "--master-metrics-port",
                    str(port),
                    "--min-free-bytes",
                    "4096",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["free_bytes"], 7168)

    def test_gpu_probe_reports_checked_health_and_shares_metrics_port(self):
        from rtp_llm.utils.kvcm_probe import run_probe

        with (
            mooncake_server() as (port, _),
            patch(
                "rtp_llm.utils.kvcm_probe.subprocess.run",
                return_value=subprocess.CompletedProcess(
                    [], 1, "", "StartWrite sdk_error_code=51"
                ),
            ) as child,
        ):
            with self.assertRaises(RuntimeError) as error:
                run_probe(
                    "kvcm:6381",
                    "rollout-group",
                    '[{"type":"mooncake"}]',
                    size=4096,
                    mooncake_master="127.0.0.1:50151",
                    master_metrics_port=port,
                )
            self.assertEqual(child.call_args.kwargs["env"]["MASTER_METRICS"], str(port))
        for expected in (
            '"free_bytes": 7168',
            '"health_status_code": 200',
            "sdk_error_code=51",
        ):
            self.assertIn(expected, str(error.exception))

    def test_unhealthy_storage_blocks_before_gpu_subprocess(self):
        from rtp_llm.utils.kvcm_probe import run_probe

        with (
            mooncake_server(health_status=503) as (port, _),
            patch(
                "rtp_llm.utils.kvcm_probe.subprocess.run",
                side_effect=AssertionError(
                    "GPU probe must not start with unhealthy storage"
                ),
            ),
            self.assertRaisesRegex(RuntimeError, "service preflight failed"),
        ):
            run_probe(
                "kvcm:6381",
                "rollout-group",
                '[{"type":"mooncake"}]',
                size=4096,
                mooncake_master="127.0.0.1:50151",
                master_metrics_port=port,
            )

    def test_metrics_default_and_invalid_environment_are_explicit(self):
        from rtp_llm.utils.kvcm_service_preflight import service_settings

        self.assertEqual(service_settings("master:50151"), ("master:50151", 9003))
        for raw in ("0", "65536", "garbage", "-1"):
            with (
                self.subTest(raw=raw),
                patch.dict(os.environ, {"MASTER_METRICS": raw}),
                self.assertRaisesRegex(ValueError, "MASTER_METRICS"),
            ):
                service_settings("master:50151")


if __name__ == "__main__":
    unittest.main()
