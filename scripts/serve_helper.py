"""Starts the real API as a separate uvicorn process for live tests and benchmarks."""
import os
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def running_server(port: int | None = None, startup_timeout: float = 60.0):
    """Yields (base_url, process, log_path, seconds_to_healthy). Always terminates the process."""
    port = port or free_port()
    log = Path(tempfile.mkdtemp()) / "server.log"
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    with open(log, "w") as fh:
        t0 = time.perf_counter()
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "sentinel.api.app:app", "--port", str(port), "--log-level", "warning"],
                                cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        base = f"http://127.0.0.1:{port}"
        try:
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"server exited early:\n{log.read_text()[-2000:]}")
                try:
                    if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                        break
                except httpx.HTTPError:
                    pass
                if time.perf_counter() - t0 > startup_timeout:
                    raise TimeoutError("server did not become healthy")
                time.sleep(0.2)
            yield base, proc, log, time.perf_counter() - t0
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
