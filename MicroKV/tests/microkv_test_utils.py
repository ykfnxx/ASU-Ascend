import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from microkv import KVStoreClient


def build_server() -> Path:
    subprocess.run(["make", "-s", "kv_stored"], cwd=ROOT, check=True)
    return ROOT / "build" / "kv_stored"


def wait_for_socket(socket_path: str, proc: subprocess.Popen[str]) -> None:
    deadline = time.time() + 5
    last_error = None
    while time.time() < deadline:
        if proc.poll() is not None:
            stderr = proc.stderr.read() if proc.stderr is not None else ""
            raise RuntimeError(f"server exited before creating socket: {stderr.strip()}")
        if os.path.exists(socket_path):
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.connect(socket_path)
                return
            except OSError as exc:
                last_error = exc
        time.sleep(0.02)
    raise RuntimeError(f"server did not create socket {socket_path}: {last_error}")


class MicroKVServer:
    def __init__(self) -> None:
        self.temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self.socket_path = ""
        self.proc: subprocess.Popen[str] | None = None
        self.client: KVStoreClient | None = None

    def __enter__(self) -> "MicroKVServer":
        self.temp_dir = tempfile.TemporaryDirectory()
        self.socket_path = os.path.join(self.temp_dir.name, "microkv.sock")
        server_bin = build_server()
        self.proc = subprocess.Popen(
            [str(server_bin), "--socket", self.socket_path],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        wait_for_socket(self.socket_path, self.proc)
        self.client = KVStoreClient(self.socket_path)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
        if self.proc is not None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=3)
            if self.proc.stderr is not None:
                self.proc.stderr.close()
            self.proc = None
        if self.temp_dir is not None:
            self.temp_dir.cleanup()
            self.temp_dir = None

    def new_client(self) -> KVStoreClient:
        return KVStoreClient(self.socket_path)
