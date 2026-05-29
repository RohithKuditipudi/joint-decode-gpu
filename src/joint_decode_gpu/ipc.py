from __future__ import annotations

import json
import logging
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)

IPC_PREFIX = "__JDGPU__:"


def emit_ipc(payload: dict[str, Any]) -> None:
    sys.stdout.write(IPC_PREFIX + json.dumps(payload) + "\n")
    sys.stdout.flush()


def read_ipc(proc: subprocess.Popen, *, expect_kind: str) -> dict[str, Any]:
    assert proc.stdout is not None
    while True:
        line = proc.stdout.readline()
        if not line:
            rc = proc.poll()
            raise RuntimeError(f"joint-decode worker pid={proc.pid} exited with rc={rc} before sending IPC")
        line = line.rstrip("\n")
        if line.startswith(IPC_PREFIX):
            payload = json.loads(line[len(IPC_PREFIX) :])
            kind = payload.get("kind")
            if kind != expect_kind:
                raise RuntimeError(f"expected IPC kind={expect_kind!r}, got {kind!r}")
            return payload
        if line:
            logger.debug("[worker pid=%d] %s", proc.pid, line)
