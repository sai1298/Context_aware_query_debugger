from __future__ import annotations

import subprocess
import sys


class SandboxExecutor:
    """Run untrusted code in a subprocess with timeout and optional memory limits."""

    def __init__(self, timeout: int = 10, memory_limit_mb: int = 256) -> None:
        self.timeout = timeout
        self.memory_limit_mb = memory_limit_mb

    def _preexec(self) -> None:
        """Set resource limits on Unix before the child process starts."""
        try:
            import resource  # type: ignore[import]
            limit_bytes = self.memory_limit_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
        except Exception:
            # Gracefully skip on Windows or if resource module unavailable.
            pass

    def execute(self, code: str, stdin: str = "") -> tuple[bool, str, str]:
        """Execute *code* in an isolated subprocess.

        Returns:
            (success, stdout, stderr)
            Never raises — all exceptions are caught and returned as (False, "", error_str).
        """
        try:
            preexec = None if sys.platform == "win32" else self._preexec

            result = subprocess.run(
                [sys.executable, "-c", code],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                preexec_fn=preexec,
            )
            success = result.returncode == 0
            return success, result.stdout, result.stderr
        except subprocess.TimeoutExpired as exc:
            return False, "", f"TimeoutExpired: code did not finish within {self.timeout}s"
        except Exception as exc:
            return False, "", str(exc)
