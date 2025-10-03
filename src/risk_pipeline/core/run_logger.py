import os
import sys
from datetime import datetime


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass


class RunLogger:
    """Redirect stdout/stderr to a log file while preserving console output."""

    def __init__(self, folder: str = 'logs', filename: str = 'last_run.log'):
        self.folder = folder or 'logs'
        self.filename = filename or 'last_run.log'
        self._orig_stdout = None
        self._orig_stderr = None
        self._file = None
        self._installed = False

    def install(self):
        if self._installed:
            return
        try:
            os.makedirs(self.folder, exist_ok=True)
            path = os.path.join(self.folder, self.filename)
            # overwrite for each run
            self._file = open(path, 'w', encoding='utf-8')
            header = f"[{datetime.now().isoformat()}] RUN START\n"
            self._file.write(header)
            self._file.flush()
            self._orig_stdout, self._orig_stderr = sys.stdout, sys.stderr
            sys.stdout = _Tee(self._orig_stdout, self._file)
            sys.stderr = _Tee(self._orig_stderr, self._file)
            self._installed = True
        except Exception:
            # Best-effort logging; ignore failures
            self._installed = False

    def uninstall(self):
        if not self._installed:
            return
        try:
            if self._orig_stdout is not None:
                sys.stdout = self._orig_stdout
            if self._orig_stderr is not None:
                sys.stderr = self._orig_stderr
            if self._file:
                try:
                    self._file.write(f"[{datetime.now().isoformat()}] RUN END\n")
                except Exception:
                    pass
                self._file.flush()
                self._file.close()
        finally:
            self._installed = False

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.uninstall()

