"""Cancellable subprocess execution; never invokes a shell."""
from collections import deque
import os
from pathlib import Path
import selectors
import shlex
import signal
import subprocess
import threading
import time


class Cancelled(RuntimeError):
    pass


class Context:
    def __init__(self, emit=None):
        self.cancel = threading.Event()
        self.emit = emit or (lambda fraction, message: None)
        self.log_path = None

    def log(self, message):
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as output:
                output.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")

    def check(self):
        if self.cancel.is_set():
            raise Cancelled("Processing cancelled")

    def report(self, fraction, message):
        self.check()
        self.log(message)
        self.emit(fraction, message)


def run(command, ctx, fraction=0.0, env=None, activity=None):
    ctx.check()
    ctx.log("Command: " + shlex.join([str(x) for x in command]))
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    if env:
        environment.update(env)
    tail = deque(maxlen=30)
    with subprocess.Popen([str(x) for x in command], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, start_new_session=True,
                          env=environment) as process:
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        pending = b""
        last = 0.0
        started = time.monotonic()
        heartbeat = started
        try:
            while selector.get_map():
                ctx.check()
                now = time.monotonic()
                if now - heartbeat >= 10:
                    status = activity() if activity else "subprocess is still running"
                    ctx.report(fraction, f"{Path(str(command[0])).name}: {status} ({int(now - started)}s elapsed)")
                    heartbeat = now
                for key, _ in selector.select(0.2):
                    chunk = os.read(key.fileobj.fileno(), 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    pending += chunk.replace(b"\r", b"\n")
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1)
                        text = line.decode(errors="replace").strip()
                        if text:
                            tail.append(text)
                            ctx.log(text)
                            if time.monotonic() - last > 0.5:
                                ctx.report(fraction, text)
                                last = time.monotonic()
                    pending = pending[-16384:]
            if pending:
                tail.append(pending.decode(errors="replace"))
                ctx.log(pending.decode(errors="replace"))
            while process.poll() is None:
                ctx.check()
                ctx.cancel.wait(0.2)
            code = process.returncode
            ctx.log(f"Exit code: {code}")
            if code:
                raise RuntimeError(f"{Path(str(command[0])).name} failed ({code}):\n" + "\n".join(tail))
        finally:
            selector.close()
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


def frames(directory):
    result = sorted(Path(directory).glob("*.png"))
    if not result:
        raise RuntimeError(f"No frames produced in {directory}")
    return result


def verify_frames(directory, count):
    result = frames(directory)
    expected = [f"{i:08d}.png" for i in range(1, count + 1)]
    if [p.name for p in result] != expected or any(p.stat().st_size == 0 for p in result):
        raise RuntimeError(f"Incomplete frame sequence in {directory}; expected {count} frames")
    return result