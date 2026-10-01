from types import SimpleNamespace
import threading

import app
from pipeline.config import Settings


def test_completion_publishes_download_and_finishes(tmp_path, monkeypatch):
    output = tmp_path / "outputs" / "enhanced-test.mp4"
    output.parent.mkdir()
    output.write_bytes(b"video")
    started = threading.Event()
    finish = threading.Event()

    def process(*args):
        ctx = args[-1]
        ctx.report(0.7, "Interpolation is running")
        started.set()
        assert finish.wait(5)
        return output

    monkeypatch.setattr(app, "process_video", process)
    ui = app.build_ui(Settings(root=tmp_path, workers=1))
    enhance = next(function.fn for function in ui.fns.values() if function.fn.__name__ == "enhance")
    request = SimpleNamespace(session_hash="test-session")
    poll = next(function.fn for function in ui.fns.values() if function.fn.__name__ == "poll")
    enhance(["input.mp4"], "realesr-animevideov3", "rife-v4", 2, 60, 1, False, False, request)
    assert started.wait(5)
    try:
        assert "Interpolation is running" in poll(request)[1]
        # Pausing UI polling must not cancel the independent backend job.
        assert not app._active[request.session_hash].cancel.is_set()
    finally:
        finish.set()
    assert app._processing.acquire(timeout=5)
    app._processing.release()
    update = poll(request)
    assert update[0] == [str(output)]
    assert update[2] == "Complete — videos are ready to download."
    assert "test-session" not in app._active
    assert ui.config["dependencies"][0]["show_progress"] == "hidden"
    refresh = next(function.fn for function in ui.fns.values()
                   if function.fn.__name__ == "completed_outputs")
    assert refresh() == [str(output)]