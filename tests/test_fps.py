from gesec_viewer.fps import FPSCounter


def test_fps_counter_uses_recent_window():
    counter = FPSCounter(window_seconds=1.0)

    counter.tick(0.0)
    counter.tick(0.5)
    fps = counter.tick(1.0)

    assert fps == 2.0


def test_fps_counter_resets():
    counter = FPSCounter()
    counter.tick(0.0)
    counter.tick(0.5)
    counter.reset()

    assert counter.fps(1.0) == 0.0
