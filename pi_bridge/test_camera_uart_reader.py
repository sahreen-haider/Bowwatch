"""
Phase 1 validation (architecture doc SS5/SS6): parser edge cases from
SS2.2's table, plus the reconnect state machine from SS2.4/fig4.
No real serial hardware needed -- fake readline() sources throughout.

Run: python -m unittest pi_bridge.test_camera_uart_reader -v
"""
import time
import unittest

from camera_uart_reader import CameraUartReader, State, _FrameParser


class FrameParserTests(unittest.TestCase):
    def test_basic_frame(self):
        p = _FrameParser()
        self.assertIsNone(p.feed_line("OBS 145"))
        self.assertIsNone(p.feed_line("-12.3 0.87 4.1"))
        self.assertIsNone(p.feed_line("31.0 0.62 2.0"))
        latency, obs = p.feed_line("END")
        self.assertEqual(latency, 145.0)
        self.assertEqual(len(obs), 2)
        self.assertEqual(obs[0].bearing_deg, -12.3)

    def test_zero_detection_heartbeat(self):
        p = _FrameParser()
        p.feed_line("OBS 50")
        latency, obs = p.feed_line("END")
        self.assertEqual(latency, 50.0)
        self.assertEqual(obs, [])

    def test_garbage_line_outside_frame_discarded(self):
        p = _FrameParser()
        self.assertIsNone(p.feed_line("garbage\x00"))
        self.assertIsNone(p.feed_line("OBS 10"))
        _, obs = p.feed_line("END")
        self.assertEqual(obs, [])

    def test_new_obs_before_end_discards_partial(self):
        p = _FrameParser()
        p.feed_line("OBS 10")
        p.feed_line("1.0 0.5 1.0")  # this detection should be discarded
        p.feed_line("OBS 20")       # new frame starts before END
        latency, obs = p.feed_line("END")
        self.assertEqual(latency, 20.0)
        self.assertEqual(obs, [])

    def test_malformed_detection_line_dropped_rest_kept(self):
        p = _FrameParser()
        p.feed_line("OBS 10")
        p.feed_line("not a number 0.5 1.0")  # dropped
        p.feed_line("1.0 2.0")               # wrong field count, dropped
        p.feed_line("5.0 0.5 1.0")           # valid
        _, obs = p.feed_line("END")
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].bearing_deg, 5.0)

    def test_stray_end_ignored(self):
        p = _FrameParser()
        self.assertIsNone(p.feed_line("END"))  # no crash, no frame emitted


class FakeSerial:
    """Scripted readline() source -- no real port needed."""

    def __init__(self, lines):
        self._lines = iter(lines)

    def readline(self):
        line = next(self._lines, "")
        if line is TimeoutError:
            raise TimeoutError()
        return (line + "\n").encode("utf-8") if line else b""


class ReaderStateMachineTests(unittest.TestCase):
    def test_reader_parses_happy_path_via_real_thread(self):
        lines = ["OBS 10", "1.0 0.5 1.0", "END"]
        reader = CameraUartReader(
            "fake", 115200, health_timeout_s=5.0, reconnect_delay_s=0.01,
            serial_factory=lambda port, baud: FakeSerial(lines))
        reader.start()
        try:
            for _ in range(200):  # poll up to ~1s instead of a fixed sleep
                if reader.latest():
                    break
                time.sleep(0.005)
        finally:
            reader.stop()
        obs = reader.latest()
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs[0].bearing_deg, 1.0)
        self.assertTrue(reader.is_healthy())

    def test_reaches_failed_after_max_attempts(self):
        reader = CameraUartReader(
            "fake", 115200, reconnect_delay_s=0.001, max_reconnect_attempts=3,
            serial_factory=lambda port, baud: (_ for _ in ()).throw(OSError("no port")))
        with self.assertRaises(RuntimeError):
            reader._run()
        self.assertEqual(reader.state, State.FAILED)

    def test_is_healthy_false_before_any_frame(self):
        reader = CameraUartReader("fake", 115200)
        self.assertFalse(reader.is_healthy())

    def test_latest_empty_before_any_frame(self):
        reader = CameraUartReader("fake", 115200)
        self.assertEqual(reader.latest(), [])


if __name__ == "__main__":
    unittest.main()
