"""
Pi-side UART reader for the Jetson camera obstacle bridge.
Protocol: architecture/vatoz_camera_perception_detailed_architecture.md SS2.2
State machine: same doc SS2.4 / fig4_state_machine.png

ponytail: Option A per team decision -- no WorldObject "camera_confirmed"
tag/persistence. feed_camera_obstacles() (not built yet, Phase 3) dedups
within a tick only. Add a real sink later if operator display or
confidence-weighted avoidance needs it.
"""
import threading
import time
from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class CameraObservation:
    bearing_deg: float
    confidence: float
    size_estimate_deg: float


class State(Enum):
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECT_WAIT = auto()
    FAILED = auto()


class _FrameParser:
    """Pure OBS/END block parser -- no I/O, so it's testable with plain
    strings. Edge cases per architecture doc SS2.2's table."""

    def __init__(self):
        self._latency_ms = None
        self._observations = []
        self._in_frame = False

    def feed_line(self, line: str):
        """Returns (latency_ms, list[CameraObservation]) when a frame
        completes, else None."""
        line = line.strip()
        if line.startswith("OBS"):
            parts = line.split()
            if len(parts) != 2:
                self._in_frame = False
                return None
            try:
                self._latency_ms = float(parts[1])
            except ValueError:
                self._in_frame = False
                return None
            self._observations = []
            self._in_frame = True
            return None
        if line == "END":
            if not self._in_frame:
                return None  # stray END, ignore
            self._in_frame = False
            result = (self._latency_ms, self._observations)
            self._observations = []
            return result
        if self._in_frame:
            parts = line.split()
            if len(parts) == 3:
                try:
                    bearing, conf, size = (float(p) for p in parts)
                    self._observations.append(CameraObservation(bearing, conf, size))
                except ValueError:
                    pass  # drop malformed detection line, keep parsing the block
            return None
        return None  # line outside any frame and not OBS/END -> discard


class CameraUartReader:
    """Background thread reading the Jetson's UART obstacle stream.
    fig4_state_machine.png: DISCONNECTED -> CONNECTING -> CONNECTED;
    a read failure (timeout / bad line / malformed frame) -> RECONNECT_WAIT
    -> CONNECTING (attempt += 1); attempts >= max -> FAILED (terminal).
    """

    def __init__(self, port: str, baudrate: int,
                 reconnect_delay_s: float = 2.0,
                 max_reconnect_attempts: int = 10,
                 health_timeout_s: float = 1.0,
                 serial_factory=None):
        self.port = port
        self.baudrate = baudrate
        self.reconnect_delay_s = reconnect_delay_s
        self.max_reconnect_attempts = max_reconnect_attempts
        self.health_timeout_s = health_timeout_s
        self._serial_factory = serial_factory or self._default_serial_factory

        self.state = State.DISCONNECTED
        self._lock = threading.Lock()
        self._latest = []
        self._last_frame_monotonic = 0.0
        self._thread = None
        self._stop = threading.Event()

    @staticmethod
    def _default_serial_factory(port, baudrate):
        import serial  # pyserial -- already a vatoz-core dependency
        return serial.Serial(port, baudrate, timeout=1.0)

    def start(self) -> None:
        """Spawns the background read thread."""
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="camera-uart-reader")
        self._thread.start()

    def stop(self) -> None:
        """Signals the thread to exit and joins it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def latest(self) -> list:
        """Never blocks, never raises. [] if no data has arrived yet."""
        with self._lock:
            return list(self._latest)

    def is_healthy(self) -> bool:
        """True if a complete frame was parsed within health_timeout_s,
        measured on the Pi's own monotonic clock only -- never compares
        against the Jetson's processing_latency_ms for staleness."""
        with self._lock:
            return (time.monotonic() - self._last_frame_monotonic) < self.health_timeout_s

    def _run(self) -> None:
        attempt = 0
        ser = None
        parser = _FrameParser()
        self.state = State.CONNECTING
        while not self._stop.is_set():
            if self.state == State.CONNECTING:
                try:
                    ser = self._serial_factory(self.port, self.baudrate)
                    self.state = State.CONNECTED
                except Exception:
                    self.state = State.RECONNECT_WAIT
            elif self.state == State.CONNECTED:
                try:
                    raw = ser.readline()
                    if not raw:
                        raise TimeoutError("readline timeout")
                    line = raw.decode("utf-8")
                    frame = parser.feed_line(line)
                    if frame is not None:
                        _latency_ms, observations = frame
                        with self._lock:
                            self._latest = observations
                            self._last_frame_monotonic = time.monotonic()
                except (TimeoutError, UnicodeDecodeError, OSError):
                    self.state = State.RECONNECT_WAIT
            elif self.state == State.RECONNECT_WAIT:
                time.sleep(self.reconnect_delay_s)
                attempt += 1
                if attempt >= self.max_reconnect_attempts:
                    self.state = State.FAILED
                    raise RuntimeError(
                        f"CameraUartReader: giving up after {attempt} "
                        f"reconnect attempts on {self.port}")
                self.state = State.CONNECTING
            elif self.state == State.FAILED:
                return
