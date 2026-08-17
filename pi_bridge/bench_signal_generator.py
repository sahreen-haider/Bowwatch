"""
Phase 1 bench signal generator (architecture doc SS5): emits synthetic
OBS/END frames onto a serial port, so CameraUartReader can be validated
with no real Jetson/camera/LiDAR.

Loopback setup on WSL/Linux (two virtual ports wired together):
    socat -d -d pty,raw,echo=0,link=/tmp/ttyV0 pty,raw,echo=0,link=/tmp/ttyV1
Then point this script at /tmp/ttyV0 and CameraUartReader at /tmp/ttyV1.

Usage:
    python bench_signal_generator.py /tmp/ttyV0 --rate 10 --max-obstacles 3
"""
import argparse
import random
import time

import serial


def make_frame(latency_ms: float, n_obstacles: int) -> str:
    lines = [f"OBS {latency_ms:.0f}"]
    for _ in range(n_obstacles):
        bearing = random.uniform(-45, 45)
        confidence = random.uniform(0.4, 0.95)
        size = random.uniform(1.0, 6.0)
        lines.append(f"{bearing:.1f} {confidence:.2f} {size:.1f}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("port")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--rate", type=float, default=10.0, help="frames/sec")
    ap.add_argument("--max-obstacles", type=int, default=3)
    args = ap.parse_args()

    ser = serial.Serial(args.port, args.baud, timeout=1.0)
    period = 1.0 / args.rate
    print(f"Publishing synthetic OBS/END frames to {args.port} at {args.rate} Hz. Ctrl+C to stop.")
    try:
        while True:
            n = random.randint(0, args.max_obstacles)
            latency_ms = random.uniform(20, 200)
            ser.write(make_frame(latency_ms, n).encode("utf-8"))
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
