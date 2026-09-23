# SPDX-License-Identifier: Apache-2.0
"""Generate synthetic MCAP recordings so the example runs with no robot.

Writes files that look like what a real stack records: several topics at
different rates, interleaved by log time, chunked and zstd-compressed, with a
summary section -- because those are the properties the ingest job depends on.

Four channels, chosen to cover the cases the reader has to handle:

===========================  ==============  =============================
Topic                        Rate            Why it is here
===========================  ==============  =============================
``/imu``                     100 Hz          The high-rate JSON telemetry
``/odom``                    20 Hz           A second JSON schema
``/camera/front/compressed`` 5 Hz            Binary payload, *no* schema
``/diagnostics``             1 Hz            Sparse, string-heavy JSON
===========================  ==============  =============================

The camera payload is random bytes, not a real JPEG. It stands in for the
thing that makes these files large; nothing decodes it.

Run with::

    python examples/mcap_ingest/make_sample_mcap.py --out ./sample_logs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from mcap.well_known import MessageEncoding, SchemaEncoding
from mcap.writer import Writer

NS_PER_S = 1_000_000_000

#: Where the synthetic recording starts: 2024-01-01T00:00:00Z, in nanoseconds.
DEFAULT_START_NS = 1_704_067_200 * NS_PER_S

#: Bytes per synthetic camera frame. Small enough that a demo run stays in the
#: tens of MB, large enough that payload storage is visible in the output.
FRAME_BYTES = 4096


@dataclass(frozen=True)
class TopicSpec:
    """One channel to record: what it publishes and how often."""

    topic: str
    hz: float
    schema_name: str
    #: None for a channel with no schema at all (MCAP schema_id 0), which is
    #: legal and which the reader has to cope with.
    json_schema: dict[str, Any] | None
    message_encoding: str = MessageEncoding.JSON


TOPICS: tuple[TopicSpec, ...] = (
    TopicSpec(
        topic="/imu",
        hz=100.0,
        schema_name="sensor_msgs/Imu",
        json_schema={
            "type": "object",
            "properties": {
                "accel": {"type": "array", "items": {"type": "number"}},
                "gyro": {"type": "array", "items": {"type": "number"}},
                "temperature_c": {"type": "number"},
            },
        },
    ),
    TopicSpec(
        topic="/odom",
        hz=20.0,
        schema_name="nav_msgs/Odometry",
        json_schema={
            "type": "object",
            "properties": {
                "x": {"type": "number"},
                "y": {"type": "number"},
                "heading_rad": {"type": "number"},
                "speed_mps": {"type": "number"},
            },
        },
    ),
    TopicSpec(
        topic="/camera/front/compressed",
        hz=5.0,
        schema_name="",
        json_schema=None,
        # A custom encoding string. MCAP allows any encoding a producer likes;
        # the well-known list is a convention, not a constraint.
        message_encoding="jpeg",
    ),
    TopicSpec(
        topic="/diagnostics",
        hz=1.0,
        schema_name="diagnostic_msgs/DiagnosticStatus",
        json_schema={
            "type": "object",
            "properties": {
                "level": {"type": "integer"},
                "name": {"type": "string"},
                "message": {"type": "string"},
            },
        },
    ),
)

#: Diagnostic lines, cycled through so the column has repeated values worth
#: filtering on.
DIAGNOSTIC_MESSAGES = (
    (0, "battery", "nominal"),
    (0, "motor_controller", "nominal"),
    (1, "lidar", "dropped frames"),
    (0, "planner", "nominal"),
    (2, "battery", "cell imbalance detected"),
)


def _payload(spec: TopicSpec, index: int, rng: np.random.Generator) -> bytes:
    """One message on ``spec``, deterministic given ``rng``."""
    if spec.topic == "/imu":
        return json.dumps(
            {
                "accel": [round(float(v), 4) for v in rng.normal(0, 0.2, 3)],
                "gyro": [round(float(v), 4) for v in rng.normal(0, 0.05, 3)],
                "temperature_c": round(38.0 + float(rng.normal(0, 0.5)), 3),
            }
        ).encode("utf-8")
    if spec.topic == "/odom":
        return json.dumps(
            {
                "x": round(index * 0.05, 4),
                "y": round(float(np.sin(index * 0.01)), 4),
                "heading_rad": round(float(index * 0.001 % 6.283), 4),
                "speed_mps": round(1.0 + float(rng.normal(0, 0.1)), 4),
            }
        ).encode("utf-8")
    if spec.topic == "/diagnostics":
        level, name, message = DIAGNOSTIC_MESSAGES[index % len(DIAGNOSTIC_MESSAGES)]
        return json.dumps({"level": level, "name": name, "message": message}).encode(
            "utf-8"
        )
    # The camera frame: opaque bytes, sized like a small compressed image.
    return rng.integers(0, 256, FRAME_BYTES, dtype=np.uint8).tobytes()


def plan_messages(duration_s: float, start_ns: int) -> list[tuple[int, TopicSpec, int]]:
    """Every message the file will hold, in log-time order.

    Sorting once here is what makes the recording look like a real one: the
    channels are interleaved by time, not written topic by topic, so a reader
    asking for one topic genuinely has to skip past the others.
    """
    planned: list[tuple[int, TopicSpec, int]] = []
    for spec in TOPICS:
        count = max(1, int(duration_s * spec.hz))
        period_ns = int(NS_PER_S / spec.hz)
        planned.extend((start_ns + i * period_ns, spec, i) for i in range(count))
    # Sort on the timestamp and the topic, never on the spec itself:
    # TopicSpec is not ordered, and equal timestamps do occur at these rates.
    planned.sort(key=lambda item: (item[0], item[1].topic))
    return planned


def write_sample_file(
    path: Path,
    *,
    duration_s: float = 10.0,
    start_ns: int = DEFAULT_START_NS,
    seed: int = 0,
) -> int:
    """Write one recording and return how many messages it holds."""
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer = Writer(stream)
        writer.start(profile="", library="lancedb-ray-example")

        channel_ids: dict[str, int] = {}
        for spec in TOPICS:
            if spec.json_schema is None:
                # schema_id 0: the channel declares no schema.
                schema_id = 0
            else:
                schema_id = writer.register_schema(
                    name=spec.schema_name,
                    encoding=SchemaEncoding.JSONSchema,
                    data=json.dumps(spec.json_schema).encode("utf-8"),
                )
            channel_ids[spec.topic] = writer.register_channel(
                topic=spec.topic,
                message_encoding=spec.message_encoding,
                schema_id=schema_id,
            )

        planned = plan_messages(duration_s, start_ns)
        for log_time, spec, index in planned:
            writer.add_message(
                channel_id=channel_ids[spec.topic],
                log_time=log_time,
                # Published a little before it was logged, as a real pipeline
                # would be -- the two columns are not redundant.
                publish_time=log_time - 1_000_000,
                sequence=index,
                data=_payload(spec, index, rng),
            )
        writer.finish()
    return len(planned)


def write_sample_files(
    out_dir: Path, *, files: int = 4, duration_s: float = 10.0, seed: int = 0
) -> list[Path]:
    """Write consecutive recordings, as a robot rotating log files would."""
    paths = []
    for i in range(files):
        path = out_dir / f"robot_log_{i:04d}.mcap"
        write_sample_file(
            path,
            duration_s=duration_s,
            # Back to back, so the files together cover one continuous window.
            start_ns=DEFAULT_START_NS + int(i * duration_s * NS_PER_S),
            seed=seed + i,
        )
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./sample_logs", help="Output directory")
    parser.add_argument("--files", type=int, default=4, help="Recordings to write")
    parser.add_argument(
        "--duration", type=float, default=10.0, help="Seconds per recording"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out).expanduser().resolve()
    paths = write_sample_files(
        out_dir, files=args.files, duration_s=args.duration, seed=args.seed
    )
    total_bytes = sum(p.stat().st_size for p in paths)
    print(f"Wrote {len(paths)} recordings to {out_dir} ({total_bytes / 1e6:.1f} MB)")
    for path in paths:
        print(f"  {path.name}  {path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
