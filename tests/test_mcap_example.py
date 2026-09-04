"""Tests for the MCAP ingest example.

The example is code people copy, so it is tested like library code: the reader
against deliberately awkward files (no schema, no index, no statistics,
timestamps Arrow cannot hold), and the pipeline end to end through real Ray
tasks against a real LanceDB table.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import lancedb
import pyarrow as pa
import pytest
from mcap.well_known import MessageEncoding, SchemaEncoding
from mcap.writer import IndexType, Writer

import ingest_mcap as ingest_module
import make_sample_mcap as sample_module
import mcap_source
from ingest_mcap import (
    decode_files,
    expected_row_count,
    ingest,
    time_slice,
    topic_counts,
)
from mcap_source import (
    INT64_MAX,
    ChannelSummary,
    build_schema,
    discover_files,
    empty_batch,
    iter_record_batches,
    summarize_file,
)

# One second of the sample recording, in nanoseconds.
SECOND_NS = 1_000_000_000


@dataclass(frozen=True)
class Msg:
    """One message to write into a test log."""

    topic: str
    data: bytes
    log_time: int
    publish_time: Optional[int] = None
    sequence: int = 0
    encoding: str = MessageEncoding.JSON
    #: None registers the channel with no schema at all (MCAP schema_id 0).
    schema_name: Optional[str] = "test/Message"


def write_log(
    path: Path,
    messages: list[Msg],
    *,
    chunked: bool = True,
    statistics: bool = True,
    summary: bool = True,
) -> Path:
    """Write an MCAP file containing exactly ``messages``.

    The three flags cover the files that turn up in practice and break naive
    readers: unchunked recordings, recordings whose writer was killed before it
    could count anything, and recordings with no summary section at all.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        writer = Writer(
            stream,
            use_chunking=chunked,
            use_statistics=statistics,
            use_summary_offsets=summary,
            repeat_channels=summary,
            repeat_schemas=summary,
            index_types=IndexType.ALL if summary else IndexType.NONE,
        )
        writer.start(profile="", library="lancedb-ray-tests")
        channels: dict[tuple[str, str, Optional[str]], int] = {}
        for message in messages:
            key = (message.topic, message.encoding, message.schema_name)
            if key not in channels:
                schema_id = 0
                if message.schema_name is not None:
                    schema_id = writer.register_schema(
                        name=message.schema_name,
                        encoding=SchemaEncoding.JSONSchema,
                        data=b"{}",
                    )
                channels[key] = writer.register_channel(
                    topic=message.topic,
                    message_encoding=message.encoding,
                    schema_id=schema_id,
                )
            writer.add_message(
                channel_id=channels[key],
                log_time=message.log_time,
                publish_time=(
                    message.log_time
                    if message.publish_time is None
                    else message.publish_time
                ),
                sequence=message.sequence,
                data=message.data,
            )
        writer.finish()
    return path


def simple_log(path: Path, count: int = 10, *, topic: str = "/a") -> Path:
    """A log of ``count`` JSON messages on one topic, one per second."""
    return write_log(
        path,
        [
            Msg(topic=topic, data=json.dumps({"i": i}).encode(), log_time=i * SECOND_NS)
            for i in range(count)
        ],
    )


def read_all(path: Union[str, Path], **kwargs: Any) -> pa.Table:
    """Every batch a file yields, concatenated."""
    batches = list(iter_record_batches(path, **kwargs))
    include_payload = kwargs.get("include_payload", True)
    if not batches:
        return pa.Table.from_batches([empty_batch(include_payload=include_payload)])
    return pa.Table.from_batches(batches)


@pytest.fixture(scope="session")
def sample_logs(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Three consecutive synthetic recordings, generated once per session."""
    out_dir = tmp_path_factory.mktemp("mcap_logs")
    sample_module.write_sample_files(out_dir, files=3, duration_s=2.0)
    return out_dir


@pytest.fixture(scope="session")
def sample_files(sample_logs: Path) -> list[Path]:
    return discover_files(sample_logs)


#: Messages per topic in one 2-second sample recording.
SAMPLE_PER_FILE = {
    "/imu": 200,
    "/odom": 40,
    "/camera/front/compressed": 10,
    "/diagnostics": 2,
}
SAMPLE_FILE_ROWS = sum(SAMPLE_PER_FILE.values())


class TestSchema:
    def test_columns_and_types(self) -> None:
        schema = build_schema()
        assert schema.names == [
            "source_file",
            "topic",
            "schema_name",
            "message_encoding",
            "sequence",
            "log_time",
            "log_time_ns",
            "publish_time_ns",
            "payload_size",
            "json_payload",
            "payload",
        ]
        assert schema.field("log_time").type == pa.timestamp("ns")
        assert schema.field("log_time_ns").type == pa.int64()
        assert schema.field("payload").type == pa.large_binary()

    def test_payload_can_be_dropped(self) -> None:
        schema = build_schema(include_payload=False)
        assert "payload" not in schema.names
        assert "payload_size" in schema.names

    def test_only_decoded_columns_are_nullable(self) -> None:
        # Everything else is a property of the message record itself, so a null
        # would mean the reader lost it rather than that it was absent.
        nullable = {f.name for f in build_schema() if f.nullable}
        assert nullable == {"json_payload", "payload"}

    def test_empty_batch_carries_the_schema(self) -> None:
        batch = empty_batch()
        assert batch.num_rows == 0
        assert batch.schema == build_schema()
        assert empty_batch(include_payload=False).schema == build_schema(
            include_payload=False
        )


class TestIterRecordBatches:
    def test_reads_every_message(self, tmp_path: Path) -> None:
        table = read_all(simple_log(tmp_path / "log.mcap", 25))
        assert table.num_rows == 25
        assert table.schema == build_schema()
        assert table.column("topic").to_pylist() == ["/a"] * 25

    def test_columns_carry_the_message_record(self, tmp_path: Path) -> None:
        path = write_log(
            tmp_path / "log.mcap",
            [
                Msg(
                    topic="/telemetry",
                    data=b'{"v": 1}',
                    log_time=5 * SECOND_NS,
                    publish_time=4 * SECOND_NS,
                    sequence=7,
                    schema_name="telemetry/V1",
                )
            ],
        )
        row = read_all(path).to_pylist()[0]
        assert row["source_file"] == str(path)
        assert row["topic"] == "/telemetry"
        assert row["schema_name"] == "telemetry/V1"
        assert row["message_encoding"] == "json"
        assert row["sequence"] == 7
        assert row["log_time_ns"] == 5 * SECOND_NS
        assert row["publish_time_ns"] == 4 * SECOND_NS
        assert row["payload_size"] == len(b'{"v": 1}')
        assert row["json_payload"] == '{"v": 1}'
        assert row["payload"] == b'{"v": 1}'

    def test_log_time_matches_the_raw_nanoseconds(self, tmp_path: Path) -> None:
        path = simple_log(tmp_path / "log.mcap", 3)
        table = read_all(path)
        as_ns = table.column("log_time").cast(pa.int64()).to_pylist()
        assert as_ns == table.column("log_time_ns").to_pylist()

    def test_messages_come_back_in_log_time_order(self, tmp_path: Path) -> None:
        path = write_log(
            tmp_path / "log.mcap",
            [
                Msg(topic="/a", data=b"{}", log_time=30 * SECOND_NS),
                Msg(topic="/b", data=b"{}", log_time=10 * SECOND_NS),
                Msg(topic="/a", data=b"{}", log_time=20 * SECOND_NS),
            ],
        )
        times = read_all(path).column("log_time_ns").to_pylist()
        assert times == sorted(times)

    @pytest.mark.parametrize(
        ("count", "batch_size", "expected"),
        [
            (10, 10, [10]),
            (10, 4, [4, 4, 2]),
            (10, 1, [1] * 10),
            (10, 100, [10]),
            (0, 4, []),
        ],
    )
    def test_batch_size_bounds_each_batch(
        self, tmp_path: Path, count: int, batch_size: int, expected: list[int]
    ) -> None:
        path = simple_log(tmp_path / "log.mcap", count)
        batches = list(iter_record_batches(path, batch_size=batch_size))
        assert [b.num_rows for b in batches] == expected
        assert all(b.schema == build_schema() for b in batches)

    def test_rejects_a_non_positive_batch_size(self, tmp_path: Path) -> None:
        path = simple_log(tmp_path / "log.mcap", 1)
        with pytest.raises(ValueError, match="batch_size must be positive"):
            # The generator is lazy, so the error surfaces on first use --
            # which is where a caller would hit it.
            list(iter_record_batches(path, batch_size=0))

    def test_an_empty_file_yields_nothing(self, tmp_path: Path) -> None:
        path = write_log(tmp_path / "log.mcap", [])
        assert list(iter_record_batches(path)) == []

    def test_topics_are_pushed_down(self, tmp_path: Path) -> None:
        path = write_log(
            tmp_path / "log.mcap",
            [Msg(topic="/a", data=b"{}", log_time=i * SECOND_NS) for i in range(5)]
            + [Msg(topic="/b", data=b"{}", log_time=i * SECOND_NS) for i in range(5)],
        )
        assert read_all(path, topics=["/a"]).num_rows == 5
        assert read_all(path, topics=["/a", "/b"]).num_rows == 10
        assert read_all(path, topics=["/missing"]).num_rows == 0

    def test_time_bounds_are_half_open(self, tmp_path: Path) -> None:
        path = simple_log(tmp_path / "log.mcap", 10)
        window = read_all(path, start_time=3 * SECOND_NS, end_time=6 * SECOND_NS)
        assert window.column("log_time_ns").to_pylist() == [
            3 * SECOND_NS,
            4 * SECOND_NS,
            5 * SECOND_NS,
        ]

    def test_payload_can_be_left_behind(self, tmp_path: Path) -> None:
        path = simple_log(tmp_path / "log.mcap", 4)
        table = read_all(path, include_payload=False)
        assert "payload" not in table.schema.names
        # The size is still recorded: what was dropped is the bytes, not the
        # knowledge of how large they were.
        assert all(size > 0 for size in table.column("payload_size").to_pylist())

    def test_a_channel_with_no_schema_reads_as_empty(self, tmp_path: Path) -> None:
        path = write_log(
            tmp_path / "log.mcap",
            [Msg(topic="/raw", data=b"\x00\x01", log_time=1, schema_name=None)],
        )
        row = read_all(path).to_pylist()[0]
        assert row["schema_name"] == ""

    def test_json_payload_is_only_set_for_json_channels(self, tmp_path: Path) -> None:
        path = write_log(
            tmp_path / "log.mcap",
            [
                Msg(topic="/json", data=b'{"a": 1}', log_time=1),
                Msg(
                    topic="/binary",
                    data=b"\x00\x01\x02",
                    log_time=2,
                    encoding="jpeg",
                    schema_name=None,
                ),
            ],
        )
        rows = {row["topic"]: row for row in read_all(path).to_pylist()}
        assert rows["/json"]["json_payload"] == '{"a": 1}'
        assert rows["/binary"]["json_payload"] is None
        assert rows["/binary"]["payload"] == b"\x00\x01\x02"

    def test_undecodable_json_keeps_its_bytes(self, tmp_path: Path) -> None:
        # A JSON channel is UTF-8 by definition, so this message is corrupt or
        # mislabelled. It must not take the row down with it.
        path = write_log(
            tmp_path / "log.mcap",
            [Msg(topic="/json", data=b"\xff\xfe not utf-8", log_time=1)],
        )
        row = read_all(path).to_pylist()[0]
        assert row["json_payload"] is None
        assert row["payload"] == b"\xff\xfe not utf-8"

    def test_malformed_json_is_carried_through_verbatim(self, tmp_path: Path) -> None:
        # Valid UTF-8, invalid JSON. The column is the message text, not a
        # promise that it parses, so nothing is dropped.
        path = write_log(
            tmp_path / "log.mcap",
            [Msg(topic="/json", data=b'{"a": ', log_time=1)],
        )
        assert read_all(path).to_pylist()[0]["json_payload"] == '{"a": '

    def test_unicode_survives_the_round_trip(self, tmp_path: Path) -> None:
        payload = json.dumps({"msg": "café ☕"}).encode("utf-8")
        path = write_log(
            tmp_path / "log.mcap", [Msg(topic="/json", data=payload, log_time=1)]
        )
        row = read_all(path).to_pylist()[0]
        assert json.loads(row["json_payload"])["msg"] == "café ☕"

    @pytest.mark.parametrize("field", ["log_time", "publish_time"])
    def test_a_timestamp_beyond_int64_is_refused(
        self, tmp_path: Path, field: str
    ) -> None:
        # MCAP timestamps are uint64 and Arrow's are signed, so the top half of
        # the range is representable in a log and not in the table. Wrapping it
        # into a negative timestamp would be silent corruption.
        too_large = INT64_MAX + 1
        message = Msg(
            topic="/clock",
            data=b"{}",
            log_time=too_large if field == "log_time" else 1,
            publish_time=too_large if field == "publish_time" else 1,
        )
        path = write_log(tmp_path / "log.mcap", [message])
        with pytest.raises(ValueError, match=f"{field}={too_large}"):
            read_all(path)

    def test_a_timestamp_at_the_int64_ceiling_is_allowed(self, tmp_path: Path) -> None:
        path = write_log(
            tmp_path / "log.mcap",
            [Msg(topic="/clock", data=b"{}", log_time=INT64_MAX)],
        )
        assert read_all(path).column("log_time_ns").to_pylist() == [INT64_MAX]

    @pytest.mark.parametrize(
        ("chunked", "statistics", "summary"),
        [
            (True, True, True),
            (False, True, True),
            (True, False, True),
            (False, False, False),
        ],
        ids=["indexed", "unchunked", "no-statistics", "no-summary"],
    )
    def test_reads_files_without_an_index(
        self, tmp_path: Path, chunked: bool, statistics: bool, summary: bool
    ) -> None:
        # A recording whose writer was killed has no summary section, and a
        # reader that only knows how to seek the index reads nothing from it.
        path = write_log(
            tmp_path / "log.mcap",
            [
                Msg(topic="/a", data=b"{}", log_time=i * SECOND_NS, sequence=i)
                for i in range(6)
            ],
            chunked=chunked,
            statistics=statistics,
            summary=summary,
        )
        assert read_all(path).num_rows == 6

    def test_accepts_a_string_path(self, tmp_path: Path) -> None:
        path = simple_log(tmp_path / "log.mcap", 3)
        assert read_all(str(path)).num_rows == 3

    def test_reads_the_generated_sample(self, sample_files: list[Path]) -> None:
        table = read_all(sample_files[0])
        assert table.num_rows == SAMPLE_FILE_ROWS
        counts = {
            topic: table.column("topic").to_pylist().count(topic)
            for topic in SAMPLE_PER_FILE
        }
        assert counts == SAMPLE_PER_FILE


class TestSummarizeFile:
    def test_reads_counts_from_the_summary_section(
        self, sample_files: list[Path]
    ) -> None:
        summary = summarize_file(sample_files[0])
        assert summary.scanned is False
        assert summary.message_count == SAMPLE_FILE_ROWS
        assert summary.path == str(sample_files[0])
        assert {c.topic: c.message_count for c in summary.channels} == SAMPLE_PER_FILE

    def test_describes_each_channel(self, sample_files: list[Path]) -> None:
        channels = {c.topic: c for c in summarize_file(sample_files[0]).channels}
        assert channels["/imu"].schema_name == "sensor_msgs/Imu"
        assert channels["/imu"].message_encoding == "json"
        # The camera channel is registered with no schema at all.
        assert channels["/camera/front/compressed"].schema_name == ""
        assert channels["/camera/front/compressed"].message_encoding == "jpeg"

    def test_time_range_covers_the_messages(self, sample_files: list[Path]) -> None:
        summary = summarize_file(sample_files[0])
        times = read_all(sample_files[0]).column("log_time_ns").to_pylist()
        assert summary.start_time_ns == min(times)
        assert summary.end_time_ns == max(times)

    @pytest.mark.parametrize(
        ("statistics", "summary_section"), [(False, True), (False, False)]
    )
    def test_falls_back_to_a_scan(
        self, tmp_path: Path, statistics: bool, summary_section: bool
    ) -> None:
        path = write_log(
            tmp_path / "log.mcap",
            [Msg(topic="/a", data=b"{}", log_time=i * SECOND_NS) for i in range(4)]
            + [Msg(topic="/b", data=b"{}", log_time=SECOND_NS, schema_name=None)],
            statistics=statistics,
            summary=summary_section,
        )
        summary = summarize_file(path)
        assert summary.scanned is True
        assert summary.message_count == 5
        assert {c.topic: c.message_count for c in summary.channels} == {
            "/a": 4,
            "/b": 1,
        }
        assert summary.start_time_ns == 0
        assert summary.end_time_ns == 3 * SECOND_NS
        assert {c.topic: c.schema_name for c in summary.channels} == {
            "/a": "test/Message",
            "/b": "",
        }

    def test_scanning_an_empty_file(self, tmp_path: Path) -> None:
        path = write_log(tmp_path / "log.mcap", [], statistics=False, summary=False)
        summary = summarize_file(path)
        assert summary.message_count == 0
        assert summary.channels == ()
        assert summary.start_time_ns == 0
        assert summary.end_time_ns == 0

    def test_matching_count_selects_topics(self, sample_files: list[Path]) -> None:
        summary = summarize_file(sample_files[0])
        assert summary.matching_count() == SAMPLE_FILE_ROWS
        assert summary.matching_count(["/imu"]) == SAMPLE_PER_FILE["/imu"]
        assert (
            summary.matching_count(["/imu", "/odom"])
            == SAMPLE_PER_FILE["/imu"] + SAMPLE_PER_FILE["/odom"]
        )
        assert summary.matching_count(["/nope"]) == 0
        # An empty list is a filter matching nothing, not an absent filter.
        assert summary.matching_count([]) == 0

    def test_matching_count_agrees_with_a_filtered_read(
        self, sample_files: list[Path]
    ) -> None:
        # The whole point of reading the summary is that it predicts the read.
        summary = summarize_file(sample_files[0])
        for topics in ([], ["/imu"], ["/odom", "/diagnostics"], None):
            assert (
                summary.matching_count(topics)
                == read_all(sample_files[0], topics=topics).num_rows
            )

    def test_channel_summary_is_hashable(self) -> None:
        # Frozen dataclasses, so a summary can go in a set or a dict key.
        assert len({ChannelSummary("/a", "S", "json", 1)}) == 1


class TestDiscoverFiles:
    def test_finds_files_recursively_and_sorted(self, tmp_path: Path) -> None:
        simple_log(tmp_path / "b.mcap", 1)
        simple_log(tmp_path / "a.mcap", 1)
        simple_log(tmp_path / "nested" / "c.mcap", 1)
        (tmp_path / "notes.txt").write_text("not a log")
        found = discover_files(tmp_path)
        assert [p.name for p in found] == ["a.mcap", "b.mcap", "c.mcap"]

    def test_accepts_a_single_file(self, tmp_path: Path) -> None:
        path = simple_log(tmp_path / "one.mcap", 1)
        assert discover_files(path) == [path]
        assert discover_files(str(path)) == [path]

    def test_an_empty_directory_finds_nothing(self, tmp_path: Path) -> None:
        assert discover_files(tmp_path) == []


class TestSampleGenerator:
    def test_plan_is_ordered_and_complete(self) -> None:
        planned = sample_module.plan_messages(2.0, start_ns=0)
        assert [t for t, _, _ in planned] == sorted(t for t, _, _ in planned)
        counts: dict[str, int] = {}
        for _, spec, _ in planned:
            counts[spec.topic] = counts.get(spec.topic, 0) + 1
        assert counts == SAMPLE_PER_FILE

    def test_a_short_duration_still_emits_one_message_per_topic(self) -> None:
        # 0.1s of a 1 Hz topic rounds to zero messages, which would leave the
        # channel out of the file entirely.
        planned = sample_module.plan_messages(0.1, start_ns=0)
        topics = {spec.topic for _, spec, _ in planned}
        assert topics == set(SAMPLE_PER_FILE)

    def test_timestamps_start_where_asked(self) -> None:
        planned = sample_module.plan_messages(1.0, start_ns=1234)
        assert min(t for t, _, _ in planned) == 1234

    def test_written_file_matches_its_plan(self, tmp_path: Path) -> None:
        path = tmp_path / "one.mcap"
        written = sample_module.write_sample_file(path, duration_s=1.0)
        assert written == summarize_file(path).message_count

    def test_the_same_seed_writes_the_same_file(self, tmp_path: Path) -> None:
        first = tmp_path / "a.mcap"
        second = tmp_path / "b.mcap"
        third = tmp_path / "c.mcap"
        sample_module.write_sample_file(first, duration_s=1.0, seed=1)
        sample_module.write_sample_file(second, duration_s=1.0, seed=1)
        sample_module.write_sample_file(third, duration_s=1.0, seed=2)
        assert first.read_bytes() == second.read_bytes()
        assert first.read_bytes() != third.read_bytes()

    def test_payloads_have_the_shape_each_topic_declares(self) -> None:
        import numpy as np

        rng = np.random.default_rng(0)
        by_topic = {spec.topic: spec for spec in sample_module.TOPICS}
        imu = json.loads(sample_module._payload(by_topic["/imu"], 0, rng))
        assert len(imu["accel"]) == 3 and len(imu["gyro"]) == 3
        odom = json.loads(sample_module._payload(by_topic["/odom"], 3, rng))
        assert odom["x"] == pytest.approx(0.15)
        diag = json.loads(sample_module._payload(by_topic["/diagnostics"], 2, rng))
        assert diag["name"] == "lidar"
        frame = sample_module._payload(by_topic["/camera/front/compressed"], 0, rng)
        assert len(frame) == sample_module.FRAME_BYTES

    def test_files_are_written_back_to_back(self, tmp_path: Path) -> None:
        paths = sample_module.write_sample_files(tmp_path, files=3, duration_s=1.0)
        assert [p.name for p in paths] == [
            "robot_log_0000.mcap",
            "robot_log_0001.mcap",
            "robot_log_0002.mcap",
        ]
        summaries = [summarize_file(p) for p in paths]
        for earlier, later in zip(summaries, summaries[1:], strict=False):
            assert earlier.end_time_ns < later.start_time_ns

    def test_cli_writes_the_requested_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        out = tmp_path / "logs"
        monkeypatch.setattr(
            "sys.argv",
            [
                "make_sample_mcap.py",
                "--out",
                str(out),
                "--files",
                "2",
                "--duration",
                "1.0",
                "--seed",
                "3",
            ],
        )
        sample_module.main()
        assert len(discover_files(out)) == 2
        assert "Wrote 2 recordings" in capsys.readouterr().out


class TestDecodeFiles:
    def test_yields_one_stream_of_batches_per_file(self, tmp_path: Path) -> None:
        first = simple_log(tmp_path / "a.mcap", 5)
        second = simple_log(tmp_path / "b.mcap", 3)
        batch = pa.table({"path": [str(first), str(second)]})
        tables = list(decode_files(batch, batch_size=2))
        assert [t.num_rows for t in tables] == [2, 2, 1, 2, 1]
        assert all(t.schema == build_schema() for t in tables)

    def test_an_empty_file_still_announces_the_schema(self, tmp_path: Path) -> None:
        # Without this a run whose first file is empty has no schema to create
        # the table from.
        path = write_log(tmp_path / "empty.mcap", [])
        tables = list(decode_files(pa.table({"path": [str(path)]})))
        assert [t.num_rows for t in tables] == [0]
        assert tables[0].schema == build_schema()

    def test_passes_its_filters_down(self, tmp_path: Path) -> None:
        path = write_log(
            tmp_path / "log.mcap",
            [
                Msg(topic="/a", data=b"{}", log_time=1),
                Msg(topic="/b", data=b"{}", log_time=2),
            ],
        )
        tables = list(
            decode_files(
                pa.table({"path": [str(path)]}),
                topics=["/b"],
                include_payload=False,
            )
        )
        assert tables[0].column("topic").to_pylist() == ["/b"]
        assert "payload" not in tables[0].schema.names


class TestExpectedRowCount:
    def test_sums_every_file(self, sample_files: list[Path]) -> None:
        assert expected_row_count(sample_files) == SAMPLE_FILE_ROWS * len(sample_files)

    def test_respects_a_topic_filter(self, sample_files: list[Path]) -> None:
        assert expected_row_count(sample_files, ["/odom"]) == SAMPLE_PER_FILE[
            "/odom"
        ] * len(sample_files)

    def test_no_files_is_no_rows(self) -> None:
        assert expected_row_count([]) == 0


class TestIngest:
    def test_writes_every_message_in_one_commit(
        self, sample_files: list[Path], db_dir: str
    ) -> None:
        result = ingest(sample_files, uri=db_dir)
        assert result.files == len(sample_files)
        assert result.expected_rows == SAMPLE_FILE_ROWS * len(sample_files)
        assert result.rows_written == result.expected_rows
        assert result.rows_in_table == result.expected_rows
        assert result.complete
        # The property this library exists for: however many tasks wrote, the
        # table advanced by exactly one version.
        assert result.versions == 1
        assert result.fragments >= 1
        assert result.elapsed_s > 0

    def test_the_table_holds_what_the_files_held(
        self, sample_files: list[Path], db_dir: str
    ) -> None:
        ingest(sample_files, uri=db_dir)
        table = lancedb.connect(db_dir).open_table("messages")
        rows = table.search(None).limit(None).to_arrow()
        assert rows.schema.names == build_schema().names
        assert set(rows.column("source_file").to_pylist()) == {
            str(p) for p in sample_files
        }
        # Every payload came through: nothing was truncated to metadata.
        assert all(
            size == len(payload)
            for size, payload in zip(
                rows.column("payload_size").to_pylist(),
                rows.column("payload").to_pylist(),
                strict=True,
            )
        )

    def test_topic_filter_reaches_the_table(
        self, sample_files: list[Path], db_dir: str
    ) -> None:
        result = ingest(sample_files, uri=db_dir, topics=["/imu"])
        assert result.complete
        assert result.rows_in_table == SAMPLE_PER_FILE["/imu"] * len(sample_files)
        rows = lancedb.connect(db_dir).open_table("messages")
        assert set(
            rows.search(None).limit(None).to_arrow().column("topic").to_pylist()
        ) == {"/imu"}

    def test_can_index_without_copying_payloads(
        self, sample_files: list[Path], db_dir: str
    ) -> None:
        result = ingest(sample_files[:1], uri=db_dir, include_payload=False)
        assert result.complete
        table = lancedb.connect(db_dir).open_table("messages")
        assert "payload" not in table.schema.names
        assert "payload_size" in table.schema.names

    def test_append_counts_only_what_it_added(
        self, sample_files: list[Path], db_dir: str
    ) -> None:
        ingest(sample_files[:1], uri=db_dir)
        result = ingest(sample_files[1:2], uri=db_dir, mode="append")
        assert result.rows_written == SAMPLE_FILE_ROWS
        assert result.rows_in_table == SAMPLE_FILE_ROWS * 2
        assert result.complete
        assert result.versions == 2

    def test_overwrite_replaces_the_table(
        self, sample_files: list[Path], db_dir: str
    ) -> None:
        ingest(sample_files, uri=db_dir)
        result = ingest(sample_files[:1], uri=db_dir, mode="overwrite")
        assert result.rows_in_table == SAMPLE_FILE_ROWS
        assert result.complete

    def test_a_custom_table_name(self, sample_files: list[Path], db_dir: str) -> None:
        ingest(sample_files[:1], uri=db_dir, table="robot_messages")
        assert (
            lancedb.connect(db_dir).open_table("robot_messages").count_rows()
            == SAMPLE_FILE_ROWS
        )

    def test_batch_size_does_not_change_the_result(
        self, sample_files: list[Path], db_dir: str, tmp_path: Path
    ) -> None:
        other = str(tmp_path / "other_db")
        first = ingest(sample_files[:1], uri=db_dir, batch_size=7)
        second = ingest(sample_files[:1], uri=other, batch_size=10_000)
        assert first.rows_in_table == second.rows_in_table == SAMPLE_FILE_ROWS

    def test_files_with_no_messages_leave_an_empty_table(
        self, tmp_path: Path, db_dir: str
    ) -> None:
        # Not an error: the schema is known even though no file had a row, so
        # the table stands empty rather than the run failing at the end.
        write_log(tmp_path / "a.mcap", [])
        write_log(tmp_path / "b.mcap", [])
        result = ingest(discover_files(tmp_path), uri=db_dir)
        assert result.expected_rows == 0
        assert result.rows_in_table == 0
        assert result.complete
        assert lancedb.connect(db_dir).open_table("messages").schema.names == (
            build_schema().names
        )

    def test_refuses_an_empty_file_list(self, db_dir: str) -> None:
        with pytest.raises(ValueError, match="no MCAP files"):
            ingest([], uri=db_dir)


class TestQueries:
    @pytest.fixture
    def ingested(self, sample_files: list[Path], db_dir: str) -> str:
        ingest(sample_files, uri=db_dir)
        return db_dir

    def test_topic_counts_match_the_recordings(self, ingested: str) -> None:
        counts = topic_counts(ingested)
        assert counts == {topic: count * 3 for topic, count in SAMPLE_PER_FILE.items()}

    def test_topic_counts_on_an_empty_table(self, db_dir: str) -> None:
        lancedb.connect(db_dir).create_table("messages", schema=build_schema())
        assert topic_counts(db_dir) == {}

    def test_time_slice_is_half_open(
        self, ingested: str, sample_files: list[Path]
    ) -> None:
        summary = summarize_file(sample_files[0])
        rows = time_slice(
            ingested,
            start_ns=summary.start_time_ns,
            end_ns=summary.end_time_ns,
        ).take_all()
        times = [int(row["log_time_ns"]) for row in rows]
        assert times, "expected the first recording's window to match rows"
        assert min(times) == summary.start_time_ns
        # end_ns is exclusive, so the last message of the file is outside it.
        assert max(times) < summary.end_time_ns

    def test_time_slice_can_filter_topics(
        self, ingested: str, sample_files: list[Path]
    ) -> None:
        summary = summarize_file(sample_files[0])
        rows = time_slice(
            ingested,
            start_ns=summary.start_time_ns,
            end_ns=summary.end_time_ns + 1,
            topics=["/odom", "/diagnostics"],
        ).take_all()
        assert {str(row["topic"]) for row in rows} == {"/odom", "/diagnostics"}
        assert len(rows) == SAMPLE_PER_FILE["/odom"] + SAMPLE_PER_FILE["/diagnostics"]

    def test_time_slice_quotes_topic_names(self, tmp_path: Path, db_dir: str) -> None:
        # A topic name is whatever the recorder called it, and a quote in one
        # would otherwise end the SQL literal early.
        path = write_log(
            tmp_path / "log.mcap",
            [
                Msg(topic="/it's", data=b"{}", log_time=1),
                Msg(topic="/other", data=b"{}", log_time=2),
            ],
        )
        ingest([path], uri=db_dir)
        rows = time_slice(
            db_dir, start_ns=0, end_ns=INT64_MAX, topics=["/it's"]
        ).take_all()
        assert [str(row["topic"]) for row in rows] == ["/it's"]

    def test_time_slice_projects_columns(self, ingested: str) -> None:
        ds = time_slice(
            ingested,
            start_ns=0,
            end_ns=INT64_MAX,
            columns=["topic", "log_time_ns"],
        )
        assert ds.schema().names == ["topic", "log_time_ns"]

    def test_a_window_outside_the_recording_is_empty(self, ingested: str) -> None:
        assert time_slice(ingested, start_ns=0, end_ns=1).count() == 0


class TestCli:
    @staticmethod
    def _run(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> None:
        """Run the ingest CLI without letting it tear down the test's Ray."""
        import ray

        monkeypatch.setattr("sys.argv", ["ingest_mcap.py", *argv])
        monkeypatch.setattr(ray, "shutdown", lambda: None)
        ingest_module.main()

    def test_end_to_end(
        self,
        sample_logs: Path,
        db_dir: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(monkeypatch, ["--logs", str(sample_logs), "--uri", db_dir])
        out = capsys.readouterr().out
        assert "All checks passed." in out
        assert f"rows written : {SAMPLE_FILE_ROWS * 3:,}" in out
        assert "versions     : 1" in out
        # The per-topic report is the part that reads one column of many.
        assert "/imu" in out and "/camera/front/compressed" in out
        assert lancedb.connect(db_dir).open_table("messages").count_rows() == (
            SAMPLE_FILE_ROWS * 3
        )

    def test_honours_its_flags(
        self,
        sample_logs: Path,
        db_dir: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(
            monkeypatch,
            [
                "--logs",
                str(sample_logs),
                "--uri",
                db_dir,
                "--table",
                "imu",
                "--topics",
                "/imu",
                "--batch-size",
                "64",
                "--no-payload",
            ],
        )
        assert "All checks passed." in capsys.readouterr().out
        table = lancedb.connect(db_dir).open_table("imu")
        assert table.count_rows() == SAMPLE_PER_FILE["/imu"] * 3
        assert "payload" not in table.schema.names

    def test_appending_skips_the_single_version_check(
        self,
        sample_logs: Path,
        db_dir: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(monkeypatch, ["--logs", str(sample_logs), "--uri", db_dir])
        capsys.readouterr()
        self._run(
            monkeypatch,
            ["--logs", str(sample_logs), "--uri", db_dir, "--mode", "append"],
        )
        out = capsys.readouterr().out
        assert "All checks passed." in out
        assert lancedb.connect(db_dir).open_table("messages").count_rows() == (
            SAMPLE_FILE_ROWS * 6
        )

    def test_a_single_file_argument(
        self,
        sample_files: list[Path],
        db_dir: str,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        self._run(monkeypatch, ["--logs", str(sample_files[0]), "--uri", db_dir])
        assert "Found 1 MCAP file(s)" in capsys.readouterr().out

    def test_a_directory_with_no_logs_exits(
        self, tmp_path: Path, db_dir: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        with pytest.raises(SystemExit, match="No .mcap files"):
            self._run(monkeypatch, ["--logs", str(tmp_path), "--uri", db_dir])


class TestHelpers:
    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (0, "0.0 B"),
            (512, "512.0 B"),
            (2048, "2.0 KB"),
            (5 * 1024**2, "5.0 MB"),
            (3 * 1024**3, "3.0 GB"),
            (4 * 1024**4, "4096.0 GB"),
        ],
    )
    def test_byte_formatting(self, count: int, expected: str) -> None:
        assert ingest_module._fmt_bytes(count) == expected

    def test_table_bytes_counts_the_dataset(
        self, sample_files: list[Path], db_dir: str
    ) -> None:
        ingest(sample_files[:1], uri=db_dir)
        assert ingest_module._table_bytes(db_dir, "messages") > 0

    def test_row_count_of_a_missing_table_is_zero(self, db_dir: str) -> None:
        # The append path asks before the table exists, so this is the first
        # thing an ingest into a fresh directory does.
        assert ingest_module._row_count(db_dir, "messages") == 0

    def test_row_count_of_an_existing_table(
        self, sample_files: list[Path], db_dir: str
    ) -> None:
        ingest(sample_files[:1], uri=db_dir)
        assert ingest_module._row_count(db_dir, "messages") == SAMPLE_FILE_ROWS


def test_module_exports_are_importable() -> None:
    # The example is copied by people who will import from it; keep the names
    # it advertises stable.
    for name in ("build_schema", "iter_record_batches", "summarize_file"):
        assert callable(getattr(mcap_source, name))
