# SPDX-License-Identifier: Apache-2.0
"""Turn MCAP log files into Arrow batches, one file at a time, streaming.

`MCAP <https://mcap.dev>`_ is the container format robotics stacks record into:
one file holds many *channels* (topics), each carrying timestamped messages in
whatever encoding that channel declared -- JSON here, CDR/protobuf on a real
robot. Messages from every channel are interleaved by time and the whole thing
is chunked and compressed.

Two properties of that shape drive everything below:

* **A file is not splittable.** Chunks are compressed and the message index
  lives in a summary section at the end, so there is no byte range a second
  reader could start from. The unit of parallelism is therefore the *file*.
* **A file does not fit in memory.** A recording is routinely tens of GB, most
  of it camera payloads. So this module never returns a table -- it yields
  :class:`pyarrow.RecordBatch` objects as it reads, and the caller decides how
  many are in flight.

The schema is deliberately one flat table for all channels rather than one
table per topic: the query these logs get asked first is "which messages, on
any topic, came from this window", and that is a filter, not a join.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Optional, Union

import pyarrow as pa
from mcap.reader import make_reader
from mcap.records import Channel, Message, Schema
from mcap.well_known import MessageEncoding

#: Rows per emitted batch. 8,192 rows of metadata is a few hundred KB; with
#: payloads it is bounded by the payloads themselves, so a log full of camera
#: frames wants a smaller number.
DEFAULT_BATCH_SIZE = 8192

#: Arrow has no unsigned 64-bit timestamp: ``timestamp('ns')`` and ``int64``
#: both top out here. MCAP timestamps are uint64, so the top half of that range
#: is representable in a log and not in the table.
INT64_MAX = 2**63 - 1


def build_schema(*, include_payload: bool = True) -> pa.Schema:
    """The table every batch conforms to.

    Declared explicitly rather than inferred, for the reason schemas usually
    are in this library: the table is created before the first message has been
    read, and a batch that happens to contain only JSON channels must not
    produce a different schema from one that also carries binary payloads.

    ``json_payload`` is the message text for JSON channels and null everywhere
    else; ``payload`` is the raw bytes for every channel. Splitting them is what
    lets a metadata query read neither -- Lance reads only the columns a query
    projects, so counting messages per topic never touches the blobs.
    """
    fields = [
        pa.field("source_file", pa.string(), nullable=False),
        pa.field("topic", pa.string(), nullable=False),
        # Empty string, not null, when the channel declared no schema: MCAP
        # says schema_id 0 means "no schema", which is a fact about the
        # channel rather than a missing value.
        pa.field("schema_name", pa.string(), nullable=False),
        pa.field("message_encoding", pa.string(), nullable=False),
        # uint32 on the wire, widened: Arrow's unsigned types survive the round
        # trip but make every downstream comparison a signed/unsigned question.
        pa.field("sequence", pa.int64(), nullable=False),
        # Naive, deliberately. MCAP timestamps are "nanoseconds since a
        # user-understood epoch" -- usually the Unix epoch, but a robot that
        # boots without a clock logs since boot instead. Stamping UTC on the
        # second case would be a lie, so the tz-aware reading is left to
        # whoever knows which it is.
        pa.field("log_time", pa.timestamp("ns"), nullable=False),
        # The raw value, kept alongside it: it is exact, it is what the MCAP
        # index is keyed on, and an integer range filter needs no timestamp
        # literal syntax to write.
        pa.field("log_time_ns", pa.int64(), nullable=False),
        pa.field("publish_time_ns", pa.int64(), nullable=False),
        pa.field("payload_size", pa.int64(), nullable=False),
        pa.field("json_payload", pa.string(), nullable=True),
    ]
    if include_payload:
        fields.append(pa.field("payload", pa.large_binary(), nullable=True))
    return pa.schema(fields)


@dataclass(frozen=True)
class ChannelSummary:
    """One channel in a file, and how many messages it holds."""

    topic: str
    schema_name: str
    message_encoding: str
    message_count: int


@dataclass(frozen=True)
class FileSummary:
    """What a file contains, read from its summary section where possible."""

    path: str
    message_count: int
    start_time_ns: int
    end_time_ns: int
    channels: tuple[ChannelSummary, ...]
    #: True when the counts came from a full scan because the file carried no
    #: statistics record -- worth knowing, because that scan is not free.
    scanned: bool = False

    def matching_count(self, topics: Optional[Sequence[str]] = None) -> int:
        """Messages this file holds on ``topics`` (all of them when None)."""
        if topics is None:
            return self.message_count
        wanted = set(topics)
        return sum(c.message_count for c in self.channels if c.topic in wanted)


def _checked_ns(value: int, *, field: str, path: str, topic: str) -> int:
    """Reject a timestamp that Arrow cannot hold rather than wrapping it."""
    if value > INT64_MAX:
        raise ValueError(
            f"{field}={value} on topic {topic!r} in {path} exceeds int64; "
            "MCAP timestamps are uint64 and Arrow's are signed, so this value "
            "cannot be stored. Rebase the log's clock before ingesting it."
        )
    return value


def _json_text(channel: Channel, data: bytes) -> Optional[str]:
    """The message as text, for JSON channels only.

    No ``json.loads``: the point of the column is to carry the message through
    for whoever wants to parse it, and a channel whose bytes do not parse is
    still data -- the raw payload column keeps it either way.
    """
    if channel.message_encoding != MessageEncoding.JSON:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        # A JSON channel is UTF-8 by definition, so this is a corrupt or
        # mislabelled message. It stays in the table with its bytes intact.
        return None


class _BatchBuilder:
    """Accumulates messages column-wise and hands back Arrow batches."""

    def __init__(self, source_file: str, *, include_payload: bool) -> None:
        self._source_file = source_file
        self._include_payload = include_payload
        self._schema = build_schema(include_payload=include_payload)
        self._reset()

    def _reset(self) -> None:
        self._topic: list[str] = []
        self._schema_name: list[str] = []
        self._encoding: list[str] = []
        self._sequence: list[int] = []
        self._log_time: list[int] = []
        self._publish_time: list[int] = []
        self._payload_size: list[int] = []
        self._json: list[Optional[str]] = []
        self._payload: list[Optional[bytes]] = []

    def __len__(self) -> int:
        return len(self._topic)

    def append(
        self, schema: Optional[Schema], channel: Channel, message: Message
    ) -> None:
        """Add one MCAP message."""
        self._topic.append(channel.topic)
        self._schema_name.append(schema.name if schema is not None else "")
        self._encoding.append(channel.message_encoding)
        self._sequence.append(message.sequence)
        self._log_time.append(
            _checked_ns(
                message.log_time,
                field="log_time",
                path=self._source_file,
                topic=channel.topic,
            )
        )
        self._publish_time.append(
            _checked_ns(
                message.publish_time,
                field="publish_time",
                path=self._source_file,
                topic=channel.topic,
            )
        )
        self._payload_size.append(len(message.data))
        self._json.append(_json_text(channel, message.data))
        if self._include_payload:
            self._payload.append(message.data)

    def flush(self) -> pa.RecordBatch:
        """Emit what has accumulated and start a new batch."""
        rows = len(self)
        log_time = pa.array(self._log_time, pa.int64())
        columns: list[pa.Array] = [
            pa.array([self._source_file] * rows, pa.string()),
            pa.array(self._topic, pa.string()),
            pa.array(self._schema_name, pa.string()),
            pa.array(self._encoding, pa.string()),
            pa.array(self._sequence, pa.int64()),
            # Cast rather than construct: the values are already exact
            # nanoseconds, and casting cannot reinterpret them as anything else.
            log_time.cast(pa.timestamp("ns")),
            log_time,
            pa.array(self._publish_time, pa.int64()),
            pa.array(self._payload_size, pa.int64()),
            pa.array(self._json, pa.string()),
        ]
        if self._include_payload:
            columns.append(pa.array(self._payload, pa.large_binary()))
        self._reset()
        return pa.RecordBatch.from_arrays(columns, schema=self._schema)


def empty_batch(*, include_payload: bool = True) -> pa.RecordBatch:
    """A zero-row batch carrying the schema.

    A file that matched no messages still has to announce a schema, or a Ray
    pipeline whose first file is empty has no schema to write with.
    """
    schema = build_schema(include_payload=include_payload)
    return pa.RecordBatch.from_pylist([], schema=schema)


def iter_record_batches(
    path: Union[str, Path],
    *,
    topics: Optional[Sequence[str]] = None,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    include_payload: bool = True,
) -> Iterator[pa.RecordBatch]:
    """Stream one MCAP file out as Arrow batches, in log-time order.

    ``topics``, ``start_time`` and ``end_time`` are pushed down into the MCAP
    reader, which uses the file's chunk index to skip chunks that cannot
    contain a match -- so a narrow query decompresses a fraction of the file
    rather than all of it and filtering afterwards.

    :param path: the ``.mcap`` file to read.
    :param topics: channels to read, or None for every channel.
    :param start_time: inclusive lower bound on ``log_time``, in nanoseconds.
    :param end_time: exclusive upper bound on ``log_time``, in nanoseconds.
    :param batch_size: maximum rows per emitted batch.
    :param include_payload: keep the raw message bytes. False drops the
        ``payload`` column entirely, which is the difference between indexing a
        log and copying it.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    source = str(path)
    builder = _BatchBuilder(source, include_payload=include_payload)
    with Path(path).open("rb") as stream:
        reader = make_reader(stream)
        for schema, channel, message in reader.iter_messages(
            topics=topics, start_time=start_time, end_time=end_time
        ):
            builder.append(schema, channel, message)
            if len(builder) >= batch_size:
                yield builder.flush()
    if len(builder):
        yield builder.flush()


def summarize_file(path: Union[str, Path]) -> FileSummary:
    """Read what a file contains without decoding its messages.

    An indexed MCAP carries a statistics record in its summary section, so this
    is a seek and a handful of reads no matter how large the file is. A file
    written without one -- a recording that was cut off, or a writer with
    indexing disabled -- has to be scanned instead, which is why the result
    says which happened.
    """
    source = str(path)
    with Path(path).open("rb") as stream:
        reader = make_reader(stream)
        summary = reader.get_summary()
        if summary is not None and summary.statistics is not None:
            statistics = summary.statistics
            channels = tuple(
                ChannelSummary(
                    topic=channel.topic,
                    schema_name=(
                        summary.schemas[channel.schema_id].name
                        if channel.schema_id in summary.schemas
                        else ""
                    ),
                    message_encoding=channel.message_encoding,
                    message_count=statistics.channel_message_counts.get(channel_id, 0),
                )
                for channel_id, channel in sorted(summary.channels.items())
            )
            return FileSummary(
                path=source,
                message_count=statistics.message_count,
                start_time_ns=statistics.message_start_time,
                end_time_ns=statistics.message_end_time,
                channels=channels,
            )
        return _scan_summary(source, stream)


def _scan_summary(source: str, stream: IO[bytes]) -> FileSummary:
    """Fall back to counting messages, for a file with no statistics record."""
    stream.seek(0)
    counts: dict[str, int] = {}
    seen: dict[str, ChannelSummary] = {}
    total = 0
    start = 0
    end = 0
    for schema, channel, message in make_reader(stream).iter_messages():
        key = channel.topic
        counts[key] = counts.get(key, 0) + 1
        if key not in seen:
            seen[key] = ChannelSummary(
                topic=channel.topic,
                schema_name=schema.name if schema is not None else "",
                message_encoding=channel.message_encoding,
                message_count=0,
            )
        start = message.log_time if total == 0 else min(start, message.log_time)
        end = max(end, message.log_time)
        total += 1
    channels = tuple(
        ChannelSummary(
            topic=info.topic,
            schema_name=info.schema_name,
            message_encoding=info.message_encoding,
            message_count=counts[topic],
        )
        for topic, info in sorted(seen.items())
    )
    return FileSummary(
        path=source,
        message_count=total,
        start_time_ns=start,
        end_time_ns=end,
        channels=channels,
        scanned=True,
    )


def discover_files(root: Union[str, Path]) -> list[Path]:
    """Every ``.mcap`` file under ``root``, sorted, or the file itself."""
    path = Path(root).expanduser()
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*.mcap") if p.is_file())
