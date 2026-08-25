# SPDX-License-Identifier: Apache-2.0
"""lancedb-ray: Ray Data integration for LanceDB and LanceDB Enterprise.

Read and write LanceDB tables as Ray Datasets, using the most parallel strategy
each backend supports:

- **Local / OSS** tables are backed by a Lance dataset, so reads run one task
  per fragment group and appends write fragments in parallel that the driver
  commits as a single atomic transaction.
- **Cloud / Enterprise** (``db://``) tables are a remote service with no
  fragment access, so reads shard the row space across tasks against a pinned
  table version and writes fan out batched requests.

Example:
    >>> import lancedb_ray as ldbr
    >>> ds = ldbr.read_lancedb("my_table", uri="/data/lancedb")  # doctest: +SKIP
    >>> ldbr.write_lancedb(ds, "copy", uri="/data/lancedb", mode="create")  # doctest: +SKIP
"""

from ._plan import OffsetRange
from ._retry import RetryPolicy
from .connection import LanceDBConnectionSpec
from .datasink import LanceDBDatasink, WriteStats
from .datasource import LanceDBDatasource
from .io import read_lancedb, write_lancedb

__version__ = "0.1.0"

__all__ = [
    "LanceDBConnectionSpec",
    "LanceDBDatasink",
    "LanceDBDatasource",
    "OffsetRange",
    "RetryPolicy",
    "WriteStats",
    "__version__",
    "read_lancedb",
    "write_lancedb",
]
