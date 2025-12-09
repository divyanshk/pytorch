# mypy: allow-untyped-defs
"""Type definitions and protocols for the data loading module."""

from typing import Any, Optional, Protocol


class _BaseQueue(Protocol):
    """Protocol for basic queue operations shared by all queue types.

    This protocol is satisfied by both `queue.SimpleQueue` (used in threading)
    and `multiprocessing.Queue` (used in multiprocessing).
    """

    def put(
        self,
        item: Any,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> None: ...

    def get(
        self,
        block: bool = True,
        timeout: Optional[float] = None,
    ) -> Any: ...


class _MPQueue(_BaseQueue, Protocol):
    """Protocol for multiprocessing queues with additional cleanup methods.

    Extends `_BaseQueue` with `cancel_join_thread()` and `close()` methods
    that are specific to `multiprocessing.Queue`.
    """

    def cancel_join_thread(self) -> None: ...

    def close(self) -> None: ...
