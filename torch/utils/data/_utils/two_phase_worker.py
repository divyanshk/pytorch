# mypy: allow-untyped-defs
r"""Two-phase hybrid worker implementation for DataLoader.

This module implements a pipelined approach to data loading with two phases:
1. Fetch Phase: Workers in a separate process fetch raw data from the dataset
2. Transform Phase: Workers (threads) in main process apply collate_fn and pin memory

This separation allows for better parallelization when transforms are compute-intensive.
"""

import multiprocessing
import queue
import threading

import torch
from torch._utils import ExceptionWrapper

from . import STATUS_CHECK_INTERVAL
from .worker import _IterableDatasetStopIteration, _ResumeIteration


def _fetch_worker_thread(
    dataset_kind,
    dataset,
    index_queue,
    raw_data_queue,
    done_event,
    auto_collation,
    drop_last,
    init_fn,
    worker_id,
    num_workers,
) -> None:
    """
    Fetch worker thread that runs inside the fetch process.

    Reads indices from index_queue, fetches raw data points from the dataset,
    and puts them into raw_data_queue without applying any collation or transforms.

    Args:
        dataset_kind: Type of dataset (Map or Iterable)
        dataset: The dataset object
        index_queue: Queue to receive indices
        raw_data_queue: Queue to send raw data
        done_event: Event to signal shutdown
        auto_collation: Whether auto-collation is enabled
        drop_last: Whether to drop incomplete batches
        init_fn: Worker initialization function
        worker_id: ID of this worker thread
        num_workers: Total number of fetch workers
    """
    try:
        torch.set_num_threads(1)

        from torch.utils.data import _DatasetKind

        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)
        except Exception:
            init_exception = ExceptionWrapper(
                where=f"in DataLoader fetch worker thread {worker_id}"
            )

        iteration_end = False
        dataset_iter = None  # type: ignore[assignment]

        # Main fetch loop
        while True:
            try:
                r = index_queue.get(timeout=STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue

            if isinstance(r, _ResumeIteration):
                # Acknowledge the main process
                raw_data_queue.put((r, None))
                iteration_end = False
                dataset_iter = None  # type: ignore[assignment]
                continue
            elif r is None:
                # Received the final signal
                break
            elif done_event.is_set() or iteration_end:
                # Skip processing if shutting down
                continue

            idx, index = r

            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    # Fetch raw data without collation
                    if auto_collation:
                        if dataset_kind == _DatasetKind.Map:
                            # Fetch individual data points
                            if (
                                hasattr(dataset, "__getitems__")
                                and dataset.__getitems__
                            ):
                                raw_data = dataset.__getitems__(index)
                            else:
                                raw_data = [dataset[i] for i in index]
                            data = raw_data
                        else:
                            # Iterable dataset
                            if dataset_iter is None:
                                dataset_iter = iter(dataset)
                            raw_data_list = []
                            for _ in index:
                                try:
                                    raw_data_list.append(next(dataset_iter))
                                except StopIteration:
                                    iteration_end = True
                                    break
                            if len(raw_data_list) == 0 or (
                                drop_last and len(raw_data_list) < len(index)
                            ):
                                raise StopIteration
                            data = raw_data_list
                    else:
                        data = dataset[index]

                except Exception as e:
                    if (
                        isinstance(e, StopIteration)
                        and dataset_kind == _DatasetKind.Iterable
                    ):
                        data = _IterableDatasetStopIteration(worker_id)
                        iteration_end = True
                    else:
                        data = ExceptionWrapper(
                            where=f"in DataLoader fetch worker thread {worker_id}"
                        )

            # # Collate and put into the queue
            # from torch.utils.data._utils.collate import default_collate
            # from . import pin_memory as pin_memory_module

            # raw_data_queue.put(
            #     (idx, pin_memory_module.pin_memory(default_collate(data)))
            # )

            raw_data_queue.put((idx, data))
            del data, idx, index, r

    except KeyboardInterrupt:
        pass


def _fetch_process_main(
    dataset_kind,
    dataset,
    index_queue,
    raw_data_queue,
    done_event,
    auto_collation,
    drop_last,
    init_fn,
    num_fetch_workers,
):
    """
    Main function for the fetch process.

    This creates multiple fetch worker threads within the process.

    Args:
        dataset_kind: Type of dataset
        dataset: The dataset object
        index_queue: Multiprocessing queue for receiving indices
        raw_data_queue: Multiprocessing queue for sending raw data
        done_event: Multiprocessing event for shutdown signaling
        auto_collation: Whether auto-collation is enabled
        drop_last: Whether to drop incomplete batches
        init_fn: Worker initialization function
        num_fetch_workers: Number of fetch worker threads to create
    """
    try:
        # Create fetch worker threads inside this process
        fetch_threads = []
        for i in range(num_fetch_workers):
            t = threading.Thread(
                target=_fetch_worker_thread,
                args=(
                    dataset_kind,
                    dataset,
                    index_queue,
                    raw_data_queue,
                    done_event,
                    auto_collation,
                    drop_last,
                    init_fn,
                    i,
                    num_fetch_workers,
                ),
                daemon=True,
            )
            t.start()
            fetch_threads.append(t)

        # Wait for all threads to complete
        for t in fetch_threads:
            t.join()

    except KeyboardInterrupt:
        pass


def _transform_worker_loop(
    raw_data_queue,
    result_queue,
    done_event,
    collate_fn,
    worker_id,
    pin_memory=False,
) -> None:
    """
    Phase 2: Transform worker loop that applies collation and pinning.

    This worker reads raw data from raw_data_queue, applies the collate_fn
    (which contains transforms), optionally pins memory, and puts the result
    into the result_queue.

    Args:
        raw_data_queue: Queue to receive raw data from fetch process
        result_queue: Queue to send transformed data to main process
        done_event: Event to signal shutdown
        collate_fn: Collation function (includes transforms)
        worker_id: ID of this worker
        pin_memory: Whether to pin memory after transformation
    """
    try:
        torch.set_num_threads(1)

        # Set thread name for debugging
        threading.current_thread().name = f"DataLoader_transform_thread_{worker_id}"

        # Main transform loop
        while not done_event.is_set():
            try:
                r = raw_data_queue.get(timeout=STATUS_CHECK_INTERVAL)
            except queue.Empty:
                # Check if we should exit
                if done_event.is_set():
                    break
                continue
            except Exception:
                # Handle any other exceptions (e.g., queue closed, deserialization errors)
                if done_event.is_set():
                    break
                continue

            if isinstance(r, _ResumeIteration):
                # Acknowledge and forward to result queue
                result_queue.put((r, None))
                continue
            elif r is None:
                # Received the final signal
                break
            elif done_event.is_set():
                # Skip processing if shutting down
                break

            idx, data = r

            # Check if data is already an exception or stop iteration
            if isinstance(data, (ExceptionWrapper, _IterableDatasetStopIteration)):
                result_queue.put((idx, data))
                continue

            # Apply collate function (which includes transforms)
            try:
                transformed_data = collate_fn(data)

                # Pin memory if enabled
                if pin_memory and not isinstance(transformed_data, ExceptionWrapper):
                    try:
                        from . import pin_memory as pin_memory_module

                        transformed_data = pin_memory_module.pin_memory(
                            transformed_data
                        )
                    except Exception:
                        transformed_data = ExceptionWrapper(
                            where=f"in pin_memory for DataLoader transform worker thread {worker_id}"
                        )
            except Exception:
                transformed_data = ExceptionWrapper(
                    where=f"in DataLoader transform worker thread {worker_id}"
                )

            result_queue.put((idx, transformed_data))
            del transformed_data, data, idx, r
            # result_queue.put((idx, data))
            # del data, idx, r

    except KeyboardInterrupt:
        pass


class TwoPhaseHybridDataLoaderIter:
    """
    DataLoader iterator with two-phase hybrid workers for pipelined data loading.

    This iterator creates:
    1. Fetch process: A separate process containing multiple fetch worker threads
    2. Transform workers: Multiple threads in main process

    The architecture is:
        Main Process: index_queue -> [Fetch Process (threads)] -> raw_data_queue -> [Transform Threads] -> result_queue

    This allows better parallelization when transforms/collations are compute-intensive,
    as data fetching (in separate process) and transformation can happen concurrently.

    Args:
        dataset_kind: Type of dataset (Map or Iterable)
        dataset: The dataset object
        auto_collation: Whether auto-collation is enabled
        collate_fn: Collation function (includes transforms)
        drop_last: Whether to drop incomplete batches
        init_fn: Worker initialization function
        num_fetch_workers: Number of fetch worker threads in the fetch process (default: 2)
        num_transform_workers: Number of transform worker threads (default: 2)
        pin_memory: Whether to pin memory after transformation
        raw_data_queue_size: Size of the intermediate queue (default: 10)
        multiprocessing_context: Multiprocessing context to use
    """

    def __init__(
        self,
        dataset_kind,
        dataset,
        auto_collation,
        collate_fn,
        drop_last,
        init_fn=None,
        num_fetch_workers=2,
        num_transform_workers=2,
        pin_memory=False,
        raw_data_queue_size=1000,
        multiprocessing_context=None,
    ):
        self._dataset_kind = dataset_kind
        self._dataset = dataset
        self._auto_collation = auto_collation
        self._collate_fn = collate_fn
        self._drop_last = drop_last
        self._init_fn = init_fn
        self._num_fetch_workers = num_fetch_workers
        self._num_transform_workers = num_transform_workers
        self._pin_memory = pin_memory
        self._shutdown = False

        # Determine multiprocessing context
        if multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        self._mp_context = multiprocessing_context

        # Create multiprocessing queues and events for IPC
        self._index_queue = self._mp_context.Queue()
        self._raw_data_queue = self._mp_context.Queue(maxsize=raw_data_queue_size)
        self._fetch_done_event = self._mp_context.Event()

        # Create threading queue and event for transform workers
        self._result_queue = queue.SimpleQueue()
        self._transform_done_event = threading.Event()

        # Start the fetch process
        self._fetch_process = self._mp_context.Process(
            target=_fetch_process_main,
            args=(
                dataset_kind,
                dataset,
                self._index_queue,
                self._raw_data_queue,
                self._fetch_done_event,
                auto_collation,
                drop_last,
                init_fn,
                num_fetch_workers,
            ),
            daemon=True,
        )
        self._fetch_process.start()

        # Create transform worker threads in main process
        self._transform_workers = []
        for i in range(num_transform_workers):
            w = threading.Thread(
                target=_transform_worker_loop,
                args=(
                    self._raw_data_queue,
                    self._result_queue,
                    self._transform_done_event,
                    collate_fn,
                    i,
                    pin_memory,
                ),
                daemon=True,
            )
            w.start()
            self._transform_workers.append(w)

    def put_indices(self, idx, indices):
        """
        Put indices into the index queue for fetch workers to process.

        Args:
            idx: Task index
            indices: Batch indices to fetch
        """
        if not self._shutdown:
            self._index_queue.put((idx, indices))

    def get_result(self, timeout=STATUS_CHECK_INTERVAL):
        """
        Get a result from the result queue.

        Args:
            timeout: Timeout for queue.get

        Returns:
            Tuple of (success, data) where success is a boolean and data is the result
        """
        try:
            data = self._result_queue.get(timeout=timeout)
            return (True, data)
        except queue.Empty:
            return (False, None)

    def shutdown(self):
        """
        Shutdown all workers gracefully.
        """
        if not self._shutdown:
            self._shutdown = True

            # First, shutdown transform workers (they depend on fetch process)
            self._transform_done_event.set()

            # Drain the raw_data_queue to unblock any waiting transform workers
            # This prevents them from hanging on queue.get()
            try:
                while not self._raw_data_queue.empty():
                    try:
                        self._raw_data_queue.get_nowait()
                    except:
                        break
            except:
                pass

            # Send None signals to transform workers
            for _ in range(self._num_transform_workers):
                try:
                    self._raw_data_queue.put(None, timeout=0.1)
                except:
                    pass

            # Wait for transform workers to finish
            for w in self._transform_workers:
                w.join(timeout=STATUS_CHECK_INTERVAL)

            # Now shutdown fetch process
            self._fetch_done_event.set()
            for _ in range(self._num_fetch_workers):
                try:
                    self._index_queue.put(None, timeout=0.1)
                except:
                    pass

            # Wait for fetch process to finish
            self._fetch_process.join(timeout=5.0)
            if self._fetch_process.is_alive():
                self._fetch_process.terminate()

            # Close queues
            try:
                self._index_queue.close()
            except:
                pass
            try:
                self._raw_data_queue.close()
            except:
                pass

    def __del__(self):
        """Cleanup when iterator is deleted."""
        self.shutdown()
