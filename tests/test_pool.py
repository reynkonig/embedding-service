import os
import threading
import time
import unittest
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import ClassVar
from unittest.mock import patch

from embedding_service.engine import EmbeddingServiceError, EngineResult
from embedding_service.pool import (
    EmbeddingBusyError,
    EmbeddingProcessPool,
    WorkerSnapshot,
    _worker_device,
)
from tests.helpers import make_settings


class ActivityController:
    def __init__(self, *, blocked: bool = False) -> None:
        self.condition = threading.Condition()
        self.release = threading.Event()
        if not blocked:
            self.release.set()
        self.active = 0
        self.max_active = 0
        self.entries: list[int] = []

    def enter(self, worker_index: int) -> None:
        with self.condition:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.entries.append(worker_index)
            self.condition.notify_all()

    def leave(self) -> None:
        with self.condition:
            self.active -= 1
            self.condition.notify_all()

    def wait_for_entries(self, count: int, timeout: float = 2.0) -> bool:
        with self.condition:
            return self.condition.wait_for(
                lambda: len(self.entries) >= count,
                timeout=timeout,
            )


class FakeReplica:
    instances: ClassVar[list["FakeReplica"]] = []
    controller = ActivityController()
    load_lock = threading.Lock()
    load_active = 0
    max_load_active = 0
    load_order: ClassVar[list[int]] = []
    load_delay = 0.0

    @classmethod
    def reset(cls, controller: ActivityController) -> None:
        cls.instances = []
        cls.controller = controller
        cls.load_active = 0
        cls.max_load_active = 0
        cls.load_order = []
        cls.load_delay = 0.0

    def __init__(self, settings, device, index, context) -> None:
        self.settings = settings
        self.device = device
        self.index = index
        self.context = context
        self.state = "cold"
        self.is_loaded = False
        self.last_error: str | None = None
        self.start_calls = 0
        self.stop_calls = 0
        self.restart_calls = 0
        self.load_calls = 0
        self.embed_calls: list[list[str]] = []
        self.active_calls = 0
        self.max_active_calls = 0
        self.fail_embed_count = 0
        self.fail_load = False
        self.restarted = threading.Event()
        type(self).instances.append(self)

    def start(self) -> None:
        self.start_calls += 1
        if self.state == "stopped":
            self.state = "cold"

    def load(self) -> None:
        self.load_calls += 1
        with type(self).load_lock:
            type(self).load_active += 1
            type(self).max_load_active = max(
                type(self).max_load_active,
                type(self).load_active,
            )
            type(self).load_order.append(self.index)
        try:
            if type(self).load_delay:
                time.sleep(type(self).load_delay)
            if self.fail_load:
                self.state = "failed"
                self.last_error = "LoadFailure"
                raise EmbeddingServiceError(f"cannot load {self.device}")
            self.is_loaded = True
            self.state = "ready"
            self.last_error = None
        finally:
            with type(self).load_lock:
                type(self).load_active -= 1

    def embed(self, texts) -> EngineResult:
        values = list(texts)
        self.embed_calls.append(values)
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        self.state = "busy"
        type(self).controller.enter(self.index)
        try:
            if not type(self).controller.release.wait(timeout=3):
                raise AssertionError("test did not release a blocked fake replica")
            if self.fail_embed_count:
                self.fail_embed_count -= 1
                self.state = "failed"
                self.last_error = "InferenceFailure"
                raise EmbeddingServiceError(f"inference failed on {self.device}")
            self.is_loaded = True
            self.state = "ready"
            self.last_error = None
            return EngineResult(
                embeddings=[[float(self.index + 1)] for _ in values],
                prompt_tokens=len(values),
            )
        finally:
            self.active_calls -= 1
            type(self).controller.leave()

    def restart(self) -> None:
        self.restart_calls += 1
        self.state = "cold"
        self.is_loaded = False
        self.last_error = None
        self.restarted.set()

    def stop(self) -> None:
        self.stop_calls += 1
        self.state = "stopped"
        self.is_loaded = False

    def snapshot(self) -> WorkerSnapshot:
        return WorkerSnapshot(
            index=self.index,
            device=self.device,
            pid=10_000 + self.index,
            state=self.state,
            loaded=self.is_loaded,
            last_error=self.last_error,
        )


@contextmanager
def fake_pool(
    settings,
    *,
    controller: ActivityController | None = None,
) -> Iterator[EmbeddingProcessPool]:
    controller = controller or ActivityController()
    FakeReplica.reset(controller)
    with patch("embedding_service.pool.ReplicaProcess", FakeReplica):
        pool = EmbeddingProcessPool(settings, context=object())
    try:
        yield pool
    finally:
        controller.release.set()
        pool.close()


def wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


class ReplicaPoolConstructionTest(unittest.TestCase):
    def test_creates_exactly_one_replica_per_configured_device(self) -> None:
        settings = make_settings(devices=("cuda:0", "cuda:3", "cpu"))

        with fake_pool(settings) as pool:
            self.assertEqual(
                [(worker.index, worker.device) for worker in pool.workers],
                [(0, "cuda:0"), (1, "cuda:3"), (2, "cpu")],
            )
            self.assertEqual(pool.snapshot()["worker_count"], 3)
            self.assertEqual(pool.snapshot()["available_workers"], 3)

    def test_device_map_uses_one_sharded_replica(self) -> None:
        settings = make_settings(device_map="balanced")

        with fake_pool(settings) as pool:
            self.assertEqual(len(pool.workers), 1)
            self.assertEqual(pool.workers[0].device, "device-map")
            self.assertEqual(pool.snapshot()["mode"], "sharded")

    def test_start_and_close_are_idempotent(self) -> None:
        settings = make_settings(devices=("cuda:0", "cuda:1"))

        with fake_pool(settings) as pool:
            pool.start()
            pool.start()
            workers = list(pool.workers)
            self.assertEqual([worker.start_calls for worker in workers], [1, 1])
            pool.close()
            pool.close()
            self.assertEqual([worker.stop_calls for worker in workers], [1, 1])

    def test_cuda_worker_isolated_as_logical_device_zero(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_worker_device("cuda:7", sharded=False), "cuda:0")
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "7")

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_worker_device("cpu", sharded=False), "cpu")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)
            self.assertEqual(_worker_device("cuda:7", sharded=True), "auto")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)


class ReplicaPoolSchedulingTest(unittest.TestCase):
    def test_concurrent_first_requests_load_only_one_replica_at_a_time(self) -> None:
        settings = make_settings(devices=("cuda:0", "cuda:1"))

        with fake_pool(settings) as pool:
            FakeReplica.load_delay = 0.03
            with ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(pool.embed, (["first"], ["second"])))

            self.assertEqual(len(results), 2)
            self.assertEqual(FakeReplica.max_load_active, 1)
            self.assertEqual(sum(worker.load_calls for worker in pool.workers), 2)

    def test_two_replicas_run_two_requests_concurrently_and_queue_the_third(
        self,
    ) -> None:
        controller = ActivityController(blocked=True)
        settings = make_settings(
            devices=("cuda:0", "cuda:1"),
            max_pending_requests=2,
            queue_timeout_seconds=2,
        )

        with fake_pool(settings, controller=controller) as pool:
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [
                    executor.submit(pool.embed, [f"request-{index}"])
                    for index in range(3)
                ]
                self.assertTrue(controller.wait_for_entries(2))
                self.assertEqual(len(controller.entries), 2)
                self.assertTrue(any(not future.done() for future in futures))
                self.assertEqual(controller.max_active, 2)

                controller.release.set()
                results = [future.result(timeout=2) for future in futures]

            self.assertEqual([result.prompt_tokens for result in results], [1, 1, 1])
            self.assertEqual(controller.max_active, 2)
            self.assertEqual(
                [worker.max_active_calls for worker in pool.workers],
                [1, 1],
            )
            snapshot = pool.snapshot()
            self.assertEqual(snapshot["active_requests"], 0)
            self.assertEqual(snapshot["available_workers"], 2)
            self.assertEqual(snapshot["completed_requests"], 3)
            self.assertEqual(snapshot["failed_requests"], 0)

    def test_bounded_capacity_rejects_a_request_beyond_active_and_pending(self) -> None:
        controller = ActivityController(blocked=True)
        settings = make_settings(
            devices=("cuda:0",),
            max_pending_requests=1,
            queue_timeout_seconds=2,
        )

        with fake_pool(settings, controller=controller) as pool:
            with ThreadPoolExecutor(max_workers=2) as executor:
                active = executor.submit(pool.embed, ["active"])
                self.assertTrue(controller.wait_for_entries(1))
                pending = executor.submit(pool.embed, ["pending"])
                self.assertTrue(
                    wait_until(lambda: getattr(pool._capacity, "_value", -1) == 0)
                )

                with self.assertRaisesRegex(EmbeddingBusyError, "queue is full"):
                    pool.embed(["rejected"])

                controller.release.set()
                active.result(timeout=2)
                pending.result(timeout=2)

            self.assertEqual(pool.snapshot()["completed_requests"], 2)
            self.assertEqual(pool.snapshot()["failed_requests"], 0)

    def test_queue_timeout_releases_capacity_for_later_requests(self) -> None:
        controller = ActivityController(blocked=True)
        settings = make_settings(
            devices=("cuda:0",),
            max_pending_requests=1,
            queue_timeout_seconds=0.05,
        )

        with fake_pool(settings, controller=controller) as pool:
            with ThreadPoolExecutor(max_workers=1) as executor:
                active = executor.submit(pool.embed, ["active"])
                self.assertTrue(controller.wait_for_entries(1))

                with self.assertRaisesRegex(EmbeddingBusyError, "timed out"):
                    pool.embed(["timed-out"])

                controller.release.set()
                active.result(timeout=2)

            later = pool.embed(["later"])
            self.assertEqual(later.prompt_tokens, 1)
            self.assertEqual(pool.snapshot()["completed_requests"], 2)

    def test_failed_replica_is_restarted_then_returned_to_the_pool(self) -> None:
        settings = make_settings(
            devices=("cuda:0",),
            worker_restart_cooldown_seconds=0.01,
        )

        with fake_pool(settings) as pool:
            worker = pool.workers[0]
            worker.fail_embed_count = 1

            with self.assertRaisesRegex(EmbeddingServiceError, "inference failed"):
                pool.embed(["first"])

            self.assertTrue(worker.restarted.wait(timeout=2))
            result = pool.embed(["second"])

            self.assertEqual(result.prompt_tokens, 1)
            self.assertEqual(worker.restart_calls, 1)
            snapshot = pool.snapshot()
            self.assertEqual(snapshot["failed_requests"], 1)
            self.assertEqual(snapshot["completed_requests"], 1)
            self.assertEqual(snapshot["available_workers"], 1)

    def test_started_pool_rejects_new_work_after_close(self) -> None:
        settings = make_settings(devices=("cuda:0",))

        with fake_pool(settings) as pool:
            pool.start()
            pool.close()

            with self.assertRaisesRegex(EmbeddingServiceError, "shutting down"):
                pool.embed(["late"])


class ReplicaPoolWarmupTest(unittest.TestCase):
    def test_warms_replicas_sequentially(self) -> None:
        settings = make_settings(devices=("cuda:0", "cuda:1", "cuda:2"))

        with fake_pool(settings) as pool:
            pool.load()

            self.assertEqual(FakeReplica.load_order, [0, 1, 2])
            self.assertEqual(FakeReplica.max_load_active, 1)
            self.assertTrue(pool.loaded)
            self.assertEqual(pool.snapshot()["ready_workers"], 3)

    def test_partial_warmup_failure_leaves_a_degraded_usable_pool(self) -> None:
        settings = make_settings(
            devices=("cuda:0", "cuda:1"),
            worker_restart_cooldown_seconds=10,
        )

        with fake_pool(settings) as pool:
            pool.workers[1].fail_load = True
            pool.load()

            snapshot = pool.snapshot()
            self.assertEqual(snapshot["ready_workers"], 1)
            self.assertEqual(snapshot["failed_workers"], 1)
            self.assertFalse(pool.loaded)

    def test_all_warmup_failures_are_reported(self) -> None:
        settings = make_settings(
            devices=("cuda:0", "cuda:1"),
            worker_restart_cooldown_seconds=10,
        )

        with fake_pool(settings) as pool:
            for worker in pool.workers:
                worker.fail_load = True

            with self.assertRaisesRegex(
                EmbeddingServiceError,
                "all embedding workers failed",
            ):
                pool.load()

    def test_warmup_times_out_cleanly_when_a_replica_is_busy(self) -> None:
        controller = ActivityController(blocked=True)
        settings = make_settings(
            devices=("cuda:0", "cuda:1"),
            max_pending_requests=1,
            queue_timeout_seconds=0.05,
        )

        with fake_pool(settings, controller=controller) as pool:
            with ThreadPoolExecutor(max_workers=1) as executor:
                active = executor.submit(pool.embed, ["active"])
                self.assertTrue(controller.wait_for_entries(1))

                with self.assertRaisesRegex(
                    EmbeddingServiceError,
                    "timed out reserving",
                ):
                    pool.load()

                controller.release.set()
                active.result(timeout=2)

            self.assertEqual(pool.snapshot()["available_workers"], 2)
            self.assertEqual(pool.snapshot()["active_requests"], 0)


if __name__ == "__main__":
    unittest.main()
