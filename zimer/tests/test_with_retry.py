import unittest
import asyncio
from zimer import with_retry


class TestWithRetry(unittest.TestCase):
    def test_sync_success_after_retries(self):
        calls = {"count": 0}

        @with_retry(num_retries=3)
        def flaky():
            calls["count"] += 1
            if calls["count"] < 3:
                raise ValueError("Fail")
            return "ok"

        self.assertEqual(flaky(), "ok")
        # Should have been called exactly 3 times (2 failures + 1 success)
        self.assertEqual(calls["count"], 3)

    def test_sync_exhaust_retries(self):
        @with_retry(num_retries=2)
        def always_fail():
            raise RuntimeError("nope")

        with self.assertRaises(RuntimeError):
            always_fail()

    def test_async_success_after_retries(self):
        calls = {"count": 0}

        @with_retry(num_retries=3)
        async def flaky_async():
            calls["count"] += 1
            if calls["count"] < 2:
                raise ValueError("Fail async")
            return "ok"

        result = asyncio.run(flaky_async())
        self.assertEqual(result, "ok")
        self.assertEqual(calls["count"], 2)

    def test_invalid_parameters(self):
        # num_retries must be positive int
        with self.assertRaises(ValueError):
            @with_retry(num_retries=0)  # type: ignore[arg-type]
            def f1():
                pass

        with self.assertRaises(ValueError):
            @with_retry(num_retries=-1)
            def f2():
                pass

        with self.assertRaises(ValueError):
            @with_retry(backoff=-1)
            def f3():
                pass

        with self.assertRaises(ValueError):
            @with_retry(backoff_exponent=-0.5)
            def f4():
                pass


if __name__ == "__main__":
    unittest.main() 