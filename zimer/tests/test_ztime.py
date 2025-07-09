import unittest
import time
import asyncio
from ztime import ztime

class TestZTime(unittest.TestCase):

    def test_sync_function(self):
        @ztime
        def sync_test_func():
            time.sleep(0.1)
            return "done"

        self.assertEqual(sync_test_func(), "done")

    def test_async_function(self):
        @ztime
        async def async_test_func():
            await asyncio.sleep(0.1)
            return "done"

        self.assertEqual(asyncio.run(async_test_func()), "done")

    def test_sync_with_repeats(self):
        @ztime(repeats=3)
        def sync_test_func_repeats():
            time.sleep(0.1)
            return "done"

        self.assertEqual(sync_test_func_repeats(), "done")

    def test_async_with_repeats(self):
        @ztime(repeats=3)
        async def async_test_func_repeats():
            await asyncio.sleep(0.1)
            return "done"

        self.assertEqual(asyncio.run(async_test_func_repeats()), "done")

    def test_invalid_repeats_value(self):
        with self.assertRaises(ValueError):
            @ztime(repeats=0)
            def test_func():
                pass

        with self.assertRaises(ValueError):
            @ztime(repeats=-1)
            def test_func2():
                pass

        with self.assertRaises(ValueError):
            @ztime(repeats="a")
            def test_func3():
                pass

if __name__ == '__main__':
    unittest.main()