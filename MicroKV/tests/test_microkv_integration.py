import struct
import unittest


from microkv_test_utils import MicroKVServer
from microkv import make_key


class MicroKVIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.server = MicroKVServer()
        self.server.__enter__()
        self.client = self.server.client
        assert self.client is not None

    def tearDown(self) -> None:
        if hasattr(self, "server"):
            self.server.close()

    def test_make_key_is_32_bytes_and_little_endian(self) -> None:
        key = make_key(42, layer_id=7, token_pos=9, cache_type=3)

        self.assertEqual(len(key), 32)
        self.assertEqual(key[:8], struct.pack("<Q", 42))
        self.assertEqual(key[8:16], b"\x00" * 8)
        self.assertEqual(key[16:20], struct.pack("<I", 7))
        self.assertEqual(key[20:24], struct.pack("<I", 9))
        self.assertEqual(key[24], 3)
        self.assertEqual(key[25:], b"\x00" * 7)

    def test_put_get_stores_opaque_bytes_without_slot_semantics(self) -> None:
        key = make_key("req-a", layer_id=1, token_pos=2, cache_type=0)
        value = struct.pack("<q?", 12345, True) + b"\x00raw-slot-payload"

        self.assertTrue(self.client.put(0, key, value))

        self.assertEqual(self.client.get(0, key), value)
        self.assertEqual(self.client.exists(0, key), True)

    def test_type_is_a_namespace_for_the_same_key(self) -> None:
        key = make_key("same-key", layer_id=3, token_pos=4, cache_type=0)

        self.client.put(0, key, b"type-zero")
        self.client.put(1, key, b"type-one")

        self.assertEqual(self.client.get(0, key), b"type-zero")
        self.assertEqual(self.client.get(1, key), b"type-one")
        self.assertEqual(self.client.size(0), 1)
        self.assertEqual(self.client.size(1), 1)

    def test_batch_get_and_exists_preserve_request_order(self) -> None:
        keys = [make_key("batch", 0, i) for i in range(4)]
        self.client.batch_put(0, [keys[0], keys[2], keys[3]], [b"v0", b"v2", b"v3"])

        self.assertEqual(
            self.client.batch_get(0, keys),
            [b"v0", None, b"v2", b"v3"],
        )
        self.assertEqual(
            self.client.batch_exists(0, keys),
            [True, False, True, True],
        )

    def test_clear_is_per_type(self) -> None:
        key = make_key("clear", layer_id=0, token_pos=0)
        self.client.put(0, key, b"left")
        self.client.put(1, key, b"right")

        self.client.clear(0)

        self.assertIsNone(self.client.get(0, key))
        self.assertEqual(self.client.get(1, key), b"right")
        self.assertEqual(self.client.size(0), 0)
        self.assertEqual(self.client.size(1), 1)


if __name__ == "__main__":
    unittest.main()
