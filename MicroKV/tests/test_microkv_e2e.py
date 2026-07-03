import struct
import unittest


from microkv_test_utils import MicroKVServer
from microkv import DSA_INDEX, KV_ATTENTION_K, KV_ATTENTION_V, KV_MLA_TOKEN, make_key


SLOT_RECORD = struct.Struct("<qI")


def pack_caller_record(slot_id: int, tensor_bytes: bytes) -> bytes:
    return SLOT_RECORD.pack(slot_id, len(tensor_bytes)) + tensor_bytes


def unpack_caller_record(value: bytes) -> tuple[int, bytes]:
    slot_id, tensor_len = SLOT_RECORD.unpack(value[: SLOT_RECORD.size])
    tensor = value[SLOT_RECORD.size :]
    if len(tensor) != tensor_len:
        raise ValueError("corrupt caller record")
    return slot_id, tensor


class MicroKVE2ETest(unittest.TestCase):
    def test_phase1_like_batch_read_keeps_slot_semantics_in_caller(self) -> None:
        with MicroKVServer() as server:
            client = server.client
            assert client is not None
            stored = {
                0: (100, b"k-token-0"),
                1: (101, b"k-token-1"),
                3: (130, b"k-token-3"),
            }
            keys_by_token = {
                token: make_key("req-phase1", layer_id=2, token_pos=token, cache_type=KV_ATTENTION_K)
                for token in [0, 1, 2, 3]
            }

            for token, (slot_id, tensor) in stored.items():
                client.put(KV_ATTENTION_K, keys_by_token[token], pack_caller_record(slot_id, tensor))

            simulated_topk_tokens = [3, 2, 1, 0]
            values = client.batch_get(KV_ATTENTION_K, [keys_by_token[token] for token in simulated_topk_tokens])
            exists_mask = [value is not None for value in values]
            parsed = [unpack_caller_record(value) if value is not None else None for value in values]

            self.assertEqual(exists_mask, [True, False, True, True])
            self.assertEqual(parsed, [(130, b"k-token-3"), None, (101, b"k-token-1"), (100, b"k-token-0")])

    def test_multiple_clients_observe_the_same_server_state(self) -> None:
        with MicroKVServer() as server:
            writer = server.client
            assert writer is not None
            reader = server.new_client()
            try:
                key = make_key("shared-server", layer_id=0, token_pos=0)

                writer.put(KV_ATTENTION_K, key, b"written-by-first-client")

                self.assertEqual(reader.get(KV_ATTENTION_K, key), b"written-by-first-client")
                self.assertTrue(reader.exists(KV_ATTENTION_K, key))

                reader.clear(KV_ATTENTION_K)

                self.assertIsNone(writer.get(KV_ATTENTION_K, key))
                self.assertFalse(writer.exists(KV_ATTENTION_K, key))
            finally:
                reader.close()

    def test_overwrite_zero_length_and_large_payload_roundtrip(self) -> None:
        with MicroKVServer() as server:
            client = server.client
            assert client is not None
            key = make_key("opaque-bytes", layer_id=4, token_pos=5)
            large_payload = bytes(i % 251 for i in range(512 * 1024))

            client.put(KV_ATTENTION_K, key, b"first")
            self.assertEqual(client.size(KV_ATTENTION_K), 1)

            client.put(KV_ATTENTION_K, key, b"")
            self.assertEqual(client.get(KV_ATTENTION_K, key), b"")
            self.assertTrue(client.exists(KV_ATTENTION_K, key))
            self.assertEqual(client.size(KV_ATTENTION_K), 1)

            client.put(KV_ATTENTION_K, key, large_payload)

            self.assertEqual(client.get(KV_ATTENTION_K, key), large_payload)
            self.assertEqual(client.size(KV_ATTENTION_K), 1)

    def test_restart_drops_in_memory_state(self) -> None:
        key = make_key("restart", layer_id=0, token_pos=1)
        with MicroKVServer() as server:
            client = server.client
            assert client is not None
            client.put(KV_ATTENTION_K, key, b"not-persistent")
            self.assertEqual(client.get(KV_ATTENTION_K, key), b"not-persistent")

        with MicroKVServer() as server:
            client = server.client
            assert client is not None
            self.assertIsNone(client.get(KV_ATTENTION_K, key))
            self.assertEqual(client.size(KV_ATTENTION_K), 0)

    def test_cache_type_namespaces_are_independent_for_batched_reads(self) -> None:
        with MicroKVServer() as server:
            client = server.client
            assert client is not None
            key_a = make_key("type-space", layer_id=1, token_pos=0)
            key_b = make_key("type-space", layer_id=1, token_pos=1)

            client.batch_put(KV_ATTENTION_K, [key_a, key_b], [b"k-a", b"k-b"])
            client.batch_put(KV_ATTENTION_V, [key_a, key_b], [b"v-a", b"v-b"])
            client.batch_put(DSA_INDEX, [key_a], [struct.pack("<I", 77)])

            self.assertEqual(client.batch_get(KV_ATTENTION_K, [key_a, key_b]), [b"k-a", b"k-b"])
            self.assertEqual(client.batch_get(KV_ATTENTION_V, [key_a, key_b]), [b"v-a", b"v-b"])
            self.assertEqual(client.batch_get(DSA_INDEX, [key_a, key_b]), [struct.pack("<I", 77), None])

    def test_mla_token_record_roundtrip_uses_opaque_value(self) -> None:
        with MicroKVServer() as server:
            client = server.client
            assert client is not None
            key = make_key("mla-req", layer_id=12, token_pos=34, cache_type=KV_MLA_TOKEN)
            value = b"mla-record-header" + b"k-nope-bytes" + b"k-pe-bytes"

            client.batch_put(KV_MLA_TOKEN, [key], [value])

            self.assertEqual(client.batch_get(KV_MLA_TOKEN, [key]), [value])


if __name__ == "__main__":
    unittest.main()
