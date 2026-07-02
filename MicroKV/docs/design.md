# MicroKV 设计方案

## 1. 背景与目标

为配合 vllm-ascend KV Cache Offload 的 Phase 1 验证工作，需要一个轻量级的内存 KV 语义存储服务。

- **服务定位**：仅用于验证，不会进入最终生产路径。
- **核心功能**：按 `(cache_type, key)` 存储和读取 opaque bytes；slot_id、tensor layout、exists mask 等业务语义由调用方定义和解析。
- **部署环境**：单机环境，无跨节点需求。

## 2. 设计约束

| 约束 | 说明 |
|---|---|
| 仅验证使用 | 无需持久化、无需一致性、无需高可用 |
| Key 定长 | `req_id/layer/token` 拼接成 32 字节定长 key |
| Value 透明 | MicroKV 不解释 value 内容，也不生成 slot_id |
| 读写分离 | 写入阶段与读取阶段严格分离，无并发写读 |
| 独立进程 | 与 vllm-ascend 解耦，通过 IPC 通信 |
| CPU 不参与核心计算 | Phase 1 compare 路径允许 CPU 参与，因为只是旁路验证 |

## 3. 部署形态

采用 **C++ 独立进程 + Unix domain socket + Python 客户端** 方案。

```
┌─────────────────┐         Unix domain socket         ┌─────────────────┐
│   vllm-ascend   │  ─────────────────────────────────► │    kv_stored    │
│   (Python)      │   Put/Get/Exists/Batch/Clear/Size    │  (C++ process)  │
│                 │ ◄─────────────────────────────────  │                 │
└─────────────────┘         binary response              └─────────────────┘
```

### 3.1 为什么选独立进程

- 与 vllm-ascend 完全解耦，不修改其构建系统；
- 验证工具独立演进，崩溃不影响推理进程；
- 单机环境下 Unix socket 延迟足够低；
- Phase 1 对性能无要求，开发简单优先。

### 3.2 为什么不选 pybind

pybind11 只能将 C++ 接口暴露给**同进程** Python。独立进程必须使用 IPC，因此 Python 端使用标准 socket 客户端，不引入 pybind。

## 4. Key 设计

定长 32 字节，小端字节序：

```
|  req_index (16B)  |  layer_id (4B)  |  token_pos (4B)  |  cache_type (1B)  |  reserved (7B)  |
```

| 字段 | 说明 |
|---|---|
| `req_index` | request 的唯一索引，可以是 `req_id` 的 hash 或 vllm 内部 req_idx，占 16 字节 |
| `layer_id` | 模型层号，从 `layer_name` 解析，例如 `model.layers.12.self_attn` → `12` |
| `token_pos` | token 在完整序列中的位置，使用完整序列位置而非 block 内 offset |
| `cache_type` | cache 类型编号，Phase 1 只使用 `0` |
| `reserved` | 预留扩展 |

## 5. Cache Type 设计

Phase 1 约定以下 cache type 编号：

```cpp
enum class KVCacheType : uint8_t {
    KV_ATTENTION_K = 0,   // 一个 token 的 K 数据
};
```

后续可扩展：

```cpp
KV_ATTENTION_V = 1,   // 一个 token 的 V 数据
DSA_INDEX = 2,        // 一个 token 的 k_li
DSA_INDEX_SCALE = 3,  // 一个 token 的 scale
```

`cache_type` 是 KV namespace。服务端不需要注册 value size，也不解释具体类型；Python 客户端暴露上述常量便于调用方统一使用。

## 6. 通信协议

使用固定头部的二进制协议，基于 Unix domain socket 传输。

### 6.1 请求格式

```
┌─────────┬──────────┬──────────┬────────────┬─────────┬─────────┐
│ cmd(1B) │ type(1B) │ reserved │ key_len(2B)│ val_len │  key... │
│         │          │  (2B)    │            │  (4B)   │         │
└─────────┴──────────┴──────────┴────────────┴─────────┴─────────┘
```

- `cmd`：操作类型
- `type`：KVCacheType
- `key_len`：固定为 32
- `val_len`：value 长度，Put 时必填；Get/Exists/Batch/Clear/Size 为 0

### 6.2 响应格式

```
┌────────────┬────────────┬─────────┐
│ status(1B) │ val_len(4B)│ value...│
└────────────┴────────────┴─────────┘
```

- `status`：`0` 成功，`1` 表示 key 不存在，其他非 0 表示协议或内部错误
- `val_len`：返回 payload 长度；Get 返回原始 value，Exists 返回 1 字节 bool，Size 返回 8 字节 uint64

### 6.3 操作码

| cmd | 含义 |
|---|---|
| `0x01` | Put |
| `0x02` | Get |
| `0x03` | Exists |
| `0x04` | BatchPut |
| `0x05` | BatchGet |
| `0x06` | BatchExists |
| `0x11` | Clear |
| `0x12` | Size |

## 7. C++ 服务端接口

```cpp
namespace microkv {

constexpr size_t kKeyLength = 32;

enum class KVCacheType : uint8_t {
    KV_ATTENTION_K = 0,
};

struct Key {
    std::array<uint8_t, kKeyLength> bytes{};
    bool operator==(const Key& other) const;
};

struct KeyHash {
    size_t operator()(const Key& k) const noexcept;
};

class KVStore {
public:
    bool Put(uint8_t type, const Key& key, std::vector<uint8_t> value);
    std::optional<std::vector<uint8_t>> Get(uint8_t type, const Key& key) const;
    bool Exists(uint8_t type, const Key& key) const;
    void Clear(uint8_t type);
    size_t Size(uint8_t type) const;
};

}  // namespace microkv
```

## 8. Python 客户端接口

```python
class KVStoreClient:
    def __init__(self, socket_path: str = "/tmp/microkv.sock"): ...

    def put(self, cache_type: int, key: bytes, value: bytes) -> bool: ...
    def get(self, cache_type: int, key: bytes) -> bytes | None: ...
    def exists(self, cache_type: int, key: bytes) -> bool: ...

    def batch_put(self, cache_type: int, keys: list[bytes], values: list[bytes]) -> bool: ...
    def batch_get(self, cache_type: int, keys: list[bytes]) -> list[bytes | None]: ...
    def batch_exists(self, cache_type: int, keys: list[bytes]) -> list[bool]: ...

    def clear(self, cache_type: int) -> None: ...
    def size(self, cache_type: int) -> int: ...
```

### 8.1 Key 构造 Helper

```python
def make_key(req_index: int | str,
             layer_id: int,
             token_pos: int,
             cache_type: int = 0) -> bytes:
    if isinstance(req_index, str):
        req_bytes = hashlib.sha256(req_index.encode()).digest()[:16]
    else:
        req_bytes = struct.pack("<Q", req_index) + b"\x00" * 8
    layer = struct.pack("<I", layer_id)
    token = struct.pack("<I", token_pos)
    ctype = struct.pack("<B", cache_type) + b"\x00" * 7
    return req_bytes + layer + token + ctype  # 32 bytes
```

## 9. 非目标

- MicroKV 不实现 lookup 算子语义；
- MicroKV 不生成或解释 slot_id；
- MicroKV 不实现 H2D/D2H 监听；
- MicroKV 不做持久化、淘汰、量化或并发一致性。

## 10. 目录结构

```text
ASU-Ascend/MicroKV/
├── docs/
│   └── design.md              # 本文档
├── CMakeLists.txt
├── Makefile
├── pyproject.toml
├── src/
│   ├── main.cpp               # kv_stored 入口
│   ├── server.h/.cpp          # Unix socket server
│   ├── kv_store.h/.cpp        # 核心 KV 实现
│   └── protocol.h             # 请求/响应格式定义
└── python/
    └── microkv/
        ├── __init__.py
        └── client.py          # Python 客户端
```

## 11. Phase 1 实现计划

1. 实现 `KVStore` 核心（Put/Get/Exists/Batch/Clear/Size）；
2. 实现 Unix socket server，解析二进制协议；
3. 实现 `kv_stored` 可执行文件；
4. 实现 Python 客户端 `KVStoreClient`；
5. 实现 key 构造 helper；
6. 编写无 NPU 的单元测试；

## 12. 与 vllm-ascend 的集成方式

Phase 1 验证流程：

```
准备阶段：
    从原始 KV cache 按 token 抽取 K 数据
    key = make_key(req_index, layer_id, token_pos)
    client.put(KV_ATTENTION_K, key, caller_defined_value)

Forward 旁路：
    NPU lookup 算子返回 topk_indices
    Python D2H topk_indices
    根据 seq_lens / req_ids 构造 keys [B, K, 32]
    client.batch_get(...) → list[bytes | None]
    调用方解析 bytes 中的 slot_id / tensor 数据，并自行生成 exists mask
```

## 13. 后续可扩展方向

- 增加 `KV_ATTENTION_V`、`DSA_INDEX`、`DSA_INDEX_SCALE` 等 cache type；
- 替换 socket 为共享内存，降低大 value 传输延迟；
- 如需跨节点，升级为 RDMA/TCP server。
