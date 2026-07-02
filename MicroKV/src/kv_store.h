#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>
#include <vector>

#include "protocol.h"

namespace microkv {

struct Key {
    std::array<std::uint8_t, kKeyLength> bytes{};

    bool operator==(const Key& other) const {
        return bytes == other.bytes;
    }
};

struct KeyHash {
    std::size_t operator()(const Key& key) const noexcept;
};

class KVStore {
public:
    bool Put(std::uint8_t type, const Key& key, std::vector<std::uint8_t> value);
    std::optional<std::vector<std::uint8_t>> Get(std::uint8_t type, const Key& key) const;
    bool Exists(std::uint8_t type, const Key& key) const;
    void Clear(std::uint8_t type);
    std::size_t Size(std::uint8_t type) const;

private:
    using TypeStore = std::unordered_map<Key, std::vector<std::uint8_t>, KeyHash>;
    std::unordered_map<std::uint8_t, TypeStore> stores_;
};

}  // namespace microkv
