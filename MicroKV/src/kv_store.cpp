#include "kv_store.h"

namespace microkv {

std::size_t KeyHash::operator()(const Key& key) const noexcept {
    std::size_t hash = 1469598103934665603ull;
    for (std::uint8_t byte : key.bytes) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

bool KVStore::Put(std::uint8_t type, const Key& key, std::vector<std::uint8_t> value) {
    stores_[type][key] = std::move(value);
    return true;
}

std::optional<std::vector<std::uint8_t>> KVStore::Get(std::uint8_t type, const Key& key) const {
    const auto type_it = stores_.find(type);
    if (type_it == stores_.end()) {
        return std::nullopt;
    }

    const auto value_it = type_it->second.find(key);
    if (value_it == type_it->second.end()) {
        return std::nullopt;
    }

    return value_it->second;
}

bool KVStore::Exists(std::uint8_t type, const Key& key) const {
    const auto type_it = stores_.find(type);
    if (type_it == stores_.end()) {
        return false;
    }
    return type_it->second.find(key) != type_it->second.end();
}

void KVStore::Clear(std::uint8_t type) {
    stores_.erase(type);
}

std::size_t KVStore::Size(std::uint8_t type) const {
    const auto type_it = stores_.find(type);
    if (type_it == stores_.end()) {
        return 0;
    }
    return type_it->second.size();
}

}  // namespace microkv
