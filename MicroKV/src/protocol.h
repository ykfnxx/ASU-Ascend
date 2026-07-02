#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace microkv {

constexpr std::size_t kKeyLength = 32;
constexpr std::size_t kRequestHeaderLength = 10;
constexpr std::size_t kResponseHeaderLength = 5;

enum class Command : std::uint8_t {
    Put = 0x01,
    Get = 0x02,
    Exists = 0x03,
    BatchPut = 0x04,
    BatchGet = 0x05,
    BatchExists = 0x06,
    Clear = 0x11,
    Size = 0x12,
};

enum class Status : std::uint8_t {
    Ok = 0,
    NotFound = 1,
    BadRequest = 2,
    InternalError = 3,
};

inline std::uint16_t LoadLe16(const std::uint8_t* data) {
    return static_cast<std::uint16_t>(data[0]) |
           (static_cast<std::uint16_t>(data[1]) << 8);
}

inline std::uint32_t LoadLe32(const std::uint8_t* data) {
    return static_cast<std::uint32_t>(data[0]) |
           (static_cast<std::uint32_t>(data[1]) << 8) |
           (static_cast<std::uint32_t>(data[2]) << 16) |
           (static_cast<std::uint32_t>(data[3]) << 24);
}

inline std::uint64_t LoadLe64(const std::uint8_t* data) {
    std::uint64_t value = 0;
    for (int i = 7; i >= 0; --i) {
        value = (value << 8) | data[i];
    }
    return value;
}

inline void StoreLe16(std::uint8_t* data, std::uint16_t value) {
    data[0] = static_cast<std::uint8_t>(value & 0xff);
    data[1] = static_cast<std::uint8_t>((value >> 8) & 0xff);
}

inline void StoreLe32(std::uint8_t* data, std::uint32_t value) {
    data[0] = static_cast<std::uint8_t>(value & 0xff);
    data[1] = static_cast<std::uint8_t>((value >> 8) & 0xff);
    data[2] = static_cast<std::uint8_t>((value >> 16) & 0xff);
    data[3] = static_cast<std::uint8_t>((value >> 24) & 0xff);
}

inline void StoreLe64(std::uint8_t* data, std::uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        data[i] = static_cast<std::uint8_t>((value >> (i * 8)) & 0xff);
    }
}

inline void AppendLe32(std::vector<std::uint8_t>* out, std::uint32_t value) {
    std::uint8_t data[4];
    StoreLe32(data, value);
    out->insert(out->end(), data, data + 4);
}

inline void AppendLe64(std::vector<std::uint8_t>* out, std::uint64_t value) {
    std::uint8_t data[8];
    StoreLe64(data, value);
    out->insert(out->end(), data, data + 8);
}

}  // namespace microkv
