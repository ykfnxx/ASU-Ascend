#include "server.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <iostream>
#include <limits>
#include <thread>

#include "protocol.h"

namespace microkv {
namespace {

constexpr std::uint32_t kMaxBatchCount = 1'000'000;

bool ReadExact(int fd, void* data, std::size_t len) {
    auto* cursor = static_cast<std::uint8_t*>(data);
    std::size_t remaining = len;
    while (remaining > 0) {
        const ssize_t n = ::recv(fd, cursor, remaining, 0);
        if (n == 0) {
            return false;
        }
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        cursor += n;
        remaining -= static_cast<std::size_t>(n);
    }
    return true;
}

bool WriteAll(int fd, const void* data, std::size_t len) {
    const auto* cursor = static_cast<const std::uint8_t*>(data);
    std::size_t remaining = len;
    while (remaining > 0) {
        const ssize_t n = ::send(fd, cursor, remaining, MSG_NOSIGNAL);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            return false;
        }
        cursor += n;
        remaining -= static_cast<std::size_t>(n);
    }
    return true;
}

bool ReadU32(int fd, std::uint32_t* value) {
    std::uint8_t buf[4];
    if (!ReadExact(fd, buf, sizeof(buf))) {
        return false;
    }
    *value = LoadLe32(buf);
    return true;
}

}  // namespace

Server::Server(std::string socket_path) : socket_path_(std::move(socket_path)) {}

Server::~Server() {
    if (server_fd_ >= 0) {
        ::close(server_fd_);
    }
    if (!socket_path_.empty()) {
        ::unlink(socket_path_.c_str());
    }
}

int Server::Run() {
    ::unlink(socket_path_.c_str());

    server_fd_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd_ < 0) {
        std::cerr << "socket failed: " << std::strerror(errno) << "\n";
        return 1;
    }

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    if (socket_path_.size() >= sizeof(addr.sun_path)) {
        std::cerr << "socket path too long: " << socket_path_ << "\n";
        return 1;
    }
    std::strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    if (::bind(server_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "bind failed: " << std::strerror(errno) << "\n";
        return 1;
    }

    if (::listen(server_fd_, 16) < 0) {
        std::cerr << "listen failed: " << std::strerror(errno) << "\n";
        return 1;
    }

    while (true) {
        const int client_fd = ::accept(server_fd_, nullptr, nullptr);
        if (client_fd < 0) {
            if (errno == EINTR) {
                continue;
            }
            std::cerr << "accept failed: " << std::strerror(errno) << "\n";
            return 1;
        }
        std::thread([this, client_fd]() {
            HandleClient(client_fd);
            ::close(client_fd);
        }).detach();
    }
}

bool Server::HandleClient(int client_fd) {
    while (true) {
        std::uint8_t header[kRequestHeaderLength];
        if (!ReadExact(client_fd, header, sizeof(header))) {
            return true;
        }
        if (!HandleRequest(client_fd, header)) {
            return false;
        }
    }
}

bool Server::HandleRequest(int client_fd, const std::uint8_t* header) {
    const auto command = static_cast<Command>(header[0]);
    const std::uint8_t type = header[1];
    const std::uint16_t key_len = LoadLe16(header + 4);
    const std::uint32_t val_len = LoadLe32(header + 6);

    if (command == Command::Put) {
        if (key_len != kKeyLength) {
            SendResponse(client_fd, Status::BadRequest, {});
            return false;
        }
        Key key;
        std::vector<std::uint8_t> value;
        if (!ReadKey(client_fd, &key) || !ReadBytes(client_fd, val_len, &value)) {
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(store_mutex_);
            store_.Put(type, key, std::move(value));
        }
        return SendResponse(client_fd, Status::Ok, {});
    }

    if (command == Command::Get) {
        if (key_len != kKeyLength || val_len != 0) {
            SendResponse(client_fd, Status::BadRequest, {});
            return false;
        }
        Key key;
        if (!ReadKey(client_fd, &key)) {
            return false;
        }
        std::optional<std::vector<std::uint8_t>> value;
        {
            std::lock_guard<std::mutex> lock(store_mutex_);
            value = store_.Get(type, key);
        }
        if (!value.has_value()) {
            return SendResponse(client_fd, Status::NotFound, {});
        }
        return SendResponse(client_fd, Status::Ok, *value);
    }

    if (command == Command::Exists) {
        if (key_len != kKeyLength || val_len != 0) {
            SendResponse(client_fd, Status::BadRequest, {});
            return false;
        }
        Key key;
        if (!ReadKey(client_fd, &key)) {
            return false;
        }
        bool exists = false;
        {
            std::lock_guard<std::mutex> lock(store_mutex_);
            exists = store_.Exists(type, key);
        }
        const std::vector<std::uint8_t> payload{static_cast<std::uint8_t>(exists ? 1 : 0)};
        return SendResponse(client_fd, Status::Ok, payload);
    }

    if (command == Command::BatchPut) {
        if (key_len != kKeyLength || val_len != 0) {
            SendResponse(client_fd, Status::BadRequest, {});
            return false;
        }
        std::uint32_t count = 0;
        if (!ReadU32(client_fd, &count) || count > kMaxBatchCount) {
            SendResponse(client_fd, Status::BadRequest, {});
            return false;
        }
        for (std::uint32_t i = 0; i < count; ++i) {
            Key key;
            std::uint32_t item_len = 0;
            std::vector<std::uint8_t> value;
            if (!ReadKey(client_fd, &key) ||
                !ReadU32(client_fd, &item_len) ||
                !ReadBytes(client_fd, item_len, &value)) {
                return false;
            }
            std::lock_guard<std::mutex> lock(store_mutex_);
            store_.Put(type, key, std::move(value));
        }
        return SendResponse(client_fd, Status::Ok, {});
    }

    if (command == Command::BatchGet) {
        if (key_len != kKeyLength || val_len != 0) {
            SendResponse(client_fd, Status::BadRequest, {});
            return false;
        }
        std::uint32_t count = 0;
        if (!ReadU32(client_fd, &count) || count > kMaxBatchCount) {
            SendResponse(client_fd, Status::BadRequest, {});
            return false;
        }
        std::vector<std::uint8_t> payload;
        AppendLe32(&payload, count);
        for (std::uint32_t i = 0; i < count; ++i) {
            Key key;
            if (!ReadKey(client_fd, &key)) {
                return false;
            }
            std::optional<std::vector<std::uint8_t>> value;
            {
                std::lock_guard<std::mutex> lock(store_mutex_);
                value = store_.Get(type, key);
            }
            payload.push_back(static_cast<std::uint8_t>(value.has_value() ? 1 : 0));
            if (value.has_value()) {
                AppendLe32(&payload, static_cast<std::uint32_t>(value->size()));
                payload.insert(payload.end(), value->begin(), value->end());
            } else {
                AppendLe32(&payload, 0);
            }
        }
        return SendResponse(client_fd, Status::Ok, payload);
    }

    if (command == Command::BatchExists) {
        if (key_len != kKeyLength || val_len != 0) {
            SendResponse(client_fd, Status::BadRequest, {});
            return false;
        }
        std::uint32_t count = 0;
        if (!ReadU32(client_fd, &count) || count > kMaxBatchCount) {
            SendResponse(client_fd, Status::BadRequest, {});
            return false;
        }
        std::vector<std::uint8_t> payload;
        AppendLe32(&payload, count);
        for (std::uint32_t i = 0; i < count; ++i) {
            Key key;
            if (!ReadKey(client_fd, &key)) {
                return false;
            }
            bool exists = false;
            {
                std::lock_guard<std::mutex> lock(store_mutex_);
                exists = store_.Exists(type, key);
            }
            payload.push_back(static_cast<std::uint8_t>(exists ? 1 : 0));
        }
        return SendResponse(client_fd, Status::Ok, payload);
    }

    if (command == Command::Clear) {
        if (key_len != 0 || val_len != 0) {
            SendResponse(client_fd, Status::BadRequest, {});
            return false;
        }
        {
            std::lock_guard<std::mutex> lock(store_mutex_);
            store_.Clear(type);
        }
        return SendResponse(client_fd, Status::Ok, {});
    }

    if (command == Command::Size) {
        if (key_len != 0 || val_len != 0) {
            SendResponse(client_fd, Status::BadRequest, {});
            return false;
        }
        std::vector<std::uint8_t> payload;
        std::size_t size = 0;
        {
            std::lock_guard<std::mutex> lock(store_mutex_);
            size = store_.Size(type);
        }
        AppendLe64(&payload, static_cast<std::uint64_t>(size));
        return SendResponse(client_fd, Status::Ok, payload);
    }

    SendResponse(client_fd, Status::BadRequest, {});
    return false;
}

bool Server::SendResponse(int client_fd, Status status, const std::vector<std::uint8_t>& payload) {
    if (payload.size() > std::numeric_limits<std::uint32_t>::max()) {
        return false;
    }
    std::uint8_t header[kResponseHeaderLength]{};
    header[0] = static_cast<std::uint8_t>(status);
    StoreLe32(header + 1, static_cast<std::uint32_t>(payload.size()));
    return WriteAll(client_fd, header, sizeof(header)) &&
           (payload.empty() || WriteAll(client_fd, payload.data(), payload.size()));
}

bool Server::ReadKey(int client_fd, Key* key) {
    return ReadExact(client_fd, key->bytes.data(), key->bytes.size());
}

bool Server::ReadBytes(int client_fd, std::uint32_t len, std::vector<std::uint8_t>* out) {
    out->resize(len);
    return len == 0 || ReadExact(client_fd, out->data(), out->size());
}

}  // namespace microkv
